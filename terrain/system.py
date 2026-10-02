"""
pairwise-terrain-system - TerrainSystem

Fixed implementation:
- LOD: distance-based selection, geomorph transitions, crack stitching
- Heightmap: bilinear sampling, bounds checking, 16-bit quantized storage
- Sculpting: brush falloff, normal recompute, undo stack
- Texturing: height+slope blending, smooth transitions, texture array
- Collision: swept heightfield collider, detail objects, async queries
"""

from __future__ import annotations

import math
from array import array
from concurrent.futures import Future, ThreadPoolExecutor


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class TerrainSystem:
    HEIGHT_SCALE = 0.01  # uint16 quantization step in metres

    def __init__(self, size: int = 65, max_lod: int = 4, lod_distance: float = 64.0):
        self.size = size
        self.max_lod = max_lod
        self.lod_distance = lod_distance
        self.morph_band = 0.25  # fraction of each LOD ring used for geomorphing
        # Compressed heightmap: uint16 quantization (2 bytes/vertex vs 4 for float32).
        self._heights = array("H", [0] * (size * size))
        self._normals = [(0.0, 1.0, 0.0)] * (size * size)
        self._undo_stack: list[dict] = []
        self._texture_array: list[dict] = [
            {"name": "grass", "max_slope": 0.7, "min_height": 0.0, "max_height": 60.0},
            {"name": "rock", "max_slope": 1.0, "min_height": 0.0, "max_height": 120.0},
            {"name": "snow", "max_slope": 0.5, "min_height": 60.0, "max_height": 1000.0},
        ]
        self._collision_details: list[dict] = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="terrain-collision")
        self.state = {}

    # ------------------------------------------------------------------ LOD

    def calculate_lod(self, chunk_x: float = 0.0, chunk_z: float = 0.0,
                      camera: tuple[float, float, float] | None = None,
                      neighbor_lods: dict[str, int] | None = None) -> dict:
        """Distance-based LOD with geomorph transition and crack stitching."""
        if camera is None:
            camera = (0.0, 0.0, 0.0)
        dx = chunk_x - camera[0]
        dz = chunk_z - camera[2]
        distance = math.hypot(dx, dz)

        # Distance judgement: continuous LOD, snapped to an integer level.
        continuous = distance / max(self.lod_distance, 1e-6)
        level = min(int(continuous), self.max_lod)
        # Geomorph factor: 0 at the inner edge of a ring, 1 at the outer edge,
        # so vertices blend smoothly instead of popping between levels.
        morph = min(max(continuous - level, 0.0), 1.0)
        if level >= self.max_lod:
            morph = 0.0
        morph = _smoothstep(1.0 - self.morph_band, 1.0, morph) if morph > 0 else 0.0

        step = 1 << level
        # Crack stitching: for each edge shared with a lower-LOD neighbour,
        # collapse every other vertex onto the coarser grid so no T-junction
        # gaps open up between chunks.
        stitch = {}
        for edge in ("north", "south", "east", "west"):
            neighbor = (neighbor_lods or {}).get(edge, level)
            if neighbor > level:
                coarse = 1 << neighbor
                stitch[edge] = [i for i in range(self.size)
                                if (i * step) % coarse != 0]
            else:
                stitch[edge] = []

        return {
            "level": level,
            "step": step,
            "morph": morph,
            "distance": distance,
            "stitch": stitch,
            "skirt_depth": 0.5 * step,  # geometry skirt hides residual seams
        }

    # ------------------------------------------------------------- Heightmap

    def _index(self, x: int, z: int) -> int:
        return z * self.size + x

    def get_height(self, x: int, z: int) -> float:
        return self._heights[self._index(x, z)] * self.HEIGHT_SCALE

    def set_height(self, x: int, z: int, height: float) -> None:
        q = int(round(height / self.HEIGHT_SCALE))
        self._heights[self._index(x, z)] = max(0, min(65535, q))

    def sample_height(self, x: float = 0.0, z: float = 0.0) -> float:
        """Bilinear height sampling with boundary clamping (no nearest-neighbour
        jaggies, no out-of-bounds crashes)."""
        # Boundary check: clamp into the valid heightfield range.
        x = min(max(float(x), 0.0), self.size - 1.0)
        z = min(max(float(z), 0.0), self.size - 1.0)

        x0, z0 = int(x), int(z)
        x1 = min(x0 + 1, self.size - 1)
        z1 = min(z0 + 1, self.size - 1)
        fx, fz = x - x0, z - z0

        h00 = self.get_height(x0, z0)
        h10 = self.get_height(x1, z0)
        h01 = self.get_height(x0, z1)
        h11 = self.get_height(x1, z1)
        # Bilinear interpolation.
        top = h00 + (h10 - h00) * fx
        bottom = h01 + (h11 - h01) * fx
        return top + (bottom - top) * fz

    # ------------------------------------------------------------ Sculpting

    def sculpt_terrain(self, x: float = 0.0, z: float = 0.0, radius: float = 8.0,
                       strength: float = 1.0, record_undo: bool = True) -> dict:
        """Sculpt with smooth brush falloff, normal recompute and undo support."""
        cx = min(max(float(x), 0.0), self.size - 1.0)
        cz = min(max(float(z), 0.0), self.size - 1.0)
        radius = max(float(radius), 1e-3)

        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(self.size - 1, int(math.ceil(cx + radius)))
        z0 = max(0, int(math.floor(cz - radius)))
        z1 = min(self.size - 1, int(math.ceil(cz + radius)))

        affected = []
        for vz in range(z0, z1 + 1):
            for vx in range(x0, x1 + 1):
                dist = math.hypot(vx - cx, vz - cz)
                if dist > radius:
                    continue
                # Brush falloff: smooth cosine, 1 at centre -> 0 at the rim,
                # so strokes blend instead of leaving hard edges.
                falloff = 0.5 * (1.0 + math.cos(math.pi * dist / radius))
                old = self.get_height(vx, vz)
                new = old + strength * falloff
                affected.append((vx, vz, old))
                self.set_height(vx, vz, new)

        if record_undo:
            self._undo_stack.append({"region": (x0, z0, x1, z1), "heights": affected})

        # Recompute normals for the touched region (plus a 1-vertex halo) so
        # lighting stays correct after sculpting.
        self._update_normals(max(0, x0 - 1), max(0, z0 - 1),
                             min(self.size - 1, x1 + 1), min(self.size - 1, z1 + 1))
        return {"affected": len(affected), "region": (x0, z0, x1, z1)}

    def undo_sculpt(self) -> bool:
        """Restore the heightmap to its state before the last sculpt stroke."""
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        for vx, vz, old in entry["heights"]:
            self.set_height(vx, vz, old)
        x0, z0, x1, z1 = entry["region"]
        self._update_normals(max(0, x0 - 1), max(0, z0 - 1),
                             min(self.size - 1, x1 + 1), min(self.size - 1, z1 + 1))
        return True

    def _update_normals(self, x0: int, z0: int, x1: int, z1: int) -> None:
        for vz in range(z0, z1 + 1):
            for vx in range(x0, x1 + 1):
                hl = self.get_height(max(vx - 1, 0), vz)
                hr = self.get_height(min(vx + 1, self.size - 1), vz)
                hd = self.get_height(vx, max(vz - 1, 0))
                hu = self.get_height(vx, min(vz + 1, self.size - 1))
                nx, ny, nz = hl - hr, 2.0, hd - hu
                length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                self._normals[self._index(vx, vz)] = (nx / length, ny / length, nz / length)

    def get_normal(self, x: int, z: int) -> tuple[float, float, float]:
        x = min(max(int(x), 0), self.size - 1)
        z = min(max(int(z), 0), self.size - 1)
        return self._normals[self._index(x, z)]

    # ------------------------------------------------------------- Texturing

    def blend_textures(self, x: float = 0.0, z: float = 0.0,
                       transition: float = 0.15) -> dict:
        """Blend weights driven by height AND slope, with smooth transitions,
        sampled from a single texture array (one lookup, not one per texture)."""
        height = self.sample_height(x, z)
        nx, ny, nz = self.get_normal(int(round(min(max(x, 0.0), self.size - 1.0))),
                                     int(round(min(max(z, 0.0), self.size - 1.0))))
        slope = 1.0 - ny  # 0 = flat, 1 = vertical cliff

        raw = []
        for layer in self._texture_array:
            # Smooth height and slope bands -> gradual transitions, no hard edges.
            w_height = (1.0 - _smoothstep(layer["max_height"] - transition * 100,
                                          layer["max_height"], height))
            w_height *= _smoothstep(layer["min_height"] - transition * 100,
                                    layer["min_height"], height)
            w_slope = 1.0 - _smoothstep(layer["max_slope"] - transition,
                                        layer["max_slope"], slope)
            raw.append(max(0.0, w_height * w_slope))

        total = sum(raw) or 1.0
        weights = {layer["name"]: w / total for layer, w in zip(self._texture_array, raw)}
        return {
            "weights": weights,
            "height": height,
            "slope": slope,
            # Texture array: all layers sampled together in one pass.
            "texture_array": [layer["name"] for layer in self._texture_array],
        }

    # ------------------------------------------------------------- Collision

    def add_collision_detail(self, x: float, z: float, radius: float, height: float,
                             kind: str = "rock") -> None:
        """Register a detail collider (rock, pit, ...) on top of the heightfield."""
        self._collision_details.append(
            {"x": float(x), "z": float(z), "radius": float(radius),
             "height": float(height), "kind": kind})

    def _collision_surface_height(self, x: float, z: float) -> float:
        """Heightfield plus detail colliders (rocks raise it, pits lower it)."""
        surface = self.sample_height(x, z)
        for detail in self._collision_details:
            dist = math.hypot(x - detail["x"], z - detail["z"])
            if dist <= detail["radius"]:
                if detail["kind"] == "pit":
                    surface = min(surface, surface - detail["height"])
                else:
                    surface = max(surface, surface + detail["height"])
        return surface

    def _swept_collision(self, start, end, radius, steps) -> dict:
        """Swept segment query: fast movers can't tunnel through the terrain."""
        sx, sy, sz = start
        ex, ey, ez = end
        for i in range(steps + 1):
            t = i / steps
            px, py, pz = sx + (ex - sx) * t, sy + (ey - sy) * t, sz + (ez - sz) * t
            surface = self._collision_surface_height(px, pz)
            if py - radius <= surface:
                return {"hit": True, "position": (px, surface + radius, pz),
                        "t": t, "surface_height": surface}
        return {"hit": False, "position": (ex, ey, ez), "t": 1.0,
                "surface_height": self._collision_surface_height(ex, ez)}

    def get_terrain_collision(self, start: tuple[float, float, float] = (0.0, 1.0, 0.0),
                              end: tuple[float, float, float] | None = None,
                              radius: float = 0.5, steps: int = 16,
                              wait: bool = True):
        """Async swept collision query against the heightfield collider plus
        detail colliders. Returns a Future; pass wait=True for a sync result."""
        if end is None:
            end = start
        future: Future = self._executor.submit(
            self._swept_collision, tuple(start), tuple(end), float(radius), int(steps))
        return future.result() if wait else future
