"""
pairwise-terrain-system - TerrainSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class TerrainSystem:
    def __init__(self):
        self.state = {}

    def calculate_lod(self, *args, **kwargs):
        """BUG: 地形LOD不做裂缝修补，不同LOD级别相邻时有缝隙"""
        return None

    def sample_height(self, *args, **kwargs):
        """BUG: 高度图采样只做最近邻，不做双线性插值"""
        return None

    def sculpt_terrain(self, *args, **kwargs):
        """BUG: 地形雕刻只改高度，不更新法线"""
        return None

    def blend_textures(self, *args, **kwargs):
        """BUG: 地形贴图混合只按高度，不按坡度和法线"""
        return None

    def get_terrain_collision(self, *args, **kwargs):
        """BUG: 地形碰撞只用高度图，不做碰撞体"""
        return None

