import pytest
from terrain.system import TerrainSystem


class TestTerrainSystem:

    def test_calculate_lod(self):
        """地形LOD不做裂缝修补，不同LOD级别相邻时有缝隙"""
        obj = TerrainSystem()
        result = obj.calculate_lod()
        self.assertIsNotNone(result)

    def test_sample_height(self):
        """高度图采样只做最近邻，不做双线性插值"""
        obj = TerrainSystem()
        result = obj.sample_height()
        self.assertIsNotNone(result)

    def test_sculpt_terrain(self):
        """地形雕刻只改高度，不更新法线"""
        obj = TerrainSystem()
        result = obj.sculpt_terrain()
        self.assertIsNotNone(result)

    def test_blend_textures(self):
        """地形贴图混合只按高度，不按坡度和法线"""
        obj = TerrainSystem()
        result = obj.blend_textures()
        self.assertIsNotNone(result)

    def test_get_terrain_collision(self):
        """地形碰撞只用高度图，不做碰撞体"""
        obj = TerrainSystem()
        result = obj.get_terrain_collision()
        self.assertIsNotNone(result)

