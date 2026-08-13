"""世界高度图构建。

职责：把 world 点云投影到 XY 栅格，得到 H(x, y)=z，用于时序差分。
"""

from __future__ import annotations


class HeightMap:
    def __init__(self, grid_size_m: float = 0.005, aggregation: str = "median"):
        self.grid_size_m = grid_size_m
        self.aggregation = aggregation

    def build(self, points_world):
        raise NotImplementedError("TODO(M2): 投影、栅格聚合与空洞处理")

