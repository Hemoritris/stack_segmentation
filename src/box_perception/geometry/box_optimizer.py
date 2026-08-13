"""固定尺寸先验约束拟合（只优化 x, y, yaw）。"""

from __future__ import annotations


class BoxOptimizer:
    """用已知 L0, W0 作为先验，优化 x, y, yaw。

    目标：E = E_point + λ E_edge + μ E_mask。V1 可先只实现 E_point。
    """

    def __init__(self, length: float, width: float):
        self.length = length
        self.width = width

    def optimize(self, top_points_xy, init_pose):
        raise NotImplementedError("TODO(M8): 建立最小二乘 / 非线性优化")

