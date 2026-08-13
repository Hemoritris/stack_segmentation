"""箱子顶面提取（RANSAC 平面 + 法向量检查）。"""

from __future__ import annotations

from ..core.types import BoxTopPlane


def fit_top_plane(
    points_world,
    normal_threshold: float = 0.95,
    distance_threshold: float = 0.005,
) -> BoxTopPlane:
    """对单箱点云拟合近似水平顶面，法向量应约等于 [0, 0, 1]。"""
    raise NotImplementedError("TODO(M6): 离群点过滤 + RANSAC + 法向量检查")

