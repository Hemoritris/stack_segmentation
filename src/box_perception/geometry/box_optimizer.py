"""固定尺寸先验约束拟合（M8）：只优化 x, y, yaw。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class BoxOptimizer:
    """用已知 L0, W0 作为先验，优化 x, y, yaw。

    V1 只实现 E_point：点到固定尺寸矩形的“越界量”平方和，点落在矩形内则误差为 0。
    """

    def __init__(self, length: float, width: float):
        if length <= 0 or width <= 0:
            raise ValueError("length / width 必须为正")
        self.length = float(length)
        self.width = float(width)
        self.half_l = self.length / 2.0
        self.half_w = self.width / 2.0

    def _error(self, theta, points_xy: np.ndarray) -> float:
        x, y, yaw = theta
        c, s = np.cos(yaw), np.sin(yaw)
        dx = points_xy[:, 0] - x
        dy = points_xy[:, 1] - y
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        over_x = np.maximum(0.0, np.abs(lx) - self.half_l)
        over_y = np.maximum(0.0, np.abs(ly) - self.half_w)
        return float(np.sum(over_x**2 + over_y**2))

    def optimize(self, top_points_xy, init_pose, method: str = "Powell"):
        """返回 (x, y, yaw_rad, cost)。init_pose 为 (x, y, yaw_rad)。"""
        pts = np.asarray(top_points_xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("top_points_xy 必须是 (N, 2)")
        x0 = np.asarray(init_pose, dtype=np.float64)
        if x0.shape != (3,):
            raise ValueError("init_pose 必须是 (x, y, yaw_rad)")

        yaw0 = x0[2] % (2.0 * np.pi)
        if yaw0 > np.pi:
            yaw0 -= 2.0 * np.pi
        x0 = np.array([x0[0], x0[1], yaw0])

        res = minimize(self._error, x0, args=(pts,), method=method)
        x, y, yaw = res.x
        yaw = yaw % (2.0 * np.pi)
        if yaw > np.pi:
            yaw -= 2.0 * np.pi
        return float(x), float(y), float(yaw), float(res.fun)

