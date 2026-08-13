"""4DoF 几何 baseline：顶面点云 XY 投影 + minAreaRect。"""

from __future__ import annotations

import cv2
import numpy as np


def fit_min_area_rect(points_xy: np.ndarray) -> tuple[float, float, float, float, float]:
    """对 XY 投影点拟合最小外接矩形。

    Args:
        points_xy: (N, 2) 顶面点的世界 XY 坐标。

    Returns:
        (cx, cy, length, width, yaw_deg)。yaw 为长边相对 +X 的角度，归一化到 [0, 180)。
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy 必须是 (N, 2)")
    if len(pts) < 4:
        raise ValueError("至少需要 4 个点")

    hull = cv2.convexHull(pts)
    (cx, cy), (w, h), angle = cv2.minAreaRect(hull)

    # OpenCV 的 angle ∈ [-90, 0)；这里统一 length >= width，并给出长边 yaw。
    if w < h:
        w, h = h, w
        angle += 90.0
    yaw = angle % 180.0
    return float(cx), float(cy), float(w), float(h), float(yaw)

