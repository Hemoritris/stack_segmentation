"""深度图 -> 相机点云。"""

from __future__ import annotations

import numpy as np


def depth_to_pointcloud(depth, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """把深度图转换为相机坐标 (N, 3) 点云，跳过无效深度。"""
    d = np.asarray(depth, dtype=np.float32)
    v, u = np.indices(d.shape)
    valid = (d > 0) & np.isfinite(d)
    z = d[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def masked_depth_to_pointcloud(
    depth,
    mask,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """仅用 mask 内的深度反投影为相机坐标 (N, 3) 点云（M5 的第一步）。"""
    d = np.asarray(depth, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if d.shape != m.shape:
        raise ValueError("depth 与 mask 形状不一致")
    v, u = np.indices(d.shape)
    valid = m & (d > 0) & np.isfinite(d)
    z = d[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=1)

