"""深度图 -> 相机点云。"""

from __future__ import annotations

import cv2
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


def aligned_depth_to_pointcloud(
    depth_m,
    camera_matrix,
    distortion=None,
    mask=None,
    stride: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float = 6.0,
) -> np.ndarray:
    """把对齐到彩色图的米制深度反投影到彩色 optical frame。

    使用厂家 Brown-Conrady 畸变参数将像素转换为归一化无畸变光线。
    ``stride`` 只做规则降采样，不改变三维坐标。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m 必须是二维数组")
    if stride < 1:
        raise ValueError("stride 必须 >= 1")
    selected = np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask, bool).copy()
    if selected.shape != depth.shape:
        raise ValueError("mask 与 depth_m 形状不一致")
    if stride > 1:
        sampled = np.zeros(depth.shape, dtype=bool)
        sampled[::stride, ::stride] = True
        selected &= sampled
    selected &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    v, u = np.nonzero(selected)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float64)
    pixels = np.stack([u, v], axis=1).astype(np.float64).reshape(-1, 1, 2)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray([] if distortion is None else distortion, dtype=np.float64).reshape(-1)
    normalized = cv2.undistortPoints(pixels, k, d if len(d) else None).reshape(-1, 2)
    z = depth[v, u].astype(np.float64)
    return np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
