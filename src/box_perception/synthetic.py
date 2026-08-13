"""合成数据生成：在没有相机 / 标定时用于算法开发与回归测试。"""

from __future__ import annotations

import numpy as np


def box_top_points(
    center=(0.0, 0.0, 0.35),
    length: float = 0.6,
    width: float = 0.4,
    yaw_deg: float = 0.0,
    spacing: float = 0.02,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """生成覆盖箱子顶面的 world 坐标点云 (N, 3)。

    Args:
        center: 顶面中心 (x, y, z)。
        length, width: 箱体长宽（米）。
        yaw_deg: 绕 +Z 的旋转角（度）。
        spacing: 顶面采样间距（米）。
        noise: 高斯噪声标准差（米），0 表示不加噪。
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(-length / 2.0, length / 2.0 + 1e-9, spacing)
    ys = np.arange(-width / 2.0, width / 2.0 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    local = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    th = np.deg2rad(yaw_deg)
    R = np.array(
        [
            [np.cos(th), -np.sin(th), 0.0],
            [np.sin(th), np.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pts = local @ R.T + np.asarray(center, dtype=np.float64)
    if noise > 0:
        pts += rng.normal(0.0, noise, pts.shape)
    return pts


def make_overhead_camera(
    width: int,
    height: int,
    cam_height: float,
    pixel_size_m: float = 0.005,
):
    """构造一个正俯视、光轴朝下的（近似）正交投影虚拟相机。

    Returns:
        intrinsics dict，含 pixel_size_m / cx / cy / cam_height。
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return {
        "pixel_size_m": float(pixel_size_m),
        "cx": cx,
        "cy": cy,
        "cam_height": float(cam_height),
    }


def _in_rotated_rect(x, y, cx, cy, length, width, yaw_deg) -> np.ndarray:
    th = np.deg2rad(yaw_deg)
    dx = x - cx
    dy = y - cy
    lx = dx * np.cos(th) + dy * np.sin(th)
    ly = -dx * np.sin(th) + dy * np.cos(th)
    return (np.abs(lx) <= length / 2.0) & (np.abs(ly) <= width / 2.0)


def render_scene_depth(
    width: int,
    height: int,
    cam_height: float,
    intrinsics: dict,
    boxes,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """渲染正俯视正交深度图（相机坐标系 z），floor 在 world z=0。

    Args:
        boxes: 可迭代的 (cx, cy, length, width, top_z, yaw_deg)。
        正俯视正交投影下箱体侧面不可见，仅渲染顶面。
        noise: 加到深度上的高斯噪声标准差（米）。
    """
    ps = intrinsics["pixel_size_m"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    uu, vv = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    X = (uu - cx) * ps
    Y = (vv - cy) * ps
    depth = np.full((height, width), cam_height, dtype=np.float32)  # floor
    for bx, by, L, W, top_z, yaw in boxes:
        inside = _in_rotated_rect(X, Y, bx, by, L, W, yaw)
        depth[inside] = np.minimum(depth[inside], cam_height - top_z)
    if noise > 0:
        rng = np.random.default_rng(seed)
        depth = depth + rng.normal(0.0, noise, depth.shape).astype(np.float32)
        depth = np.maximum(depth, 1e-3)
    return depth


def ortho_depth_to_world(depth, pixel_size_m, cx, cy, cam_height) -> np.ndarray:
    """把正俯视正交深度图反投影为 world 坐标 (N, 3) 点云。"""
    d = np.asarray(depth, dtype=np.float64)
    v, u = np.indices(d.shape)
    valid = (d > 0) & np.isfinite(d)
    X = (u[valid] - cx) * pixel_size_m
    Y = (v[valid] - cy) * pixel_size_m
    Z = cam_height - d[valid]
    return np.stack([X, Y, Z], axis=1)
