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
    fx: float | None = None,
    fy: float | None = None,
):
    """构造一个正俯视、光轴朝下的虚拟相机。

    Returns:
        (intrinsics, T_cam_world)。intrinsics 含 fx/fy/cx/cy；T_cam_world 为 4x4 齐次变换。
    """
    fx = float(width) if fx is None else float(fx)
    fy = fx if fy is None else float(fy)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    T_cam_world = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, cam_height],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return intrinsics, T_cam_world


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
) -> np.ndarray:
    """渲染正俯视深度图（相机坐标系 z），floor 在 world z=0。

    Args:
        boxes: 可迭代的 (cx, cy, length, width, height, yaw_deg)。
        正俯视下箱体侧面不可见，仅渲染顶面。
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    uu, vv = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    depth = np.full((height, width), cam_height, dtype=np.float32)
    for bx, by, L, W, H, yaw in boxes:
        z_cam = cam_height - H
        Xb = (uu - cx) / fx * z_cam
        Yb = (vv - cy) / fy * z_cam
        inside = _in_rotated_rect(Xb, Yb, bx, by, L, W, yaw)
        depth[inside] = np.minimum(depth[inside], z_cam)
    return depth

