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

