"""相机内外参加载与坐标变换。"""

from __future__ import annotations

import numpy as np


def cam_to_world(points: np.ndarray, T_cam_world: np.ndarray) -> np.ndarray:
    """把相机坐标点云转换到 world 坐标。

    Args:
        points: (N, 3) 相机坐标点。
        T_cam_world: (4, 4) 齐次变换矩阵，P_world = T @ P_cam。

    Returns:
        (N, 3) world 坐标点。
    """
    pts = np.asarray(points, dtype=np.float64)
    T = np.asarray(T_cam_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 必须是 (N, 3)")
    if T.shape != (4, 4):
        raise ValueError("T_cam_world 必须是 (4, 4)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    return (homo @ T.T)[:, :3]


def load_intrinsics(config_path: str):
    """从 YAML 读取深度 / 彩色内参。"""
    raise NotImplementedError("TODO(M0): 解析 config/camera.yaml 内参")


def load_extrinsics(config_path: str) -> np.ndarray:
    """从 YAML 读取相机到 world 的 4x4 外参。"""
    raise NotImplementedError("TODO(M0): 解析 config/camera.yaml 外参")

