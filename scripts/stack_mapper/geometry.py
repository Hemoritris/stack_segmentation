"""几何运算：反投影、顶面/矩形拟合、中值深度、投影与坐标变换。"""

from __future__ import annotations

import warnings
from collections import deque

import cv2
import numpy as np

from .types import Intrinsics


def normalize_yaw(yaw_deg: float) -> float:
    """把 180° 对称的 yaw 归一化到 [-90, 90)。"""
    return float((float(yaw_deg) + 90.0) % 180.0 - 90.0)


def yaw_error_deg(first: float, second: float) -> float:
    """两个 yaw 的最小夹角（度）。"""
    return abs(normalize_yaw(float(first) - float(second)))


def fit_min_area_rect(points_xy: np.ndarray) -> tuple[float, float, float, float, float]:
    """XY 投影点拟合最小外接矩形，返回 (cx, cy, length, width, yaw_deg)。"""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be (N, 2)")
    if len(pts) < 4:
        raise ValueError("need at least 4 points")
    hull = cv2.convexHull(pts)
    (cx, cy), (w, h), angle = cv2.minAreaRect(hull)
    if w < h:
        w, h = h, w
        angle += 90.0
    yaw = angle % 180.0
    return float(cx), float(cy), float(w), float(h), float(yaw)


def is_horizontal(normal, threshold: float = 0.95) -> bool:
    """判断法向量是否近似水平顶面（z 分量足够大）。"""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return bool(n[2] >= threshold)


def _fit_plane_ls(points: np.ndarray) -> tuple[np.ndarray, float]:
    """最小二乘拟合平面，返回 (单位法向量, d)，满足 n·p + d = 0。"""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d


def fit_top_plane_fast(
    points_world: np.ndarray, distance_threshold: float = 0.008
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """水平先验 + 迭代稳健顶面拟合，返回 (normal, height, inlier_points, plane_rmse)。"""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points_world must be (N, >=3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")

    height = float(np.median(pts[:, 2]))
    inliers = np.abs(pts[:, 2] - height) <= distance_threshold
    if int(inliers.sum()) < 3:
        inliers = np.ones(len(pts), dtype=bool)
    normal, d = _fit_plane_ls(pts[inliers])

    refined = np.abs(pts @ normal + d) <= distance_threshold
    if int(refined.sum()) >= 3:
        inliers = refined
        normal, d = _fit_plane_ls(pts[inliers])

    inlier_pts = pts[inliers]
    dist = np.abs(inlier_pts @ normal + d)
    rmse = float(np.sqrt(np.mean(dist**2)))
    height = float(np.median(inlier_pts[:, 2]))
    return normal, height, inlier_pts, rmse


def build_ray_lookup(intrinsics: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """预计算整图“像素 → 去畸变归一化光线”查找表。"""
    width, height = intrinsics.width, intrinsics.height
    k = np.asarray(intrinsics.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1)
    v, u = np.mgrid[0:height, 0:width]
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, k, d).reshape(height, width, 2)
    return normalized[:, :, 0].astype(np.float32), normalized[:, :, 1].astype(np.float32)


def backproject_masked(
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 6.0,
) -> np.ndarray:
    """用预计算光线查找表，把 mask 内深度反投影为相机点云 (N, 3)。"""
    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    selected &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    v, u = np.nonzero(selected)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[v, u]
    nx = ray_x[v, u].astype(np.float64)
    ny = ray_y[v, u].astype(np.float64)
    return np.column_stack([nx * z, ny * z, z])


def cam_to_world(points: np.ndarray, world_T_camera: np.ndarray) -> np.ndarray:
    """相机点云 → 世界点云：P_world = world_T_camera @ P_camera。"""
    pts = np.asarray(points, dtype=np.float64)
    transform = np.asarray(world_T_camera, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.concatenate([pts, ones], axis=1) @ transform.T)[:, :3]


def project_world_points_to_image(
    points_world: np.ndarray,
    world_t_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
    """世界点云投影到（带畸变的）彩色图像，返回像素坐标 (N, 2)。"""
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(world_t_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    camera_t_world = np.linalg.inv(transform)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_camera = (homogeneous @ camera_t_world.T)[:, :3]
    visible = points_camera[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(visible):
        k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        d = np.asarray([] if distortion is None else distortion, dtype=np.float64).reshape(-1)
        projected, _ = cv2.projectPoints(
            points_camera[visible],
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            k,
            d if len(d) else None,
        )
        pixels[visible] = projected.reshape(-1, 2)
    return pixels


def median_depth_masked(
    depth_frames: deque[np.ndarray], mask: np.ndarray | None = None
) -> np.ndarray:
    """对多帧深度求逐像素时域中值，仅计算 mask 内像素（None 表示全图）。"""
    frames = list(depth_frames)
    shape = np.asarray(frames[0]).shape
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("mask 与 depth 形状不一致")
    indices = np.nonzero(mask)
    if len(indices[0]) == 0:
        return np.full(shape, np.nan, dtype=np.float32)
    vals = [np.asarray(frame)[indices].astype(np.float32) for frame in frames]
    median = median_of_n(vals)
    out = np.full(shape, np.nan, dtype=np.float32)
    out[indices] = median
    return out


def median_of_n(vals: list[np.ndarray]) -> np.ndarray:
    """对 (F, N) 时间序列逐元素求“忽略 NaN 的中位数”，与 np.nanmedian 逐位等价。"""
    if len(vals) == 1:
        return vals[0].copy()
    if len(vals) == 3:
        # 3 帧用纯 elementwise 运算特化（无排序，快于 nanmedian）。
        a, b, c = vals
        valid = (
            (~np.isnan(a)).astype(np.float32)
            + (~np.isnan(b)).astype(np.float32)
            + (~np.isnan(c)).astype(np.float32)
        )
        a0 = np.nan_to_num(a, nan=0.0)
        b0 = np.nan_to_num(b, nan=0.0)
        c0 = np.nan_to_num(c, nan=0.0)
        mn = np.minimum(np.minimum(a0, b0), c0)
        mx = np.maximum(np.maximum(a0, b0), c0)
        total = a0 + b0 + c0
        med3 = total - mn - mx
        return np.where(
            valid >= 3,
            med3,
            np.where(valid == 2, total * 0.5, np.where(valid == 1, total, np.nan)),
        ).astype(np.float32)
    stacked = np.stack(vals, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.nanmedian(stacked, axis=0).astype(np.float32)
