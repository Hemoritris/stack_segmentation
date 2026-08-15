"""箱子顶面提取（M6）：离群点过滤 + RANSAC 平面 + 法向量检查。"""

from __future__ import annotations

import numpy as np

from ..core.types import BoxTopPlane


def _fit_plane_ls(points: np.ndarray) -> tuple[np.ndarray, float]:
    """最小二乘拟合平面，返回 (单位法向量, d)，满足 n·p + d = 0。"""
    centroid = points.mean(axis=0)
    # ``full_matrices=True`` would allocate an N x N matrix for a large plane.
    # Only the three right-singular vectors are needed here.
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d


def is_horizontal(normal, threshold: float = 0.95) -> bool:
    """判断法向量是否近似水平顶面（约等于 [0, 0, 1]）。"""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return bool(n[2] >= threshold)


def fit_top_plane(
    points_world,
    distance_threshold: float = 0.005,
    max_iter: int = 200,
    seed: int = 0,
) -> BoxTopPlane:
    """对单箱点云拟合近似水平顶面，返回 BoxTopPlane。"""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points_world 必须是 (N, >=3)")
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点")

    rng = np.random.default_rng(seed)
    n = len(pts)
    best_inliers = None
    best_count = -1

    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        p = pts[idx]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        d = -float(normal @ p[0])
        inl = np.abs(pts @ normal + d) < distance_threshold
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best_inliers = inl

    if best_inliers is None or best_count < 3:
        raise RuntimeError("RANSAC 未能找到足够多的平面内点")

    inlier_pts = pts[best_inliers]
    normal, d = _fit_plane_ls(inlier_pts)
    dist = np.abs(inlier_pts @ normal + d)
    rmse = float(np.sqrt(np.mean(dist**2)))
    height = float(np.median(inlier_pts[:, 2]))
    return BoxTopPlane(normal=normal, height=height, points=inlier_pts, plane_rmse=rmse)
