#!/usr/bin/env python3
"""固定 L515 最高活动层箱体 4DoF 跟踪测试。

设计约束：

* YOLO-Seg 只产生候选 mask，不能单独确认箱体；
* 候选必须通过 RGB-D 顶面 RANSAC、尺寸、面积完整度、层高和托盘范围校验；
* 由托盘顶面、箱高和最高有效顶面高度确定活动层；
* 检测到稳定的更高层后冻结旧层，只有最高活动层继续更新 4DoF；
* 本脚本只写独立测试状态 JSON，不修改正式 StackMap。

窗口按键：Q=保存并退出，S=立即保存测试状态。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.camera.calibration import (  # noqa: E402
    cam_to_world,
    load_intrinsics,
    load_world_calibration,
)
from box_perception.camera.ros_rgbd import ROSAlignedRGBDSource  # noqa: E402
from box_perception.core.types import BoxTopPlane  # noqa: E402
from box_perception.geometry.plane_fitting import is_horizontal  # noqa: E402
from box_perception.geometry.rectangle_init import fit_min_area_rect  # noqa: E402
from box_perception.real_pipeline import project_world_points_to_image  # noqa: E402
from box_perception.segmentation.yolo_segmentor import YOLOSegmentor  # noqa: E402


@dataclass
class Candidate4DoF:
    accepted: bool
    reasons: list[str]
    mask: np.ndarray
    bbox: np.ndarray | None
    yolo_confidence: float
    layer: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    top_z: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    plane_rmse: float = float("inf")
    valid_depth_ratio: float = 0.0
    top_area_ratio: float = 0.0
    rectangle_fill_ratio: float = 0.0
    top_inlier_ratio: float = 0.0
    layer_height_error: float = float("inf")
    geometry_score: float = 0.0


@dataclass
class Track4DoF:
    id: int
    layer: int
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    confidence: float
    geometry_score: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    frozen: bool = False
    timestamp: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument(
        "--tray-reference",
        type=Path,
        default=ROOT / "record/tray_reference/tray_reference.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "record/top_layer_tracking_test",
    )
    parser.add_argument("--resume-state", type=Path, default=None)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.50)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.40, 0.30, 0.30))
    parser.add_argument("--inference-hz", type=float, default=2.0)
    parser.add_argument("--depth-median-frames", type=int, default=3)
    parser.add_argument("--read-timeout", type=float, default=8.0)
    parser.add_argument(
        "--timing-every",
        type=int,
        default=10,
        help="每 N 次推理打印一次分段平均耗时；0 表示禁用计时打印",
    )

    # Candidate geometry gates.  These deliberately reject partial YOLO masks.
    parser.add_argument("--min-mask-pixels", type=int, default=1200)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.55)
    parser.add_argument("--min-top-points", type=int, default=450)
    parser.add_argument(
        "--max-geometry-points",
        type=int,
        default=6000,
        help="单箱点云超过该数量时随机降采样（固定种子），加速凸包/矩形拟合且几乎不影响精度",
    )
    parser.add_argument("--plane-threshold", type=float, default=0.008)
    parser.add_argument("--min-normal-z", type=float, default=0.97)
    parser.add_argument("--max-plane-rmse", type=float, default=0.008)
    parser.add_argument("--min-size-ratio", type=float, default=0.82)
    parser.add_argument("--max-size-ratio", type=float, default=1.22)
    parser.add_argument("--min-top-area-ratio", type=float, default=0.65)
    parser.add_argument("--min-rectangle-fill", type=float, default=0.60)
    parser.add_argument("--min-top-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--layer-height-tolerance", type=float, default=0.055)
    parser.add_argument("--reject-image-border", action="store_true")

    # Multi-frame state machine.
    parser.add_argument("--layer-switch-cycles", type=int, default=3)
    parser.add_argument("--confirm-cycles", type=int, default=3)
    parser.add_argument("--max-match-distance", type=float, default=0.16)
    parser.add_argument("--max-match-yaw-deg", type=float, default=35.0)
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="EMA 平滑系数；1.0 表示每次直接用检测值（无滞后），越小越平滑但滞后越大",
    )
    parser.add_argument("--max-missed-cycles", type=int, default=8)
    parser.add_argument("--show-rejected", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_yaw(yaw_deg: float) -> float:
    """Normalize a rectangle's 180-degree-symmetric yaw to [-90, 90)."""
    return float((float(yaw_deg) + 90.0) % 180.0 - 90.0)


def yaw_error_deg(first: float, second: float) -> float:
    return abs(normalize_yaw(float(first) - float(second)))


def blend_yaw_deg(current: float, measured: float, alpha: float) -> float:
    delta = normalize_yaw(float(measured) - float(current))
    return normalize_yaw(float(current) + float(alpha) * delta)


def layer_from_top_height(
    top_z: float,
    tray_top_z: float,
    box_height: float,
) -> tuple[int, float]:
    if box_height <= 0.0:
        raise ValueError("box_height must be positive")
    relative = (float(top_z) - float(tray_top_z)) / float(box_height)
    layer = max(1, int(round(relative)))
    expected_top = float(tray_top_z) + layer * float(box_height)
    return layer, float(top_z) - expected_top


def geometry_rejection_reasons(
    *,
    length: float,
    width: float,
    expected_length: float,
    expected_width: float,
    top_area_ratio: float,
    rectangle_fill_ratio: float,
    top_inlier_ratio: float,
    min_size_ratio: float,
    max_size_ratio: float,
    min_top_area_ratio: float,
    min_rectangle_fill: float,
    min_top_inlier_ratio: float,
) -> list[str]:
    measured = sorted((float(length), float(width)), reverse=True)
    expected = sorted((float(expected_length), float(expected_width)), reverse=True)
    reasons: list[str] = []
    for label, value, target in zip(("L", "W"), measured, expected):
        ratio = value / max(target, 1e-9)
        if ratio < min_size_ratio or ratio > max_size_ratio:
            reasons.append(f"{label}_ratio={ratio:.2f}")
    if top_area_ratio < min_top_area_ratio:
        reasons.append(f"top_area={top_area_ratio:.2f}")
    if rectangle_fill_ratio < min_rectangle_fill:
        reasons.append(f"rect_fill={rectangle_fill_ratio:.2f}")
    if top_inlier_ratio < min_top_inlier_ratio:
        reasons.append(f"top_inliers={top_inlier_ratio:.2f}")
    return reasons


def _median_depth_masked(
    depth_frames: deque[np.ndarray], mask: np.ndarray | None = None
) -> np.ndarray:
    """对 depth_frames 求逐像素时域中值，仅计算 ``mask`` 内像素（None 表示全图）。

    非 mask 像素填充 NaN。对同一像素而言参与计算的仍是那几帧、仍是 nanmedian，
    因此与全图版本逐位等价，只是不再计算“算了却不被读取”的像素。
    """
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
    median = _median_of_n(vals)
    out = np.full(shape, np.nan, dtype=np.float32)
    out[indices] = median
    return out


def _median_of_n(vals: list[np.ndarray]) -> np.ndarray:
    """对 (F, N) 时间序列逐元素求“忽略 NaN 的中位数”，与 np.nanmedian 逐位等价。

    3 帧用纯 elementwise 运算特化（无排序，约 5 倍快于 nanmedian）；
    其它帧数退回 np.nanmedian 保持原语义。
    """
    if len(vals) == 1:
        return vals[0].copy()
    if len(vals) == 3:
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
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        return np.nanmedian(stacked, axis=0).astype(np.float32)


def _fit_plane_ls_local(points: np.ndarray) -> tuple[np.ndarray, float]:
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
    points_world: np.ndarray,
    distance_threshold: float = 0.008,
) -> BoxTopPlane:
    """水平先验 + 迭代稳健平面拟合，替代暴力 RANSAC。

    箱体顶面近似水平：先以中值高度作先验筛选内点，再对内点做 SVD 拟合真实
    法向量（保留轻微倾斜），随后用平面距离再做一次内点筛选并重新拟合。
    对外点的鲁棒性与 RANSAC 相当，但只需 2 次 SVD，取代 250 次迭代。
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points_world 必须是 (N, >=3)")
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点")

    height = float(np.median(pts[:, 2]))
    inliers = np.abs(pts[:, 2] - height) <= distance_threshold
    if int(inliers.sum()) < 3:
        inliers = np.ones(len(pts), dtype=bool)
    normal, d = _fit_plane_ls_local(pts[inliers])

    refined = np.abs(pts @ normal + d) <= distance_threshold
    if int(refined.sum()) >= 3:
        inliers = refined
        normal, d = _fit_plane_ls_local(pts[inliers])

    inlier_pts = pts[inliers]
    dist = np.abs(inlier_pts @ normal + d)
    rmse = float(np.sqrt(np.mean(dist**2)))
    height = float(np.median(inlier_pts[:, 2]))
    return BoxTopPlane(normal=normal, height=height, points=inlier_pts, plane_rmse=rmse)


def build_ray_lookup(intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    """预计算整图“像素 → 去畸变归一化光线”查找表 ``(ray_x, ray_y)``。

    相机内参与畸变固定，反投影从每帧调用 ``cv2.undistortPoints`` 改为查表，
    数值与原实现逐位一致。
    """
    width, height = intrinsics.width, intrinsics.height
    k = np.asarray(intrinsics.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1)
    v, u = np.mgrid[0:height, 0:width]
    pixels = (
        np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    )
    normalized = cv2.undistortPoints(pixels, k, d).reshape(height, width, 2)
    ray_x = normalized[:, :, 0].astype(np.float32)
    ray_y = normalized[:, :, 1].astype(np.float32)
    return ray_x, ray_y


def backproject_masked(
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 6.0,
) -> np.ndarray:
    """用预计算的光线查找表把 mask 内深度反投影为相机点云 (N, 3)。"""
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


def _load_tray(path: Path, expected_map_sha256: str | None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if expected_map_sha256 and payload.get("map_sha256") != expected_map_sha256:
        raise ValueError(
            f"tray map SHA256 mismatch: {payload.get('map_sha256')} != {expected_map_sha256}"
        )
    tray = payload.get("tray")
    if not isinstance(tray, dict):
        raise ValueError(f"tray reference is invalid: {resolved}")
    return tray


def _polygon_area(points_xy: np.ndarray) -> float:
    hull = cv2.convexHull(np.asarray(points_xy, dtype=np.float32))
    return float(abs(cv2.contourArea(hull)))


def _box_corners_xy(
    x: float, y: float, length: float, width: float, yaw_deg: float
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    ux = np.array([math.cos(yaw), math.sin(yaw)])
    uy = np.array([-math.sin(yaw), math.cos(yaw)])
    center = np.array([float(x), float(y)])
    return np.asarray(
        [
            center + sx * length * 0.5 * ux + sy * width * 0.5 * uy
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ],
        dtype=np.float64,
    )


def estimate_candidate(
    instance: Any,
    depth_m: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    world_t_camera: np.ndarray,
    tray: dict[str, Any],
    args: argparse.Namespace,
) -> Candidate4DoF:
    mask = np.asarray(instance.mask, dtype=bool)
    bbox = None if instance.bbox is None else np.asarray(instance.bbox, dtype=np.float64)
    candidate = Candidate4DoF(
        accepted=False,
        reasons=[],
        mask=mask,
        bbox=bbox,
        yolo_confidence=float(instance.confidence),
    )
    if mask.shape != depth_m.shape:
        candidate.reasons.append("mask_shape")
        return candidate
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels < args.min_mask_pixels:
        candidate.reasons.append(f"mask_pixels={mask_pixels}")
        return candidate
    if args.reject_image_border:
        border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        if border:
            candidate.reasons.append("image_border")

    valid_depth = mask & np.isfinite(depth_m) & (depth_m >= 0.2) & (depth_m <= 6.0)
    candidate.valid_depth_ratio = float(np.count_nonzero(valid_depth) / max(mask_pixels, 1))
    if candidate.valid_depth_ratio < args.min_valid_depth_ratio:
        candidate.reasons.append(f"depth={candidate.valid_depth_ratio:.2f}")
        return candidate

    # Erode away mask boundaries, which commonly contain background or side-face depth.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if np.count_nonzero(eroded & valid_depth) >= args.min_top_points:
        point_mask = eroded & valid_depth
    else:
        point_mask = valid_depth
    camera_points = backproject_masked(
        depth_m,
        point_mask,
        ray_x,
        ray_y,
        min_depth_m=0.2,
        max_depth_m=6.0,
    )
    world_points = cam_to_world(camera_points, world_t_camera)
    if len(world_points) < args.min_top_points:
        candidate.reasons.append(f"world_points={len(world_points)}")
        return candidate
    if len(world_points) > args.max_geometry_points:
        rng = np.random.default_rng(0)
        world_points = world_points[
            rng.choice(len(world_points), args.max_geometry_points, replace=False)
        ]

    # The top surface is the highest dense horizontal band within the candidate.
    high_quantile = float(np.percentile(world_points[:, 2], 82.0))
    top_seed = world_points[world_points[:, 2] >= high_quantile - 0.035]
    if len(top_seed) < args.min_top_points:
        candidate.reasons.append(f"top_seed={len(top_seed)}")
        return candidate
    if len(top_seed) > 12000:
        rng = np.random.default_rng(0)
        top_seed = top_seed[rng.choice(len(top_seed), 12000, replace=False)]
    try:
        plane = fit_top_plane_fast(
            top_seed,
            distance_threshold=args.plane_threshold,
        )
    except (ValueError, RuntimeError) as exc:
        candidate.reasons.append(f"plane:{type(exc).__name__}")
        return candidate
    candidate.plane_rmse = float(plane.plane_rmse)
    if not is_horizontal(plane.normal, args.min_normal_z):
        candidate.reasons.append(f"normal_z={float(plane.normal[2]):.3f}")
    if candidate.plane_rmse > args.max_plane_rmse:
        candidate.reasons.append(f"plane_rmse={candidate.plane_rmse * 1000.0:.1f}mm")

    normal = np.asarray(plane.normal, dtype=np.float64)
    offset = -float(np.median(np.asarray(plane.points) @ normal))
    distances = np.abs(world_points @ normal + offset)
    top_points = world_points[
        (distances <= args.plane_threshold)
        & (world_points[:, 2] >= float(plane.height) - 0.020)
    ]
    if len(top_points) < args.min_top_points:
        candidate.reasons.append(f"top_points={len(top_points)}")
        return candidate
    candidate.top_inlier_ratio = float(len(top_points) / max(len(world_points), 1))
    try:
        x, y, length, width, yaw = fit_min_area_rect(top_points[:, :2])
    except ValueError as exc:
        candidate.reasons.append(f"rectangle:{type(exc).__name__}")
        return candidate

    expected_length, expected_width, expected_height = map(float, args.box_size)
    expected_length, expected_width = sorted((expected_length, expected_width), reverse=True)
    hull_area = _polygon_area(top_points[:, :2])
    candidate.top_area_ratio = hull_area / max(expected_length * expected_width, 1e-9)
    candidate.rectangle_fill_ratio = hull_area / max(length * width, 1e-9)
    candidate.reasons.extend(
        geometry_rejection_reasons(
            length=length,
            width=width,
            expected_length=expected_length,
            expected_width=expected_width,
            top_area_ratio=candidate.top_area_ratio,
            rectangle_fill_ratio=candidate.rectangle_fill_ratio,
            top_inlier_ratio=candidate.top_inlier_ratio,
            min_size_ratio=args.min_size_ratio,
            max_size_ratio=args.max_size_ratio,
            min_top_area_ratio=args.min_top_area_ratio,
            min_rectangle_fill=args.min_rectangle_fill,
            min_top_inlier_ratio=args.min_top_inlier_ratio,
        )
    )

    top_z = float(np.median(top_points[:, 2]))
    tray_top_z = float(tray["pose_4dof"]["z_m"])
    layer, layer_error = layer_from_top_height(top_z, tray_top_z, expected_height)
    if abs(layer_error) > args.layer_height_tolerance:
        candidate.reasons.append(f"layer_dz={layer_error * 1000.0:+.0f}mm")

    candidate.layer = layer
    candidate.x = float(x)
    candidate.y = float(y)
    candidate.z = top_z - expected_height * 0.5
    candidate.yaw = normalize_yaw(yaw)
    candidate.top_z = top_z
    candidate.length = float(length)
    candidate.width = float(width)
    candidate.height = expected_height
    candidate.layer_height_error = float(layer_error)
    size_score = min(
        length / expected_length,
        expected_length / max(length, 1e-9),
        width / expected_width,
        expected_width / max(width, 1e-9),
    )
    candidate.geometry_score = float(
        np.clip(
            0.30 * size_score
            + 0.25 * min(candidate.top_area_ratio, 1.0)
            + 0.20 * min(candidate.rectangle_fill_ratio, 1.0)
            + 0.15 * min(candidate.top_inlier_ratio / 0.5, 1.0)
            + 0.10 * candidate.valid_depth_ratio,
            0.0,
            1.0,
        )
    )
    candidate.accepted = not candidate.reasons
    return candidate


def deduplicate_candidates(candidates: list[Candidate4DoF]) -> list[Candidate4DoF]:
    accepted = sorted(
        (candidate for candidate in candidates if candidate.accepted),
        key=lambda item: (item.geometry_score, item.yolo_confidence),
        reverse=True,
    )
    kept: list[Candidate4DoF] = []
    for candidate in accepted:
        duplicate = any(
            candidate.layer == other.layer
            and math.hypot(candidate.x - other.x, candidate.y - other.y) < 0.12
            for other in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


class LayerFreezeTracker:
    def __init__(
        self,
        *,
        switch_cycles: int = 3,
        confirm_cycles: int = 3,
        max_match_distance: float = 0.16,
        max_match_yaw_deg: float = 35.0,
        smoothing_alpha: float = 0.35,
        max_missed_cycles: int = 8,
    ) -> None:
        self.switch_cycles = int(switch_cycles)
        self.confirm_cycles = int(confirm_cycles)
        self.max_match_distance = float(max_match_distance)
        self.max_match_yaw_deg = float(max_match_yaw_deg)
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_missed_cycles = int(max_missed_cycles)
        self.active_layer = 0
        self.pending_layer = 0
        self.pending_cycles = 0
        self.tracks: list[Track4DoF] = []
        self.next_id = 1
        self.last_event = "waiting for a geometrically valid top layer"

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "LayerFreezeTracker":
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        tracker = cls(**kwargs)
        tracker.active_layer = int(payload.get("active_layer", 0))
        tracker.pending_layer = int(payload.get("pending_layer", 0))
        tracker.pending_cycles = int(payload.get("pending_cycles", 0))
        allowed = set(Track4DoF.__dataclass_fields__)
        tracker.tracks = [
            Track4DoF(**{key: value for key, value in item.items() if key in allowed})
            for item in payload.get("tracks", [])
        ]
        tracker.next_id = max((track.id for track in tracker.tracks), default=0) + 1
        tracker.last_event = "resumed test tracker state"
        return tracker

    def to_dict(self, *, world_frame: str, map_sha256: str | None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "purpose": "top-layer-freeze-tracking-test-only",
            "world_frame": world_frame,
            "map_sha256": map_sha256,
            "active_layer": self.active_layer,
            "pending_layer": self.pending_layer,
            "pending_cycles": self.pending_cycles,
            "last_event": self.last_event,
            "tracks": [asdict(track) for track in self.tracks],
        }

    def _switch_layer_if_stable(self, detections: list[Candidate4DoF]) -> bool:
        if not detections:
            self.pending_layer = 0
            self.pending_cycles = 0
            return False
        highest = max(detection.layer for detection in detections)
        if highest <= self.active_layer:
            self.pending_layer = 0
            self.pending_cycles = 0
            return False
        if self.active_layer > 0 and highest > self.active_layer + 1:
            self.last_event = (
                f"ignored implausible layer jump L{self.active_layer}->L{highest}"
            )
            self.pending_layer = 0
            self.pending_cycles = 0
            return False
        if highest == self.pending_layer:
            self.pending_cycles += 1
        else:
            self.pending_layer = highest
            self.pending_cycles = 1
        if self.pending_cycles < self.switch_cycles:
            self.last_event = (
                f"validating new highest layer L{highest}: "
                f"{self.pending_cycles}/{self.switch_cycles}"
            )
            return False

        old_layer = self.active_layer
        for track in self.tracks:
            if track.layer < highest:
                track.frozen = True
        self.tracks = [track for track in self.tracks if track.confirmed or not track.frozen]
        self.active_layer = highest
        self.pending_layer = 0
        self.pending_cycles = 0
        self.last_event = (
            f"initialized active layer L{highest}"
            if old_layer == 0
            else f"froze L{old_layer}; active layer is now L{highest}"
        )
        return True

    def update(self, detections: list[Candidate4DoF], timestamp: float) -> None:
        switched = self._switch_layer_if_stable(detections)
        if self.active_layer == 0:
            return
        active_detections = [item for item in detections if item.layer == self.active_layer]
        active_tracks = [
            track for track in self.tracks if track.layer == self.active_layer and not track.frozen
        ]
        for track in active_tracks:
            track.missed += 1

        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active_tracks):
            for detection_index, detection in enumerate(active_detections):
                distance = math.hypot(track.x - detection.x, track.y - detection.y)
                yaw_error = yaw_error_deg(track.yaw, detection.yaw)
                if distance <= self.max_match_distance and yaw_error <= self.max_match_yaw_deg:
                    cost = distance + 0.002 * yaw_error
                    pairs.append((cost, track_index, detection_index))
        pairs.sort()
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        alpha = self.smoothing_alpha
        for _cost, track_index, detection_index in pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = active_tracks[track_index]
            detection = active_detections[detection_index]
            track.x = (1.0 - alpha) * track.x + alpha * detection.x
            track.y = (1.0 - alpha) * track.y + alpha * detection.y
            track.z = (1.0 - alpha) * track.z + alpha * detection.z
            track.yaw = blend_yaw_deg(track.yaw, detection.yaw, alpha)
            track.length = (1.0 - alpha) * track.length + alpha * detection.length
            track.width = (1.0 - alpha) * track.width + alpha * detection.width
            track.confidence = (1.0 - alpha) * track.confidence + alpha * detection.yolo_confidence
            track.geometry_score = (
                (1.0 - alpha) * track.geometry_score + alpha * detection.geometry_score
            )
            track.hits += 1
            track.missed = 0
            track.confirmed = track.hits >= self.confirm_cycles
            track.timestamp = float(timestamp)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(active_detections):
            if detection_index in matched_detections:
                continue
            track = Track4DoF(
                id=self.next_id,
                layer=detection.layer,
                x=detection.x,
                y=detection.y,
                z=detection.z,
                yaw=detection.yaw,
                length=detection.length,
                width=detection.width,
                height=detection.height,
                confidence=detection.yolo_confidence,
                geometry_score=detection.geometry_score,
                confirmed=self.confirm_cycles <= 1,
                timestamp=float(timestamp),
            )
            self.next_id += 1
            self.tracks.append(track)

        # 已冻结的历史层轨迹永久保留；活动层轨迹（无论是否 confirmed）
        # 连续丢失超过 max_missed_cycles 帧后删除，避免箱子被拿走/长期消失后
        # 画面上仍残留黄色 HOLD 边框。
        self.tracks = [
            track
            for track in self.tracks
            if track.frozen or track.missed <= self.max_missed_cycles
        ]
        if not switched:
            confirmed = sum(
                track.confirmed and track.layer == self.active_layer for track in self.tracks
            )
            self.last_event = (
                f"tracking L{self.active_layer}: {confirmed} confirmed, "
                f"{len(active_detections)} valid observations"
            )


def _project(points_world: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    return project_world_points_to_image(
        points_world,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["k"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["distortion"], dtype=np.float64),
    )


def _draw_world_rectangle(
    image: np.ndarray,
    *,
    x: float,
    y: float,
    top_z: float,
    length: float,
    width: float,
    yaw: float,
    manifest: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    corners_xy = _box_corners_xy(x, y, length, width, yaw)
    center = np.array([[x, y, top_z]], dtype=np.float64)
    yaw_rad = math.radians(yaw)
    axes = np.array(
        [
            [x + 0.18 * math.cos(yaw_rad), y + 0.18 * math.sin(yaw_rad), top_z],
            [x - 0.14 * math.sin(yaw_rad), y + 0.14 * math.cos(yaw_rad), top_z],
        ],
        dtype=np.float64,
    )
    points = np.vstack([center, axes, np.column_stack([corners_xy, np.full(4, top_z)])])
    pixels = _project(points, manifest)
    if not np.all(np.isfinite(pixels)):
        return
    p = [tuple(np.round(point).astype(int)) for point in pixels]
    cv2.polylines(
        image, [np.asarray(p[3:], dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness
    )
    cv2.arrowedLine(image, p[0], p[1], color, thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, p[0], p[2], (255, 255, 0), thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(
        image,
        label,
        (p[0][0] + 6, p[0][1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_candidate_masks(
    image: np.ndarray,
    candidates: list[Candidate4DoF],
    active_layer: int,
    show_rejected: bool,
) -> None:
    overlay = image.copy()
    for candidate in candidates:
        if candidate.accepted:
            color = (40, 220, 40) if candidate.layer == active_layer else (0, 180, 255)
            overlay[candidate.mask] = color
        elif show_rejected:
            color = (40, 40, 230)
        else:
            continue
        contours, _ = cv2.findContours(
            candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        if candidate.bbox is not None and len(candidate.bbox.reshape(-1)) >= 4:
            x1, y1, x2, y2 = np.round(candidate.bbox.reshape(-1)[:4]).astype(int)
            label = (
                f"VALID L{candidate.layer} G={candidate.geometry_score:.2f}"
                if candidate.accepted
                else "REJECT " + ",".join(candidate.reasons[:2])
            )
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, dst=image)


def _draw_status(
    image: np.ndarray,
    *,
    tracker: LayerFreezeTracker,
    display_fps: float,
    inference_ms: float,
    valid_count: int,
    rejected_count: int,
    highest_top_z: float | None,
) -> None:
    frozen = sum(track.frozen for track in tracker.tracks)
    active = sum(not track.frozen and track.confirmed for track in tracker.tracks)
    lines = [
        f"display={display_fps:.1f} FPS  YOLO+geometry={inference_ms:.1f} ms",
        f"active_layer=L{tracker.active_layer}  active={active}  frozen={frozen}",
        f"valid/rejected={valid_count}/{rejected_count}  highest_top_z="
        + ("--" if highest_top_z is None else f"{highest_top_z:.3f}m"),
        tracker.last_event,
        "Q: save+quit  S: save state",
    ]
    for index, line in enumerate(lines):
        y = 28 + index * 27
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)


def _save_state(
    tracker: LayerFreezeTracker,
    path: Path,
    *,
    world_frame: str,
    map_sha256: str | None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tracker.to_dict(world_frame=world_frame, map_sha256=map_sha256)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.box_size) <= 0.0:
        raise ValueError("--box-size values must be positive")
    if args.inference_hz <= 0.0 or args.depth_median_frames < 1:
        raise ValueError("inference rate and depth median frame count must be positive")
    if min(args.layer_switch_cycles, args.confirm_cycles) < 1:
        raise ValueError("state-machine cycle counts must be positive")
    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1]")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    tray = _load_tray(args.tray_reference, calibration.map_sha256)
    manifest = {
        "world_T_camera": calibration.world_T_camera.tolist(),
        "intrinsics": {
            "k": intrinsics.k.tolist(),
            "distortion": intrinsics.distortion.tolist(),
        },
    }
    ray_x, ray_y = build_ray_lookup(intrinsics)
    ros_config = config["ros"]
    camera_config = config["camera"]
    segmentor = YOLOSegmentor(
        str(args.yolo_weights),
        device=args.yolo_device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        mask_threshold=args.yolo_mask_threshold,
    )
    tracker_kwargs = {
        "switch_cycles": args.layer_switch_cycles,
        "confirm_cycles": args.confirm_cycles,
        "max_match_distance": args.max_match_distance,
        "max_match_yaw_deg": args.max_match_yaw_deg,
        "smoothing_alpha": args.smoothing_alpha,
        "max_missed_cycles": args.max_missed_cycles,
    }
    if args.resume_state is not None:
        tracker = LayerFreezeTracker.load(args.resume_state, **tracker_kwargs)
    else:
        tracker = LayerFreezeTracker(**tracker_kwargs)

    output = args.output.expanduser().resolve()
    state_path = output / "top_layer_tracker_state.json"
    depth_frames: deque[np.ndarray] = deque(maxlen=args.depth_median_frames)
    candidates: list[Candidate4DoF] = []
    accepted: list[Candidate4DoF] = []
    last_inference = float("-inf")
    inference_ms = 0.0
    last_frame_time = time.monotonic()
    display_fps = 0.0
    highest_top_z: float | None = None
    timing_ms = {
        "yolo": 0.0,
        "median": 0.0,
        "geometry": 0.0,
        "dedup+track": 0.0,
        "save": 0.0,
        "total": 0.0,
    }
    timing_cycles = 0
    timing_boxes = 0

    print("========== top-layer freeze + 4DoF tracking test ==========")
    print(f"world frame: {calibration.world_frame}")
    print(
        f"box L/W/H: {args.box_size[0]:.3f}/{args.box_size[1]:.3f}/"
        f"{args.box_size[2]:.3f} m"
    )
    print(
        "YOLO is candidate-only; confirmation requires depth plane, full-size rectangle, "
        "layer height, and tray checks"
    )
    print(f"test state: {state_path}")

    with ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros_config["color_topic"],
        depth_topic=ros_config["aligned_depth_topic"],
        camera_info_topic=ros_config["camera_info_topic"],
        max_pair_offset_s=float(ros_config["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera_config["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros_config["intrinsics_tolerance"]),
    ) as source:
        cv2.namedWindow("top-layer 4DoF tracker TEST", cv2.WINDOW_NORMAL)
        while True:
            try:
                frame = source.read(args.read_timeout)
            except TimeoutError as exc:
                print(f"WARNING: {exc}; waiting for RGB-D recovery")
                continue
            depth_frames.append(frame.aligned_depth_m)
            now = time.monotonic()
            if (
                len(depth_frames) == args.depth_median_frames
                and now - last_inference >= 1.0 / args.inference_hz
            ):
                inference_start = time.monotonic()
                instances = segmentor.segment(frame.color_bgr)
                t_yolo = time.monotonic()
                candidates = []
                t_median = t_yolo
                t_geometry = t_yolo
                if instances:
                    union_mask = np.asarray(instances[0].mask, dtype=bool).copy()
                    for instance in instances[1:]:
                        union_mask |= np.asarray(instance.mask, dtype=bool)
                    median_depth = _median_depth_masked(depth_frames, union_mask)
                    t_median = time.monotonic()
                    for instance in instances:
                        try:
                            candidate = estimate_candidate(
                                instance,
                                median_depth,
                                ray_x,
                                ray_y,
                                calibration.world_T_camera,
                                tray,
                                args,
                            )
                        except Exception as exc:
                            candidate = Candidate4DoF(
                                accepted=False,
                                reasons=[f"geometry:{type(exc).__name__}"],
                                mask=np.asarray(instance.mask, dtype=bool),
                                bbox=(
                                    None
                                    if instance.bbox is None
                                    else np.asarray(instance.bbox, dtype=np.float64)
                                ),
                                yolo_confidence=float(instance.confidence),
                            )
                            print(
                                "WARNING: rejected one YOLO candidate after geometry exception: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        candidates.append(candidate)
                    t_geometry = time.monotonic()
                accepted = deduplicate_candidates(candidates)
                highest_top_z = max((item.top_z for item in accepted), default=None)
                tracker.update(accepted, frame.color_stamp_ns / 1e9)
                t_track = time.monotonic()
                _save_state(
                    tracker,
                    state_path,
                    world_frame=calibration.world_frame,
                    map_sha256=calibration.map_sha256,
                )
                t_save = time.monotonic()
                inference_ms = (t_save - inference_start) * 1000.0
                last_inference = t_save

                if args.timing_every > 0:
                    timing_ms["yolo"] += (t_yolo - inference_start) * 1000.0
                    timing_ms["median"] += (t_median - t_yolo) * 1000.0
                    timing_ms["geometry"] += (t_geometry - t_median) * 1000.0
                    timing_ms["dedup+track"] += (t_track - t_geometry) * 1000.0
                    timing_ms["save"] += (t_save - t_track) * 1000.0
                    timing_ms["total"] += inference_ms
                    timing_cycles += 1
                    timing_boxes += len(instances)
                    if timing_cycles >= args.timing_every:
                        avg = {k: v / timing_cycles for k, v in timing_ms.items()}
                        boxes_per = timing_boxes / timing_cycles
                        fps = 1000.0 / avg["total"] if avg["total"] > 0.0 else float("inf")
                        print(
                            f"[timing] n={timing_cycles} boxes={boxes_per:.1f} | "
                            f"yolo={avg['yolo']:.1f}ms median={avg['median']:.1f}ms "
                            f"geom={avg['geometry']:.1f}ms dedup+track={avg['dedup+track']:.1f}ms "
                            f"save={avg['save']:.1f}ms | total={avg['total']:.1f}ms "
                            f"({fps:.1f} fps)"
                        )
                        timing_ms = {k: 0.0 for k in timing_ms}
                        timing_cycles = 0
                        timing_boxes = 0

            display = frame.color_bgr.copy()
            _draw_candidate_masks(display, candidates, tracker.active_layer, args.show_rejected)
            tray_pose = tray["pose_4dof"]
            tray_size = tray["measured_size_m"]
            _draw_world_rectangle(
                display,
                x=float(tray_pose["x_m"]),
                y=float(tray_pose["y_m"]),
                top_z=float(tray_pose["z_m"]),
                length=float(tray_size["length"]),
                width=float(tray_size["width"]),
                yaw=float(tray_pose["yaw_deg"]),
                manifest=manifest,
                color=(0, 165, 255),
                label="TRAY",
                thickness=2,
            )
            for track in tracker.tracks:
                if track.frozen:
                    color = (255, 120, 20)
                    state = "FROZEN"
                elif track.confirmed and track.missed == 0:
                    color = (40, 230, 40)
                    state = "TRACK"
                elif track.confirmed:
                    color = (0, 210, 255)
                    state = f"HOLD m={track.missed}"
                else:
                    color = (180, 180, 40)
                    state = f"CAND {track.hits}/{args.confirm_cycles}"
                _draw_world_rectangle(
                    display,
                    x=track.x,
                    y=track.y,
                    top_z=track.z + track.height * 0.5,
                    length=track.length,
                    width=track.width,
                    yaw=track.yaw,
                    manifest=manifest,
                    color=color,
                    label=(
                        f"B{track.id} L{track.layer} {state} "
                        f"({track.x:.3f},{track.y:.3f},{track.z:.3f},{track.yaw:.1f})"
                    ),
                    thickness=3 if not track.frozen else 2,
                )
            current_time = time.monotonic()
            instant_fps = 1.0 / max(current_time - last_frame_time, 1e-6)
            last_frame_time = current_time
            display_fps = (
                instant_fps if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant_fps
            )
            _draw_status(
                display,
                tracker=tracker,
                display_fps=display_fps,
                inference_ms=inference_ms,
                valid_count=len(accepted),
                rejected_count=sum(not candidate.accepted for candidate in candidates),
                highest_top_z=highest_top_z,
            )
            cv2.imshow("top-layer 4DoF tracker TEST", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                saved = _save_state(
                    tracker,
                    state_path,
                    world_frame=calibration.world_frame,
                    map_sha256=calibration.map_sha256,
                )
                print(f"saved test tracker state: {saved}")
            if key in (ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()
    saved = _save_state(
        tracker,
        state_path,
        world_frame=calibration.world_frame,
        map_sha256=calibration.map_sha256,
    )
    print(f"test finished; state saved to {saved}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; top-layer tracker test closed.")
        raise SystemExit(130)
