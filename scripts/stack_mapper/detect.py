"""检测：YOLO 分割、托盘检测、单箱 4DoF 估计与深度兜底。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .boxmap import layer_from_top_z, slot_id_from_position
from .camera import pose_world_to_ref
from .geometry import (
    backproject_masked,
    cam_to_world,
    fit_min_area_rect,
    fit_top_plane_fast,
    is_horizontal,
    normalize_yaw,
)
from .types import CandidateBox


# ---------------------------------------------------------------------------
# YOLO-Seg 封装
# ---------------------------------------------------------------------------

class YOLOSegmentor:
    """YOLO-Seg 实例分割封装。只负责给候选 mask，箱体需经几何校验确认。"""

    def __init__(
        self,
        weights: str,
        device: str | int | None = None,
        conf: float = 0.25,
        imgsz: int = 768,
        mask_threshold: float = 0.5,
        classes: list[int] | None = None,
    ):
        weight_path = Path(weights).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f"YOLO-Seg 权重不存在: {weight_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "未找到 ultralytics。请使用同时含 rclpy + ultralytics + matplotlib 的 "
                "Python 环境（如 venv --system-site-packages 后 pip install ultralytics），"
                "不要用系统 python3。"
            ) from exc
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.mask_threshold = float(mask_threshold)
        self.classes = None if classes is None else [int(v) for v in classes]
        self.model = YOLO(str(weight_path))

    def segment(self, color_bgr: np.ndarray) -> list[Any]:
        """对一帧 BGR 图像推理，返回原图尺寸的二值实例 mask 列表。"""
        image = np.asarray(color_bgr)
        height, width = image.shape[:2]
        results = self.model.predict(
            source=np.ascontiguousarray(image),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        instances = []
        for index in range(min(len(masks), len(boxes))):
            mask = np.asarray(masks[index], dtype=np.float32)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            binary = np.ascontiguousarray(mask >= self.mask_threshold, dtype=bool)
            if not np.any(binary):
                continue
            instances.append(
                {
                    "mask": binary,
                    "bbox": np.asarray(boxes[index], dtype=np.float32),
                    "confidence": float(confidences[index]),
                }
            )
        return instances


# ---------------------------------------------------------------------------
# 托盘检测
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        raise RuntimeError("no elevated tray component was found")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < min_area_pixels:
        raise RuntimeError("largest elevated component is too small")
    return labels == index


def _estimate_ground_height(points_world: np.ndarray) -> float:
    z = np.asarray(points_world, dtype=np.float64)[:, 2]
    z = z[np.isfinite(z)]
    if len(z) < 100:
        raise RuntimeError("insufficient world points for ground-height estimation")
    lower, upper = np.percentile(z, [2.0, 75.0])
    ground = z[(z >= lower) & (z <= upper)]
    if len(ground) < 100:
        raise RuntimeError("ground-height robust subset is empty")
    return float(np.median(ground))


def detect_tray_from_depth(
    depth_m: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    world_t_camera: np.ndarray,
    *,
    frame: str = "world",
    min_elevation_m: float = 0.10,
    max_elevation_m: float = 0.55,
    min_area_pixels: int = 5000,
    plane_distance_threshold_m: float = 0.008,
    max_plane_points: int = 12000,
    min_axis_ratio_for_stable_yaw: float = 1.03,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在单帧中值深度图中估计最大抬升水平托盘顶面。"""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= 0.2) & (depth <= 6.0)
    rows, cols = np.nonzero(valid)
    points_camera = backproject_masked(
        depth, valid, ray_x, ray_y, min_depth_m=0.2, max_depth_m=6.0
    )
    points_world = cam_to_world(points_camera, world_t_camera)
    ground_z = _estimate_ground_height(points_world)

    world_z_image = np.full(depth.shape, np.nan, dtype=np.float64)
    world_z_image[rows, cols] = points_world[:, 2]
    elevated = (
        np.isfinite(world_z_image)
        & (world_z_image >= ground_z + min_elevation_m)
        & (world_z_image <= ground_z + max_elevation_m)
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    elevated = cv2.morphologyEx(elevated.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    elevated = cv2.morphologyEx(elevated, cv2.MORPH_OPEN, open_kernel)
    component = _largest_component(elevated, min_area_pixels)

    tray_points = points_world[component[rows, cols]]
    if len(tray_points) < 100:
        raise RuntimeError("elevated tray mask did not produce enough world points")
    if len(tray_points) > max_plane_points:
        rng = np.random.default_rng(0)
        plane_input = tray_points[rng.choice(len(tray_points), max_plane_points, replace=False)]
    else:
        plane_input = tray_points
    normal, _h, plane_points, _r = fit_top_plane_fast(
        plane_input, distance_threshold=plane_distance_threshold_m
    )
    if not is_horizontal(normal, 0.95):
        raise RuntimeError("detected tray plane is not horizontal")

    plane_offset = -float(np.median(plane_points @ normal))
    distances = np.abs(tray_points @ normal + plane_offset)
    top_points = tray_points[distances <= plane_distance_threshold_m]
    if len(top_points) < 100:
        raise RuntimeError("too few full-resolution tray top-plane inliers")
    top_z = float(np.median(top_points[:, 2]))
    plane_rmse = float(np.sqrt(np.mean(distances[distances <= plane_distance_threshold_m] ** 2)))
    cx, cy, length, width, yaw_0_180 = fit_min_area_rect(top_points[:, :2])
    yaw_deg = normalize_yaw(yaw_0_180)
    axis_ratio = float(length / width) if width > 0.0 else float("inf")

    result = {
        "frame": str(frame),
        "pose_4dof": {"x_m": cx, "y_m": cy, "z_m": top_z, "yaw_deg": yaw_deg},
        "frame_definition": {
            "origin": "tray_top_surface_center",
            "x_axis": "measured_long_edge",
            "y_axis": "measured_short_edge",
            "z_axis": "world_+Z",
            "yaw_zero": "world_+X",
            "yaw_positive": "counterclockwise_about_world_+Z",
            "symmetry_deg": 180,
        },
        "measured_size_m": {
            "length": length,
            "width": width,
            "top_height_above_ground": top_z - ground_z,
        },
        "surface": {
            "ground_z_m": ground_z,
            "top_z_m": top_z,
            "normal": normal.tolist(),
            "plane_rmse_m": plane_rmse,
        },
        "quality": {
            "mask_pixels": int(component.sum()),
            "top_plane_points": int(len(top_points)),
            "axis_ratio": axis_ratio,
            "yaw_stable_from_shape": bool(axis_ratio >= min_axis_ratio_for_stable_yaw),
        },
    }
    artifacts = {"image_mask": component, "top_points_world": top_points}
    return result, artifacts


def load_tray_reference(path: Path, expected_map_sha256: str | None) -> dict[str, Any]:
    """加载托盘参考，校验地图 SHA256。"""
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


def save_tray_reference(path: Path, tray: dict[str, Any], map_sha256: str | None) -> None:
    """保存托盘参考（原子写入）。"""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"map_sha256": map_sha256, "tray": tray}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 单箱 4DoF 估计
# ---------------------------------------------------------------------------

def geometry_rejection_reasons(
    length: float,
    width: float,
    expected_length: float,
    expected_width: float,
    top_area_ratio: float,
    rectangle_fill_ratio: float,
    top_inlier_ratio: float,
    min_size_ratio: float,
    max_size_ratio: float,
    min_width_ratio: float,
    min_top_area_ratio: float,
    min_rectangle_fill: float,
    min_top_inlier_ratio: float,
) -> list[str]:
    """几何完整性校验，返回拒绝原因列表（空表示通过）。"""
    measured = sorted((float(length), float(width)), reverse=True)
    expected = sorted((float(expected_length), float(expected_width)), reverse=True)
    reasons: list[str] = []
    for label, value, target in zip(("L", "W"), measured, expected):
        ratio = value / max(target, 1e-9)
        # 长轴方向测量稳定，用 min_size_ratio；短轴方向受透视/深度噪声影响
        # 容易测偏小（尤其是最底层箱子），单独用更宽松的 min_width_ratio。
        lo = min_size_ratio if label == "L" else min_width_ratio
        if ratio < lo or ratio > max_size_ratio:
            reasons.append(f"{label}_ratio={ratio:.2f}")
    if top_area_ratio < min_top_area_ratio:
        reasons.append(f"top_area={top_area_ratio:.2f}")
    if rectangle_fill_ratio < min_rectangle_fill:
        reasons.append(f"rect_fill={rectangle_fill_ratio:.2f}")
    if top_inlier_ratio < min_top_inlier_ratio:
        reasons.append(f"top_inliers={top_inlier_ratio:.2f}")
    return reasons


def estimate_candidate(
    instance: dict[str, Any],
    depth_m: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    world_t_camera: np.ndarray,
    tray_pose: dict[str, Any],
    args: argparse.Namespace,
) -> CandidateBox:
    """对单个 mask 做完整几何流水线，得到 4DoF 候选（可能被拒绝）。"""
    mask = np.asarray(instance["mask"], dtype=bool)
    bbox = None if instance["bbox"] is None else np.asarray(instance["bbox"], dtype=np.float64)
    candidate = CandidateBox(
        accepted=False,
        reasons=[],
        mask=mask,
        bbox=bbox,
        yolo_confidence=float(instance["confidence"]),
    )
    if mask.shape != depth_m.shape:
        candidate.reasons.append("mask_shape")
        return candidate
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels < args.min_mask_pixels:
        candidate.reasons.append(f"mask_pixels={mask_pixels}")
        return candidate

    valid_depth = mask & np.isfinite(depth_m) & (depth_m >= 0.2) & (depth_m <= 6.0)
    candidate.valid_depth_ratio = float(np.count_nonzero(valid_depth) / max(mask_pixels, 1))
    if candidate.valid_depth_ratio < args.min_valid_depth_ratio:
        candidate.reasons.append(f"depth={candidate.valid_depth_ratio:.2f}")
        return candidate

    # 腐蚀 mask 边缘，去掉背景/侧面深度污染。
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if np.count_nonzero(eroded & valid_depth) >= args.min_top_points:
        point_mask = eroded & valid_depth
    else:
        point_mask = valid_depth
    camera_points = backproject_masked(
        depth_m, point_mask, ray_x, ray_y, min_depth_m=0.2, max_depth_m=6.0
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

    # 取最高分位带作为顶面种子，拟合水平顶面。
    high_quantile = float(np.percentile(world_points[:, 2], 82.0))
    top_seed = world_points[world_points[:, 2] >= high_quantile - 0.035]
    if len(top_seed) < args.min_top_points:
        candidate.reasons.append(f"top_seed={len(top_seed)}")
        return candidate
    if len(top_seed) > 12000:
        rng = np.random.default_rng(0)
        top_seed = top_seed[rng.choice(len(top_seed), 12000, replace=False)]
    try:
        normal, height, plane_points, plane_rmse = fit_top_plane_fast(
            top_seed, distance_threshold=args.plane_threshold
        )
    except (ValueError, RuntimeError) as exc:
        candidate.reasons.append(f"plane:{type(exc).__name__}")
        return candidate
    candidate.plane_rmse = float(plane_rmse)
    if not is_horizontal(normal, args.min_normal_z):
        candidate.reasons.append(f"normal_z={float(normal[2]):.3f}")
    if candidate.plane_rmse > args.max_plane_rmse:
        candidate.reasons.append(f"plane_rmse={candidate.plane_rmse * 1000.0:.1f}mm")

    offset = -float(np.median(np.asarray(plane_points) @ normal))
    distances = np.abs(world_points @ normal + offset)
    top_points = world_points[
        (distances <= args.plane_threshold) & (world_points[:, 2] >= float(height) - 0.020)
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

    # 由顶面高度判定层号，再按层号确定箱型做尺寸校验与编号。
    top_z = float(np.median(top_points[:, 2]))
    tray_z = float(tray_pose["z_m"])
    layer, layer_error = layer_from_top_z(top_z, tray_z)
    if abs(layer_error) > args.layer_height_tolerance:
        candidate.reasons.append(f"layer_dz={layer_error * 1000.0:+.0f}mm")
    box_type = config.LAYER_BOX_TYPES[layer] if 1 <= layer <= config.LAYER_COUNT else None
    if box_type is None:
        candidate.reasons.append(f"layer={layer}")
    else:
        expected_length = config.BOX_TYPES[box_type]["length"]
        expected_width = config.BOX_TYPES[box_type]["width"]
        expected_height = config.BOX_TYPES[box_type]["height"]
        hull_area = float(abs(cv2.contourArea(cv2.convexHull(top_points[:, :2].astype(np.float32)))))
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
                min_size_ratio=config.SIZE_RATIO_MIN,
                max_size_ratio=config.SIZE_RATIO_MAX,
                min_width_ratio=config.SIZE_RATIO_WIDTH_MIN,
                min_top_area_ratio=args.min_top_area_ratio,
                min_rectangle_fill=args.min_rectangle_fill,
                min_top_inlier_ratio=args.min_top_inlier_ratio,
            )
        )
        x_ref, y_ref = pose_world_to_ref(x, y, tray_pose)
        slot_id = slot_id_from_position(x_ref, y_ref, expected_length, expected_width)

        candidate.layer = layer
        candidate.box_type = box_type
        candidate.x = float(x)
        candidate.y = float(y)
        candidate.z = top_z - expected_height * 0.5
        candidate.yaw = normalize_yaw(yaw)
        candidate.top_z = top_z
        candidate.length = float(length)
        candidate.width = float(width)
        candidate.height = expected_height
        candidate.layer_height_error = float(layer_error)
        candidate.slot_id = slot_id

    size_score = min(
        length / max(expected_length, 1e-9) if box_type else 1.0,
        expected_length / max(length, 1e-9) if box_type else 1.0,
        width / max(expected_width, 1e-9) if box_type else 1.0,
        expected_width / max(width, 1e-9) if box_type else 1.0,
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


# ---------------------------------------------------------------------------
# 深度兜底
# ---------------------------------------------------------------------------

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1.0)


def depth_fallback_instances(
    depth_m: np.ndarray,
    tray_pose: dict[str, Any],
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    world_t_camera: np.ndarray,
    *,
    min_elevation_m: float = 0.05,
    min_area_pixels: int = 1500,
    stride: int = 2,
    exclude_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """深度兜底：YOLO 漏检时，用深度图找“高于托盘顶面”的连通区域作为候选。

    先减去 exclude_mask（YOLO 已识别区域的并集），再对剩余区域做连通域，
    从而只把 YOLO 遗漏的箱子作为候选，而不是把整片箱子当成一个 mask。

    返回与 YOLO 实例同格式的候选列表（mask + bbox + confidence）。
    这些候选仍会经过 estimate_candidate 的完整几何校验，手/机械臂/托盘边框
    等非矩形顶面会被拒绝。
    """
    H, W = depth_m.shape
    tray_z = float(tray_pose["z_m"])

    depth_s = np.asarray(depth_m)[::stride, ::stride]
    ray_x_s = ray_x[::stride, ::stride]
    ray_y_s = ray_y[::stride, ::stride]
    valid = np.isfinite(depth_s) & (depth_s >= 0.2) & (depth_s <= 6.0)
    if int(valid.sum()) == 0:
        return []

    camera_points = backproject_masked(
        depth_s, valid, ray_x_s, ray_y_s, min_depth_m=0.2, max_depth_m=6.0
    )
    if len(camera_points) == 0:
        return []
    world_points = cam_to_world(camera_points, world_t_camera)

    world_z = np.full(depth_s.shape, np.nan, dtype=np.float64)
    v, u = np.nonzero(valid)
    world_z[v, u] = world_points[:, 2]

    elevated = (np.isfinite(world_z) & (world_z >= tray_z + min_elevation_m)).astype(np.uint8)
    if int(elevated.sum()) == 0:
        return []
    # 减去 YOLO 已识别的区域（max-pooling 降采样到 stride 分辨率），只保留遗漏部分。
    if exclude_mask is not None:
        exclude_s = np.zeros(depth_s.shape, dtype=bool)
        for dy in range(stride):
            for dx in range(stride):
                exclude_s |= np.asarray(exclude_mask, dtype=bool)[dy::stride, dx::stride]
        elevated = elevated & (~exclude_s)
        if int(elevated.sum()) == 0:
            return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    elevated = cv2.morphologyEx(elevated, cv2.MORPH_CLOSE, kernel)
    elevated = cv2.morphologyEx(elevated, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(elevated, 8)
    min_area_s = max(1, min_area_pixels // (stride * stride))
    instances: list[dict[str, Any]] = []
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area_s:
            continue
        mask_s = (labels == i)
        mask = cv2.resize(
            mask_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        ys, xs = np.nonzero(mask_s)
        x0 = int(xs.min()) * stride
        y0 = int(ys.min()) * stride
        x1 = min(W, (int(xs.max()) + 1) * stride)
        y1 = min(H, (int(ys.max()) + 1) * stride)
        instances.append(
            {
                "mask": mask,
                "bbox": np.array([x0, y0, x1, y1], dtype=np.float32),
                "confidence": 0.5,
            }
        )
    return instances


def merge_instances(
    yolo_instances: list[dict[str, Any]], fallback_instances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """合并 YOLO 与深度兜底候选，按 bbox IoU 去重（保留 YOLO 候选）。"""
    merged = list(yolo_instances)
    for fb in fallback_instances:
        if any(_bbox_iou(fb["bbox"], y["bbox"]) > 0.3 for y in yolo_instances):
            continue
        merged.append(fb)
    return merged


def deduplicate_by_slot(candidates: list[CandidateBox]) -> list[CandidateBox]:
    """按 (层, 槽位) 去重，保留 geometry_score 最高者。"""
    accepted = sorted(
        (c for c in candidates if c.accepted),
        key=lambda c: c.geometry_score,
        reverse=True,
    )
    kept: dict[tuple[int, int], CandidateBox] = {}
    for c in accepted:
        key = (c.layer, c.slot_id)
        if key not in kept:
            kept[key] = c
    return list(kept.values())
