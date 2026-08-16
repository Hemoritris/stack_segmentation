#!/usr/bin/env python3
"""正式版：固定 L515 垛堆箱体建图（双箱型 A/B，四层 AABB，标准网格编号）。

单文件独立运行，不依赖仓库内其它 Python 模块（仅使用 numpy / opencv / PyYAML /
ROS2 / ultralytics 这些已安装的库）。

任务定义
========
- 两种箱子（顺序均为 长×宽×高）：
    A: 0.40 × 0.30 × 0.30 m
    B: 0.42 × 0.27 × 0.21 m
- 从下往上堆 4 层：第 1、2 层为 A，第 3、4 层为 B。
- 每层 6 个箱子：箱子长轴沿托盘短边（tray +Y，即 3D 查看器里的青色/蓝色轴），
  箱子短轴沿托盘长边（tray +X）。因此长边方向排 3 个、短边方向排 2 个。
- 编号按“标准目标区域”划分（与放置先后无关）：俯视托盘，-Y 侧一行从左到右为
  1、2、3，+Y 侧一行为 4、5、6。

两个运行模式
============
1) ``--mode update_tray``：检测空托盘，直接更新托盘参考文件（tray_reference.json）。
2) ``--mode map_stack``：加载已有托盘参考，可在码垛任意时刻打开；自动按标准位置
   补全所有冻结层（低于当前活动层）的 boxmap，同时实时识别活动层箱子。

窗口按键：Q=保存并退出，S=立即保存状态。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 任务配置
# ---------------------------------------------------------------------------

# 箱型：长、宽、高（米），顺序与需求一致（长×宽×高）。
BOX_TYPES: dict[str, dict[str, float]] = {
    "A": {"length": 0.40, "width": 0.30, "height": 0.30},
    "B": {"length": 0.42, "width": 0.27, "height": 0.21},
}

# 层号 -> 箱型（第 1、2 层 A，第 3、4 层 B）。
LAYER_BOX_TYPES: list[str | None] = [None, "A", "A", "B", "B"]

LAYER_COUNT = 4
BOXES_PER_LAYER = 6

# 尺寸校验容差：箱子存在制造/贴合误差，长宽允许偏离标准值 ±25%。
SIZE_RATIO_MIN = 0.75
SIZE_RATIO_MAX = 1.25
# 短轴（宽度）方向受透视与深度噪声影响、在最底层易测偏小，单独放宽下限。
SIZE_RATIO_WIDTH_MIN = 0.5

# 活动层箱子连续未识别多少帧后才从 boxmap 移除（容忍短暂遮挡，如上层箱子/机械臂挡一下）。
# 10 帧约等于 1 秒（inference-hz=10），覆盖摆放上层箱子期间对下层的短暂遮挡。
MISSED_FRAMES_BEFORE_REMOVE = 10

# 层高判定容差（米）：观测顶面高度与标准层顶面高度的最大偏差。
# 托盘存在约 70mm 的边框/垫板结构，箱子实际放置面高于托盘顶面中心，
# 因此放宽到 100mm；层号本身仍由“最近邻”判定，不受此容差影响。
LAYER_HEIGHT_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Intrinsics:
    width: int
    height: int
    k: np.ndarray
    distortion: np.ndarray
    distortion_model: str
    frame_id: str

    @property
    def fx(self) -> float:
        return float(self.k[0, 0])

    @property
    def fy(self) -> float:
        return float(self.k[1, 1])

    @property
    def cx(self) -> float:
        return float(self.k[0, 2])

    @property
    def cy(self) -> float:
        return float(self.k[1, 2])


@dataclass(frozen=True)
class WorldCalibration:
    world_frame: str
    camera_frame: str
    world_T_camera: np.ndarray
    map_sha256: str | None
    result_path: Path


@dataclass
class BoxState:
    """boxmap 中一个箱子的状态。id 为该层内的标准槽位编号 1~6。"""

    id: int
    layer: int
    box_type: str
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    source: str = "standard"  # standard | measured | frozen
    timestamp: float = 0.0
    missed: int = 0  # 活动层箱子连续未识别帧数，超过阈值才移除


@dataclass
class BoxMap:
    boxes: list[BoxState] = field(default_factory=list)
    world_frame: str = "slamware_map"
    map_sha256: str = ""
    tray_reference: dict[str, Any] | None = None
    active_layer: int = 0


@dataclass
class CandidateBox:
    """单帧对单个 YOLO 实例的 4DoF 估计。"""

    accepted: bool
    reasons: list[str]
    mask: np.ndarray
    bbox: np.ndarray | None
    yolo_confidence: float
    layer: int = 0
    box_type: str = ""
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
    slot_id: int = 0  # 匹配到的标准槽位编号 1~6


# ---------------------------------------------------------------------------
# 相机标定（内联自 camera/calibration.py）
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: camera config must be a YAML mapping")
    return p, data


def load_intrinsics(config_path: str | Path, stream: str = "color") -> Intrinsics:
    path, data = _load_yaml(config_path)
    try:
        entry = data["intrinsics"][stream]
        result = Intrinsics(
            width=int(entry["width"]),
            height=int(entry["height"]),
            k=np.asarray(entry["k"], dtype=np.float64).reshape(3, 3),
            distortion=np.asarray(entry.get("distortion", []), dtype=np.float64),
            distortion_model=str(entry.get("distortion_model", "plumb_bob")),
            frame_id=str(entry["frame_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid intrinsics.{stream}") from exc
    if result.width <= 0 or result.height <= 0 or result.fx <= 0 or result.fy <= 0:
        raise ValueError(f"{path}: invalid dimensions/focal length for {stream}")
    return result


def _resolve_relative(owner: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (owner.parent / candidate).resolve()


def _validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} rotation determinant is not +1")
    return matrix


def load_world_calibration(config_path: str | Path) -> WorldCalibration:
    path, data = _load_yaml(config_path)
    try:
        entry = data["extrinsics"]
        result_path = _resolve_relative(path, entry["result_json"])
        transform_key = str(entry["transform_key"])
        expected_map_sha = entry.get("expected_map_sha256")
        expected_world = str(entry["world_frame"])
        expected_camera = str(entry["camera_frame"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid extrinsics configuration") from exc

    result = json.loads(result_path.read_text(encoding="utf-8"))
    frames = result.get("frames", {})
    if result.get("calibration_scope") != "fixed_l515_world_extrinsics_only":
        raise ValueError(f"{result_path}: not a fixed-L515 world calibration result")
    if frames.get("M") != expected_world or frames.get("Cf") != expected_camera:
        raise ValueError(f"{result_path}: frame mismatch")
    actual_map_sha = result.get("dataset", {}).get("map_sha256")
    if expected_map_sha and actual_map_sha != expected_map_sha:
        raise ValueError(f"{result_path}: map SHA mismatch")
    try:
        matrix = result["transforms"][transform_key]["matrix"]
    except KeyError as exc:
        raise ValueError(f"{result_path}: missing transform {transform_key}") from exc
    return WorldCalibration(
        world_frame=expected_world,
        camera_frame=expected_camera,
        world_T_camera=_validate_transform(matrix, transform_key),
        map_sha256=actual_map_sha,
        result_path=result_path,
    )


def cam_to_world(points: np.ndarray, world_T_camera: np.ndarray) -> np.ndarray:
    """P_world = world_T_camera @ P_camera。"""
    pts = np.asarray(points, dtype=np.float64)
    transform = np.asarray(world_T_camera, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.concatenate([pts, ones], axis=1) @ transform.T)[:, :3]


def validate_live_intrinsics(
    configured: Intrinsics,
    *,
    width: int,
    height: int,
    k: Any,
    distortion: Any,
    frame_id: str,
    tolerance: float = 1e-3,
) -> None:
    live_k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    live_d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    failures = []
    if (int(width), int(height)) != (configured.width, configured.height):
        failures.append(f"resolution {(width, height)} != {(configured.width, configured.height)}")
    if frame_id != configured.frame_id:
        failures.append(f"frame {frame_id!r} != {configured.frame_id!r}")
    if not np.allclose(live_k, configured.k, atol=tolerance, rtol=0.0):
        failures.append("K differs from calibrated color intrinsics")
    if live_d.shape != configured.distortion.shape or not np.allclose(
        live_d, configured.distortion, atol=tolerance, rtol=0.0
    ):
        failures.append("D differs from calibrated color distortion")
    if failures:
        raise ValueError("live CameraInfo mismatch: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# 几何（内联自 geometry/*）
# ---------------------------------------------------------------------------

def normalize_yaw(yaw_deg: float) -> float:
    """把 180° 对称的 yaw 归一化到 [-90, 90)。"""
    return float((float(yaw_deg) + 90.0) % 180.0 - 90.0)


def yaw_error_deg(first: float, second: float) -> float:
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
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return bool(n[2] >= threshold)


def _fit_plane_ls(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d


def fit_top_plane_fast(points_world: np.ndarray, distance_threshold: float = 0.008) -> tuple[np.ndarray, float, np.ndarray, float]:
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


def project_world_points_to_image(
    points_world: np.ndarray,
    world_t_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.nanmedian(stacked, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ROS 2 RGB-D 源（内联自 camera/ros_rgbd.py）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    aligned_depth_m: np.ndarray
    intrinsics: Intrinsics
    color_stamp_ns: int
    depth_stamp_ns: int
    pair_offset_s: float


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _decode_color_image(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding {message.encoding!r}")
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding == "rgb8":
        image = image[..., ::-1]
    elif encoding == "bgra8":
        image = image[..., :3]
    elif encoding == "rgba8":
        image = image[..., [2, 1, 0]]
    return np.ascontiguousarray(image)


def _decode_depth_image(message: Any, integer_scale_m: float = 0.001) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        row_values = int(message.step) // 2
        row = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_values)
        depth = row[:, : message.width].astype(np.float32) * float(integer_scale_m)
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        row_values = int(message.step) // 4
        row = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_values)
        depth = row[:, : message.width].astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return depth


class ROSAlignedRGBDSource:
    def __init__(
        self,
        configured_intrinsics: Intrinsics,
        *,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        max_pair_offset_s: float = 0.05,
        depth_integer_scale_m: float = 0.001,
        intrinsics_tolerance: float = 1e-3,
    ) -> None:
        try:
            import rclpy
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python interfaces unavailable. Run 'source /opt/ros/humble/setup.bash' "
                "and do not set PYTHONPATH=src."
            ) from exc
        self._rclpy = rclpy
        self._configured = configured_intrinsics
        self._max_pair_ns = int(max_pair_offset_s * 1e9)
        self._depth_scale = float(depth_integer_scale_m)
        self._intrinsics_tolerance = float(intrinsics_tolerance)
        self._colors: deque[tuple[int, Any]] = deque(maxlen=3)
        self._depths: deque[tuple[int, Any]] = deque(maxlen=3)
        self._camera_info: Any | None = None
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = rclpy.create_node("stack_box_mapper_rgbd_source")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(Image, color_topic, self._on_color, qos)
        self.node.create_subscription(Image, depth_topic, self._on_depth, qos)
        self.node.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, qos)

    def _on_color(self, message: Any) -> None:
        self._colors.append((_stamp_ns(message), message))

    def _on_depth(self, message: Any) -> None:
        self._depths.append((_stamp_ns(message), message))

    def _on_camera_info(self, message: Any) -> None:
        self._camera_info = message

    def _take_pair(self) -> RGBDFrame | None:
        if not self._colors or not self._depths or self._camera_info is None:
            return None
        best: tuple[int, int, int, int] | None = None
        for ci, (cs, _) in enumerate(self._colors):
            for di, (ds, _) in enumerate(self._depths):
                delta = abs(cs - ds)
                candidate = (delta, -min(cs, ds), ci, di)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        delta, _newest, ci, di = best
        if delta > self._max_pair_ns:
            if self._colors[0][0] < self._depths[0][0]:
                self._colors.popleft()
            else:
                self._depths.popleft()
            return None
        cs, cm = self._colors[ci]
        ds, dm = self._depths[di]
        color = _decode_color_image(cm)
        depth = _decode_depth_image(dm, self._depth_scale)
        if color.shape[:2] != depth.shape:
            raise RuntimeError("aligned depth shape does not match color")
        info = self._camera_info
        validate_live_intrinsics(
            self._configured,
            width=info.width,
            height=info.height,
            k=info.k,
            distortion=info.d,
            frame_id=info.header.frame_id,
            tolerance=self._intrinsics_tolerance,
        )
        self._colors.clear()
        self._depths.clear()
        return RGBDFrame(
            color_bgr=color,
            aligned_depth_m=depth,
            intrinsics=self._configured,
            color_stamp_ns=cs,
            depth_stamp_ns=ds,
            pair_offset_s=(ds - cs) / 1e9,
        )

    def read(self, timeout_s: float = 10.0) -> RGBDFrame:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
            frame = self._take_pair()
            if frame is not None:
                return frame
        raise TimeoutError("timed out waiting for synchronized color/aligned depth")

    def close(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self) -> "ROSAlignedRGBDSource":
        return self

    def __exit__(self, _t, _v, _tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# YOLO-Seg 封装（内联自 segmentation/yolo_segmentor.py）
# ---------------------------------------------------------------------------

class YOLOSegmentor:
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
                "未找到 ultralytics，请在包含模型依赖的环境中运行（pip install ultralytics）"
            ) from exc
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.mask_threshold = float(mask_threshold)
        self.classes = None if classes is None else [int(v) for v in classes]
        self.model = YOLO(str(weight_path))

    def segment(self, color_bgr: np.ndarray) -> list[Any]:
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
# 托盘检测（内联自 geometry/tray_detection.py）
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
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"map_sha256": map_sha256, "tray": tray}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 坐标变换（内联自 real_pipeline.py）
# ---------------------------------------------------------------------------

def pose_world_to_ref(x: float, y: float, tray_pose: dict[str, Any]) -> tuple[float, float]:
    """世界 XY → 托盘局部 XY。"""
    yaw = np.deg2rad(float(tray_pose["yaw_deg"]))
    dx = float(x) - float(tray_pose["x_m"])
    dy = float(y) - float(tray_pose["y_m"])
    c, s = np.cos(yaw), np.sin(yaw)
    return float(c * dx + s * dy), float(-s * dx + c * dy)


def pose_ref_to_world(
    x_ref: float, y_ref: float, yaw_ref_deg: float, tray_pose: dict[str, Any]
) -> tuple[float, float, float]:
    """托盘局部 (x, y, yaw) → 世界 (x, y, yaw)。"""
    yaw = np.deg2rad(float(tray_pose["yaw_deg"]))
    c, s = np.cos(yaw), np.sin(yaw)
    x_world = float(tray_pose["x_m"]) + c * x_ref - s * y_ref
    y_world = float(tray_pose["y_m"]) + s * x_ref + c * y_ref
    yaw_world = normalize_yaw(float(tray_pose["yaw_deg"]) + yaw_ref_deg)
    return x_world, y_world, yaw_world


# ---------------------------------------------------------------------------
# 层高与标准编号布局
# ---------------------------------------------------------------------------

def layer_top_zs(tray_z: float) -> list[float]:
    """返回 4 层的标准顶面高度（世界 z）。"""
    ha = BOX_TYPES["A"]["height"]
    hb = BOX_TYPES["B"]["height"]
    return [
        tray_z + ha,
        tray_z + 2 * ha,
        tray_z + 2 * ha + hb,
        tray_z + 2 * ha + 2 * hb,
    ]


def layer_support_z(layer: int, tray_z: float) -> float:
    """第 layer 层的支撑面高度（世界 z）。"""
    tops = layer_top_zs(tray_z)
    if layer == 1:
        return float(tray_z)
    return float(tops[layer - 2])


def layer_from_top_z(top_z: float, tray_z: float) -> tuple[int, float]:
    """观测顶面高度 → 最近的标准层号与偏差。"""
    tops = layer_top_zs(tray_z)
    diffs = [abs(float(top_z) - t) for t in tops]
    layer = int(np.argmin(diffs)) + 1
    return layer, float(top_z) - tops[layer - 1]


def slot_id_from_position(x_ref: float, y_ref: float, length: float, width: float) -> int:
    """托盘局部 XY → 最近的标准槽位编号 1~6。

    布局：3 列沿托盘 +X（箱子短轴 width），2 行沿 +Y（箱子长轴 length）。
    编号：-Y 侧一行 1,2,3（自 -X 到 +X），+Y 侧一行 4,5,6。
    """
    col = int(round(x_ref / max(width, 1e-9))) + 1
    row = int(round(y_ref / max(length, 1e-9) + 0.5))
    col = min(max(col, 0), 2)
    row = min(max(row, 0), 1)
    return row * 3 + col + 1


def standard_slot_world_pose(
    layer: int, slot_id: int, tray_pose: dict[str, Any]
) -> tuple[float, float, float, float, float, float, float]:
    """标准槽位的世界位姿 (x, y, z, yaw, length, width, height)。"""
    box_type = LAYER_BOX_TYPES[layer]
    assert box_type is not None
    length = BOX_TYPES[box_type]["length"]
    width = BOX_TYPES[box_type]["width"]
    height = BOX_TYPES[box_type]["height"]
    col = (slot_id - 1) % 3
    row = (slot_id - 1) // 3
    x_ref = (col - 1) * width
    y_ref = (row - 0.5) * length
    x_world, y_world, yaw_world = pose_ref_to_world(x_ref, y_ref, 90.0, tray_pose)
    z_world = layer_support_z(layer, float(tray_pose["z_m"])) + height / 2.0
    return x_world, y_world, z_world, yaw_world, length, width, height


# ---------------------------------------------------------------------------
# 单箱 4DoF 估计（多箱型）
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

    top_z = float(np.median(top_points[:, 2]))
    tray_z = float(tray_pose["z_m"])
    layer, layer_error = layer_from_top_z(top_z, tray_z)
    if abs(layer_error) > args.layer_height_tolerance:
        candidate.reasons.append(f"layer_dz={layer_error * 1000.0:+.0f}mm")
    box_type = LAYER_BOX_TYPES[layer] if 1 <= layer <= LAYER_COUNT else None
    if box_type is None:
        candidate.reasons.append(f"layer={layer}")
    else:
        expected_length = BOX_TYPES[box_type]["length"]
        expected_width = BOX_TYPES[box_type]["width"]
        expected_height = BOX_TYPES[box_type]["height"]
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
                min_size_ratio=SIZE_RATIO_MIN,
                max_size_ratio=SIZE_RATIO_MAX,
                min_width_ratio=SIZE_RATIO_WIDTH_MIN,
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


def update_boxmap(
    boxmap: BoxMap,
    accepted: list[CandidateBox],
    tray: dict[str, Any],
    map_sha256: str | None,
) -> BoxMap:
    """增量更新 boxmap。

    冻结规则：识别到更高一层的箱子时，冻结前一层。
    - 最高可见层（活动层）每帧用实测实时更新（source="measured"）；
    - 低于活动层的层一旦出现更高层，即冻结（source="frozen"），位置锁定为
      该层作为活动层时最后一次的实测值，之后不再随帧更新；
    - 中途打开时看不到的层（从未出现过）按标准位置补全 6 箱（source="standard"）。
    """
    tray_pose = tray["pose_4dof"]
    measured_by_layer: dict[int, list[CandidateBox]] = {}
    for c in accepted:
        measured_by_layer.setdefault(c.layer, []).append(c)
    active_layer = max((c.layer for c in accepted), default=0)

    def _box(c: CandidateBox, source: str) -> BoxState:
        return BoxState(
            id=c.slot_id,
            layer=c.layer,
            box_type=c.box_type,
            x=c.x,
            y=c.y,
            z=c.z,
            yaw=c.yaw,
            length=c.length,
            width=c.width,
            height=c.height,
            source=source,
            timestamp=c.yolo_confidence,
        )

    new_boxes: list[BoxState] = []
    for layer in range(1, active_layer + 1):
        cur = measured_by_layer.get(layer, [])
        existing = {b.id: b for b in boxmap.boxes if b.layer == layer}

        if layer == active_layer:
            # 活动层：累积更新 + 短暂丢失保持。
            # 当前帧识别到的槽位更新为实测；未识别到的槽位保留上一次实测，
            # 连续未识别超过阈值才移除，避免上层箱子/机械臂短暂遮挡导致误删。
            merged: dict[int, BoxState] = {}
            for c in cur:
                merged[c.slot_id] = _box(c, "measured")
            for b in existing.values():
                if b.id in merged:
                    continue
                b.missed += 1
                if b.missed <= MISSED_FRAMES_BEFORE_REMOVE:
                    merged[b.id] = b
            new_boxes.extend(merged.values())
            continue

        # 冻结层（已有更高层）：保留已有位置，忽略本帧该层实测。
        if existing:
            for b in existing.values():
                if b.source != "standard":
                    b.source = "frozen"
            new_boxes.extend(existing.values())
        else:
            # 从未出现过（例如中途打开）：按标准位置补全 6 箱。
            box_type = LAYER_BOX_TYPES[layer]
            for slot_id in range(1, BOXES_PER_LAYER + 1):
                x, y, z, yaw, length, width, height = standard_slot_world_pose(
                    layer, slot_id, tray_pose
                )
                new_boxes.append(
                    BoxState(
                        id=slot_id,
                        layer=layer,
                        box_type=box_type,
                        x=x,
                        y=y,
                        z=z,
                        yaw=yaw,
                        length=length,
                        width=width,
                        height=height,
                        source="standard",
                    )
                )

    boxmap.boxes = new_boxes
    boxmap.active_layer = active_layer
    boxmap.map_sha256 = map_sha256 or boxmap.map_sha256
    boxmap.tray_reference = tray
    return boxmap


def save_boxmap(path: Path, boxmap: BoxMap) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "purpose": "stack_box_mapper",
        "world_frame": boxmap.world_frame,
        "map_sha256": boxmap.map_sha256,
        "active_layer": boxmap.active_layer,
        "tray_reference": boxmap.tray_reference,
        "boxes": [asdict(b) for b in boxmap.boxes],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _draw_world_rectangle(
    image: np.ndarray,
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
    yaw_rad = math.radians(yaw)
    ux = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    uy = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)])
    center = np.array([x, y])
    corners = np.array(
        [center + sx * length * 0.5 * ux + sy * width * 0.5 * uy
         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
        dtype=np.float64,
    )
    axes = np.array(
        [
            [x + 0.18 * math.cos(yaw_rad), y + 0.18 * math.sin(yaw_rad), top_z],
            [x - 0.14 * math.sin(yaw_rad), y + 0.14 * math.cos(yaw_rad), top_z],
        ],
        dtype=np.float64,
    )
    points = np.vstack(
        [np.array([[x, y, top_z]]), axes, np.column_stack([corners, np.full(4, top_z)])]
    )
    pixels = project_world_points_to_image(
        points,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["k"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["distortion"], dtype=np.float64),
    )
    if not np.all(np.isfinite(pixels)):
        return
    p = [tuple(np.round(pt).astype(int)) for pt in pixels]
    cv2.polylines(
        image, [np.asarray(p[3:], dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness
    )
    cv2.arrowedLine(image, p[0], p[1], color, thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, p[0], p[2], (255, 255, 0), thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(image, label, (p[0][0] + 6, p[0][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def _draw_candidate_masks(
    image: np.ndarray, candidates: list[CandidateBox], show_rejected: bool
) -> None:
    overlay = image.copy()
    for c in candidates:
        if c.accepted:
            color = (40, 220, 40)
            overlay[c.mask] = color
        elif show_rejected:
            color = (40, 40, 230)
        else:
            continue
        contours, _ = cv2.findContours(
            c.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        if c.bbox is not None and len(c.bbox.reshape(-1)) >= 4:
            x1, y1, x2, y2 = np.round(c.bbox.reshape(-1)[:4]).astype(int)
            label = (
                f"L{c.layer} slot{c.slot_id} {c.box_type} G={c.geometry_score:.2f}"
                if c.accepted
                else "REJECT " + ",".join(c.reasons[:2])
            )
            cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, dst=image)


def _draw_status(
    image: np.ndarray,
    active_layer: int,
    box_count: int,
    inference_ms: float,
    display_fps: float,
) -> None:
    lines = [
        f"active_layer=L{active_layer}  boxes={box_count}  infer={inference_ms:.1f}ms",
        f"display={display_fps:.1f} FPS",
        "Q: save+quit  S: save state",
    ]
    for index, line in enumerate(lines):
        y = 28 + index * 27
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)


# ---------------------------------------------------------------------------
# 3D 可视化（内联自 scripts/view_stack_map_3d.py，matplotlib 懒加载）
# ---------------------------------------------------------------------------

def _cuboid_vertices(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
) -> np.ndarray:
    length, width, height = size
    local = np.array(
        [
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=float,
    )
    angle = np.deg2rad(float(yaw_deg))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return local @ rotation.T + np.asarray(center, dtype=float)


def _draw_cuboid(ax: Any, vertices: np.ndarray, color: Any, alpha: float = 0.34) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = [
        [vertices[index] for index in (0, 1, 2, 3)],
        [vertices[index] for index in (4, 5, 6, 7)],
        [vertices[index] for index in (0, 1, 5, 4)],
        [vertices[index] for index in (1, 2, 6, 5)],
        [vertices[index] for index in (2, 3, 7, 6)],
        [vertices[index] for index in (3, 0, 4, 7)],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=alpha,
        )
    )


def _draw_3d_scene(
    ax: Any,
    boxmap: BoxMap,
    *,
    elev: float,
    azim: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    std = sum(1 for b in boxmap.boxes if b.source == "standard")
    meas = sum(1 for b in boxmap.boxes if b.source == "measured")
    ax.set_title(
        f"BoxMap 3D | active L{boxmap.active_layer} | "
        f"standard={std} measured={meas}"
    )

    tray = boxmap.tray_reference
    if isinstance(tray, dict):
        tray_pose = tray.get("pose_4dof", {})
        tray_size = tray.get("measured_size_m", {})
        if {"x_m", "y_m", "z_m", "yaw_deg"} <= tray_pose.keys() and {
            "length", "width"
        } <= tray_size.keys():
            tray_height = float(
                tray_size.get("height", tray_size.get("top_height_above_ground", 0.10))
            )
            vertices = _cuboid_vertices(
                (float(tray_pose["x_m"]), float(tray_pose["y_m"]),
                 float(tray_pose["z_m"]) - tray_height / 2.0),
                (float(tray_size["length"]), float(tray_size["width"]), tray_height),
                float(tray_pose["yaw_deg"]),
            )
            _draw_cuboid(ax, vertices, "saddlebrown", alpha=0.22)
            tray_x_axis = np.array(
                [np.cos(np.deg2rad(float(tray_pose["yaw_deg"]))),
                 np.sin(np.deg2rad(float(tray_pose["yaw_deg"]))), 0.0]
            )
            tray_y_axis = np.array([-tray_x_axis[1], tray_x_axis[0], 0.0])
            origin = np.array(
                [float(tray_pose["x_m"]), float(tray_pose["y_m"]), float(tray_pose["z_m"])]
            )
            x_len = min(0.35, float(tray_size["length"]) / 2.0)
            y_len = min(0.35, float(tray_size["width"]) / 2.0)
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.006,
                tray_x_axis[0] * x_len, tray_x_axis[1] * x_len, 0.0,
                color="gold", linewidth=2.5, arrow_length_ratio=0.12,
            )
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.008,
                tray_y_axis[0] * y_len, tray_y_axis[1] * y_len, 0.0,
                color="cyan", linewidth=2.5, arrow_length_ratio=0.12,
            )
            ax.text(
                float(tray_pose["x_m"]), float(tray_pose["y_m"]),
                float(tray_pose["z_m"]) + 0.02, "TRAY",
                color="saddlebrown", weight="bold",
            )

    for box in sorted(boxmap.boxes, key=lambda item: (item.layer, item.id)):
        color = {"standard": "orange", "frozen": "dodgerblue", "measured": "limegreen"}.get(
            box.source, "gray"
        )
        alpha = 0.28 if box.source == "standard" else 0.55
        vertices = _cuboid_vertices(
            (box.x, box.y, box.z), (box.length, box.width, box.height), box.yaw
        )
        _draw_cuboid(ax, vertices, color, alpha=alpha)
        ax.text(
            box.x, box.y, box.z + box.height / 2.0 + 0.02,
            f"L{box.layer}-{box.id} {box.box_type}",
            color=color, weight="bold", fontsize=8,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _preload_mplot3d() -> None:
    """从当前解释器的 site-packages 加载新版 mpl_toolkits.mplot3d 到 sys.modules。

    部分环境里系统 site-packages 残留旧版 mpl_toolkits（通过 pkg_resources
    namespace 机制劫持 import），导致 ``from mpl_toolkits.mplot3d import Axes3D``
    加载旧版并报 ``cannot import name 'docstring'``。这里在 import matplotlib 之前
    手动加载与当前 matplotlib 配套的 mpl_toolkits，规避该问题。
    """
    import importlib.util
    import os
    import sys
    import types

    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]

    tk_dir = None
    for p in sys.path:
        if not p:
            continue
        axes3d_path = os.path.join(p, "mpl_toolkits", "mplot3d", "axes3d.py")
        if not os.path.isfile(axes3d_path):
            continue
        try:
            with open(axes3d_path, encoding="utf-8") as fh:
                if "_docstring" in fh.read(4000):  # 新版 mpl_toolkits 使用 _docstring
                    tk_dir = os.path.join(p, "mpl_toolkits")
                    break
        except OSError:
            continue
    if tk_dir is None:
        return

    tk = types.ModuleType("mpl_toolkits")
    tk.__path__ = [tk_dir]
    sys.modules["mpl_toolkits"] = tk

    m3_dir = os.path.join(tk_dir, "mplot3d")
    spec = importlib.util.spec_from_file_location(
        "mpl_toolkits.mplot3d",
        os.path.join(m3_dir, "__init__.py"),
        submodule_search_locations=[m3_dir],
    )
    if spec is None or spec.loader is None:
        return
    m3 = importlib.util.module_from_spec(spec)
    sys.modules["mpl_toolkits.mplot3d"] = m3
    spec.loader.exec_module(m3)


class BoxMap3DViewer:
    """在独立线程中运行 matplotlib 3D 窗口，定时重绘最新 boxmap。"""

    def __init__(
        self,
        *,
        elev: float = 24.0,
        azim: float = -58.0,
        x_min: float = 0.5,
        x_max: float = 2.0,
        y_min: float = -1.5,
        y_max: float = 0.0,
        z_max: float = 1.8,
    ) -> None:
        self.elev = elev
        self.azim = azim
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_max = z_max
        self._lock = threading.Lock()
        self._boxmap: BoxMap | None = None
        self._dirty = False
        self._drawn_once = False
        self._thread: threading.Thread | None = None

    def update(self, boxmap: BoxMap) -> None:
        with self._lock:
            self._boxmap = boxmap
            self._dirty = True

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            print(
                f"WARNING: 3D 可视化启动失败（{type(exc).__name__}: {exc}）。"
                f"可能是当前环境 matplotlib/mpl_toolkits 不匹配。"
                f"可加 --no-show-3d 禁用 3D 窗口，仅保留 2D 画面。"
            )

    def _run_impl(self) -> None:
        import matplotlib

        _preload_mplot3d()
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # matplotlib 只在 import 时注册一次 "3d" projection；若当时 Axes3D 因
        # 系统旧版 mpl_toolkits 加载失败而未被注册，这里显式补注册。
        try:
            matplotlib.projections.projection_registry.register(Axes3D)
        except Exception:
            pass

        fig = plt.figure("BoxMap 3D", figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        def on_key(event: Any) -> None:
            if event.key in ("q", "Q", "escape"):
                plt.close(fig)

        def on_scroll(event: Any) -> None:
            factor = 0.85 if event.button == "up" else 1.18
            for getter, setter in (
                (ax.get_xlim3d, ax.set_xlim3d),
                (ax.get_ylim3d, ax.set_ylim3d),
                (ax.get_zlim3d, ax.set_zlim3d),
            ):
                low, high = getter()
                center = (low + high) / 2.0
                half = (high - low) * factor / 2.0
                setter(center - half, center + half)
            fig.canvas.draw_idle()

        def refresh() -> None:
            with self._lock:
                boxmap = self._boxmap
                dirty = self._dirty
                self._dirty = False
            if dirty and boxmap is not None:
                # 保留用户拖动的旋转视角和滚轮缩放；仅首次使用默认值。
                if self._drawn_once:
                    elev, azim = ax.elev, ax.azim
                    x_min, x_max = ax.get_xlim3d()
                    y_min, y_max = ax.get_ylim3d()
                    z_min, z_max = ax.get_zlim3d()
                else:
                    elev, azim = self.elev, self.azim
                    x_min, x_max = self.x_min, self.x_max
                    y_min, y_max = self.y_min, self.y_max
                    z_min, z_max = 0.0, self.z_max
                    self._drawn_once = True
                _draw_3d_scene(
                    ax,
                    boxmap,
                    elev=elev,
                    azim=azim,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                )
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        refresh()
        timer = fig.canvas.new_timer(interval=500)
        timer.add_callback(refresh)
        timer.start()
        plt.show(block=True)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["update_tray", "map_stack"],
        required=True,
        help="update_tray: 检测空托盘并更新托盘参考文件；map_stack: 加载托盘并建图",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument("--tray-reference", type=Path, default=ROOT / "record/tray_reference/tray_reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / "record/stack_box_map")
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-conf", type=float, default=0.2)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.50)
    parser.add_argument("--inference-hz", type=float, default=10.0)
    parser.add_argument("--depth-median-frames", type=int, default=3)
    parser.add_argument("--read-timeout", type=float, default=8.0)
    parser.add_argument("--timing-every", type=int, default=10, help="每 N 次推理打印分段耗时，0 禁用")

    parser.add_argument("--min-mask-pixels", type=int, default=1200)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.55)
    parser.add_argument("--min-top-points", type=int, default=450)
    parser.add_argument("--max-geometry-points", type=int, default=6000)
    parser.add_argument("--plane-threshold", type=float, default=0.008)
    parser.add_argument("--min-normal-z", type=float, default=0.97)
    parser.add_argument("--max-plane-rmse", type=float, default=0.008)
    parser.add_argument("--min-top-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-rectangle-fill", type=float, default=0.50)
    parser.add_argument("--min-top-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--layer-height-tolerance", type=float, default=LAYER_HEIGHT_TOLERANCE)
    parser.add_argument("--show-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug-reject",
        action="store_true",
        help="打印每个被拒绝候选的完整拒绝原因，用于定位识别不到的问题",
    )
    parser.add_argument(
        "--show-3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 matplotlib 3D 窗口，显示所有箱子（冻结层+活动层）",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> tuple[Intrinsics, WorldCalibration, dict[str, Any]]:
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    return intrinsics, calibration, config


def _make_manifest(intrinsics: Intrinsics, calibration: WorldCalibration) -> dict[str, Any]:
    return {
        "world_T_camera": calibration.world_T_camera.tolist(),
        "intrinsics": {
            "k": intrinsics.k.tolist(),
            "distortion": intrinsics.distortion.tolist(),
        },
    }


def _make_source(
    intrinsics: Intrinsics, config: dict[str, Any]
) -> ROSAlignedRGBDSource:
    ros_config = config["ros"]
    camera_config = config["camera"]
    return ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros_config["color_topic"],
        depth_topic=ros_config["aligned_depth_topic"],
        camera_info_topic=ros_config["camera_info_topic"],
        max_pair_offset_s=float(ros_config["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera_config["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros_config["intrinsics_tolerance"]),
    )


def mode_update_tray(args: argparse.Namespace) -> int:
    intrinsics, calibration, config = _load_config(args)
    ray_x, ray_y = build_ray_lookup(intrinsics)
    print("========== update_tray mode ==========")
    print("将连续读取深度帧，取中值后检测空托盘，并更新托盘参考文件")
    print(f"托盘参考文件: {args.tray_reference.expanduser().resolve()}")

    with _make_source(intrinsics, config) as source:
        depth_frames: deque[np.ndarray] = deque(maxlen=args.depth_median_frames)
        color_bgr: np.ndarray | None = None
        while len(depth_frames) < args.depth_median_frames:
            frame = source.read(args.read_timeout)
            depth_frames.append(frame.aligned_depth_m)
            if color_bgr is None:
                color_bgr = frame.color_bgr

        median_depth = median_depth_masked(depth_frames)
        tray, artifacts = detect_tray_from_depth(
            median_depth,
            ray_x,
            ray_y,
            calibration.world_T_camera,
            frame=calibration.world_frame,
        )
        save_tray_reference(args.tray_reference, tray, calibration.map_sha256)
        print("托盘识别结果:")
        print(f"  pose_4dof: x={tray['pose_4dof']['x_m']:.4f} y={tray['pose_4dof']['y_m']:.4f} "
              f"z={tray['pose_4dof']['z_m']:.4f} yaw={tray['pose_4dof']['yaw_deg']:.2f}°")
        print(f"  size: {tray['measured_size_m']['length']:.4f} x {tray['measured_size_m']['width']:.4f} m")
        print(f"  axis_ratio={tray['quality']['axis_ratio']:.3f} "
              f"yaw_stable={tray['quality']['yaw_stable_from_shape']}")
        print(f"已写入: {args.tray_reference.expanduser().resolve()}")

        if color_bgr is not None:
            manifest = _make_manifest(intrinsics, calibration)
            display = color_bgr.copy()
            _draw_world_rectangle(
                display,
                x=float(tray["pose_4dof"]["x_m"]),
                y=float(tray["pose_4dof"]["y_m"]),
                top_z=float(tray["pose_4dof"]["z_m"]),
                length=float(tray["measured_size_m"]["length"]),
                width=float(tray["measured_size_m"]["width"]),
                yaw=float(tray["pose_4dof"]["yaw_deg"]),
                manifest=manifest,
                color=(0, 165, 255),
                label="TRAY",
                thickness=3,
            )
            cv2.namedWindow("tray detection", cv2.WINDOW_NORMAL)
            while True:
                cv2.imshow("tray detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            cv2.destroyAllWindows()
    return 0


def mode_map_stack(args: argparse.Namespace) -> int:
    intrinsics, calibration, config = _load_config(args)
    tray = load_tray_reference(args.tray_reference, calibration.map_sha256)
    tray_pose = tray["pose_4dof"]
    ray_x, ray_y = build_ray_lookup(intrinsics)
    manifest = _make_manifest(intrinsics, calibration)
    segmentor = YOLOSegmentor(
        str(args.yolo_weights),
        device=args.yolo_device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        mask_threshold=args.yolo_mask_threshold,
    )

    output = args.output.expanduser().resolve()
    boxmap_path = output / "boxmap.json"
    depth_frames: deque[np.ndarray] = deque(maxlen=args.depth_median_frames)
    candidates: list[CandidateBox] = []
    accepted: list[CandidateBox] = []
    last_inference = float("-inf")
    inference_ms = 0.0
    last_frame_time = time.monotonic()
    display_fps = 0.0
    timing_ms = {"yolo": 0.0, "median": 0.0, "geom": 0.0, "save": 0.0}
    timing_cycles = 0
    boxmap = BoxMap()
    viewer = BoxMap3DViewer() if args.show_3d else None

    print("========== map_stack mode ==========")
    print(f"托盘位姿: x={tray_pose['x_m']:.4f} y={tray_pose['y_m']:.4f} "
          f"z={tray_pose['z_m']:.4f} yaw={tray_pose['yaw_deg']:.2f}°")
    print("冻结层将按标准位置补全，活动层实时识别")
    print(f"boxmap: {boxmap_path}")
    if viewer is not None:
        viewer.start()

    with _make_source(intrinsics, config) as source:
        cv2.namedWindow("stack box mapper", cv2.WINDOW_NORMAL)
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
                t0 = time.monotonic()
                instances = segmentor.segment(frame.color_bgr)
                # 深度兜底：深度区域减去 YOLO 已识别 mask，找出 YOLO 遗漏的箱子。
                yolo_union = None
                if instances:
                    yolo_union = np.asarray(instances[0]["mask"], dtype=bool).copy()
                    for inst in instances[1:]:
                        yolo_union |= np.asarray(inst["mask"], dtype=bool)
                fallback = depth_fallback_instances(
                    frame.aligned_depth_m,
                    tray_pose,
                    ray_x,
                    ray_y,
                    calibration.world_T_camera,
                    exclude_mask=yolo_union,
                )
                instances = merge_instances(instances, fallback)
                t1 = time.monotonic()
                candidates = []
                if instances:
                    union_mask = np.asarray(instances[0]["mask"], dtype=bool).copy()
                    for inst in instances[1:]:
                        union_mask |= np.asarray(inst["mask"], dtype=bool)
                    median_depth = median_depth_masked(depth_frames, union_mask)
                    t2 = time.monotonic()
                    for inst in instances:
                        try:
                            candidates.append(
                                estimate_candidate(
                                    inst, median_depth, ray_x, ray_y,
                                    calibration.world_T_camera, tray_pose, args,
                                )
                            )
                        except Exception as exc:
                            candidates.append(
                                CandidateBox(
                                    accepted=False,
                                    reasons=[f"geometry:{type(exc).__name__}"],
                                    mask=np.asarray(inst["mask"], dtype=bool),
                                    bbox=np.asarray(inst["bbox"], dtype=np.float64),
                                    yolo_confidence=float(inst["confidence"]),
                                )
                            )
                    t3 = time.monotonic()
                else:
                    t2 = t1
                    t3 = t1
                if args.debug_reject:
                    for c in candidates:
                        if not c.accepted:
                            print(f"[reject] layer={c.layer} box_type={c.box_type} "
                                  f"reasons={c.reasons}")
                accepted = deduplicate_by_slot(candidates)
                boxmap = update_boxmap(boxmap, accepted, tray, calibration.map_sha256)
                save_boxmap(boxmap_path, boxmap)
                if viewer is not None:
                    viewer.update(boxmap)
                t4 = time.monotonic()
                inference_ms = (t4 - t0) * 1000.0
                last_inference = t4
                if args.timing_every > 0:
                    timing_ms["yolo"] += (t1 - t0) * 1000.0
                    timing_ms["median"] += (t2 - t1) * 1000.0
                    timing_ms["geom"] += (t3 - t2) * 1000.0
                    timing_ms["save"] += (t4 - t3) * 1000.0
                    timing_cycles += 1
                    if timing_cycles >= args.timing_every:
                        avg = {k: v / timing_cycles for k, v in timing_ms.items()}
                        print(
                            f"[timing] n={timing_cycles} | yolo={avg['yolo']:.1f}ms "
                            f"median={avg['median']:.1f}ms geom={avg['geom']:.1f}ms "
                            f"save={avg['save']:.1f}ms | total={inference_ms:.1f}ms"
                        )
                        timing_ms = {k: 0.0 for k in timing_ms}
                        timing_cycles = 0

            display = frame.color_bgr.copy()
            _draw_candidate_masks(display, candidates, args.show_rejected)
            tray_pose_c = tray_pose
            tray_size = tray.get("measured_size_m", {})
            if "length" in tray_size:
                _draw_world_rectangle(
                    display,
                    x=float(tray_pose_c["x_m"]),
                    y=float(tray_pose_c["y_m"]),
                    top_z=float(tray_pose_c["z_m"]),
                    length=float(tray_size["length"]),
                    width=float(tray_size["width"]),
                    yaw=float(tray_pose_c["yaw_deg"]),
                    manifest=manifest,
                    color=(0, 165, 255),
                    label="TRAY",
                    thickness=2,
                )
            for box in boxmap.boxes:
                if box.source != "measured":
                    continue  # 2D 画面只显示活动层实测箱子；冻结/标准补全层仅 3D 显示
                _draw_world_rectangle(
                    display,
                    x=box.x,
                    y=box.y,
                    top_z=box.z + box.height * 0.5,
                    length=box.length,
                    width=box.width,
                    yaw=box.yaw,
                    manifest=manifest,
                    color=(40, 230, 40),
                    label=f"L{box.layer}-{box.id} {box.box_type}",
                    thickness=3,
                )
            current_time = time.monotonic()
            instant_fps = 1.0 / max(current_time - last_frame_time, 1e-6)
            last_frame_time = current_time
            display_fps = instant_fps if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant_fps
            _draw_status(display, boxmap.active_layer, len(boxmap.boxes), inference_ms, display_fps)
            cv2.imshow("stack box mapper", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                save_boxmap(boxmap_path, boxmap)
                print(f"saved boxmap: {boxmap_path}")
            if key in (ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()
    save_boxmap(boxmap_path, boxmap)
    print(f"finished; boxmap saved to {boxmap_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "update_tray":
        return mode_update_tray(args)
    return mode_map_stack(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)

