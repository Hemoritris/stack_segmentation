"""相机标定、ROS 2 RGB-D 读取与托盘坐标变换。"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .geometry import normalize_yaw
from .types import Intrinsics, RGBDFrame, WorldCalibration


# ---------------------------------------------------------------------------
# 相机标定
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: camera config must be a YAML mapping")
    return p, data


def load_intrinsics(config_path: str | Path, stream: str = "color") -> Intrinsics:
    """从 YAML 读取厂家内参（对齐深度应使用 stream='color'）。"""
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
    """读取固定 L515 世界外参，并校验 frame 与地图 SHA256。"""
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
    """拒绝与标定分辨率、K/D 或 optical frame 不一致的实时 CameraInfo。"""
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
# ROS 2 RGB-D 源
# ---------------------------------------------------------------------------

def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _decode_color_image(message: Any) -> np.ndarray:
    """解码 sensor_msgs/Image 彩色图像（不依赖 cv_bridge）。"""
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
    """解码对齐深度为米制（RealSense ROS 发布 16UC1，单位 mm）。"""
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
    """读取时间同步的彩色 / 对齐深度帧。"""

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
        """挑选时间戳最接近的一对彩色 / 深度帧。"""
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
# 托盘坐标系变换
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
