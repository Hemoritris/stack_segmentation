"""相机内外参加载、校验与坐标变换。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    """一个相机流的针孔内参与畸变参数。"""

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
    """冻结的固定相机世界外参及其身份信息。"""

    world_frame: str
    camera_frame: str
    world_T_camera: np.ndarray
    map_sha256: str | None
    result_path: Path


def _load_yaml(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: camera config must be a YAML mapping")
    return path, data


def load_camera_config(config_path: str | Path) -> dict[str, Any]:
    """Return the complete validated YAML mapping for ROS/runtime settings."""
    return _load_yaml(config_path)[1]


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


def cam_to_world(points: np.ndarray, world_T_camera: np.ndarray) -> np.ndarray:
    """Transform camera points with ``P_world = world_T_camera @ P_camera``."""
    pts = np.asarray(points, dtype=np.float64)
    transform = _validate_transform(world_T_camera, "world_T_camera")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 必须是 (N, 3)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.concatenate([pts, ones], axis=1) @ transform.T)[:, :3]


def load_intrinsics(config_path: str | Path, stream: str = "color") -> CameraIntrinsics:
    """从 YAML 读取厂家内参；对齐深度应使用 ``stream='color'``。"""
    path, data = _load_yaml(config_path)
    try:
        entry = data["intrinsics"][stream]
        result = CameraIntrinsics(
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


def load_world_calibration(config_path: str | Path) -> WorldCalibration:
    """读取 two_camera 过滤版 JSON 中的固定 L515 世界外参。"""
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
        raise ValueError(
            f"{result_path}: frame mismatch, expected {expected_world} -> {expected_camera}"
        )
    actual_map_sha = result.get("dataset", {}).get("map_sha256")
    if expected_map_sha and actual_map_sha != expected_map_sha:
        raise ValueError(
            f"{result_path}: map SHA mismatch: expected {expected_map_sha}, got {actual_map_sha}"
        )
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


def load_extrinsics(config_path: str | Path) -> np.ndarray:
    """兼容旧调用：返回 ``world_T_camera`` 4x4 矩阵。"""
    return load_world_calibration(config_path).world_T_camera


def validate_live_intrinsics(
    configured: CameraIntrinsics,
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
