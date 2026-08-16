"""跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Intrinsics:
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
    """Box-based 垛堆地图。"""

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


@dataclass(frozen=True)
class RGBDFrame:
    """一对同步的彩色 / 对齐深度帧。"""

    color_bgr: np.ndarray
    aligned_depth_m: np.ndarray
    intrinsics: Intrinsics
    color_stamp_ns: int
    depth_stamp_ns: int
    pair_offset_s: float
