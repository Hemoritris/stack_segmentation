"""跨模块共享的数据结构。

设计原则：尽量轻量、无第三方依赖，数组字段统一约定为 numpy.ndarray，
避免各模块各自定义结构导致接口漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BoxInstance:
    """YOLO-Seg 输出的单个实例。"""

    mask: Any  # 二值 mask，numpy bool 数组
    bbox: Any  # xyxy 边界框
    confidence: float
    class_id: Optional[int] = None


@dataclass
class NewBoxObservation:
    """时序差分与 YOLO 关联后得到的新增箱子观察。"""

    instance_mask: Any  # 建议为 instance_mask ∩ change_mask
    roi: Any
    yolo_confidence: float
    change_overlap: float
    timestamp: float = 0.0


@dataclass
class BoxPointCloud:
    """单箱点云，world 坐标系下的 (N, 3) 数组。"""

    points: Any


@dataclass
class BoxTopPlane:
    """箱子顶面拟合结果。"""

    normal: Any  # 3 维法向量，应近似 [0, 0, 1]
    height: float
    points: Any  # 顶面点
    plane_rmse: float


@dataclass
class BoxEstimate:
    """单帧 / 多帧稳定的箱体 4DoF 估计结果。"""

    id: int
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    position_std: float = 0.0
    yaw_std: float = 0.0
    plane_rmse: float = 0.0
    fit_error: float = 0.0
    yolo_confidence: float = 0.0
    change_overlap: float = 0.0
    valid: bool = False


@dataclass
class BoxState:
    """StackMap 中长期维护的箱体状态。"""

    id: int
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    confidence: float = 0.0
    confirmed: bool = False
    layer: int = 0
    supported_by: list[int] = field(default_factory=list)
    supports: list[int] = field(default_factory=list)
    source: str = "fixed_l515"
    size_source: str = "measured_rgbd"
    timestamp: float = 0.0


@dataclass
class StackMap:
    """Box-based 垛堆地图（而非 Layer-based）。"""

    boxes: list[BoxState] = field(default_factory=list)
    schema_version: int = 1
    world_frame: str = "slamware_map"
    map_sha256: str = ""
    tray_reference: dict[str, Any] | None = None
