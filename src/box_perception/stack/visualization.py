"""垛堆可视化的无 ROS 数据接口。"""

from __future__ import annotations

from typing import Any


def to_marker_specs(stack_map) -> list[dict[str, Any]]:
    """生成可被 RViz/Open3D/前端适配的参数化箱体 marker 描述。"""
    return [
        {
            "id": int(box.id),
            "label": f"BOX#{box.id:03d} L{box.layer}",
            "center": [float(box.x), float(box.y), float(box.z)],
            "size": [float(box.length), float(box.width), float(box.height)],
            "yaw_deg": float(box.yaw),
            "layer": int(box.layer),
            "supported_by": list(box.supported_by),
            "supports": list(box.supports),
            "confidence": float(box.confidence),
            "size_source": box.size_source,
        }
        for box in stack_map.boxes
    ]


def to_rviz_markers(stack_map):
    """兼容旧接口：返回不依赖 ROS 消息类型的 marker specs。"""
    return to_marker_specs(stack_map)
