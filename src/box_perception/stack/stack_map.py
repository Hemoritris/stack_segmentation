"""增量式 StackMap：箱体 ID、层级、支撑关系和 JSON 持久化。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.types import BoxState, StackMap
from .support_graph import build_support_graph


class StackMapManager:
    """维护一次连续码垛过程中的参数化箱体地图。"""

    def __init__(
        self,
        *,
        world_frame: str = "slamware_map",
        map_sha256: str = "",
        tray_reference: dict[str, Any] | None = None,
    ) -> None:
        self.map = StackMap(
            world_frame=world_frame,
            map_sha256=map_sha256,
            tray_reference=tray_reference,
        )
        self._next_id = 1

    @classmethod
    def load(
        cls, path: str | Path, *, expected_map_sha256: str | None = None
    ) -> "StackMapManager":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        actual_hash = str(payload.get("map_sha256", ""))
        if expected_map_sha256 and actual_hash != expected_map_sha256:
            raise ValueError(
                f"StackMap 地图 SHA256 不匹配: {actual_hash} != {expected_map_sha256}"
            )
        manager = cls(
            world_frame=str(payload.get("world_frame", "slamware_map")),
            map_sha256=actual_hash,
            tray_reference=payload.get("tray_reference"),
        )
        manager.map.schema_version = int(payload.get("schema_version", 1))
        manager.map.boxes = [
            manager._box_from_dict(item) for item in payload.get("boxes", [])
        ]
        manager._next_id = max((box.id for box in manager.map.boxes), default=0) + 1
        manager.recompute_relations()
        return manager

    @staticmethod
    def _box_from_dict(item: dict[str, Any]) -> BoxState:
        allowed = {
            "id", "x", "y", "z", "yaw", "length", "width", "height",
            "confidence", "confirmed", "layer", "supported_by", "supports",
            "source", "size_source", "timestamp",
        }
        values = {key: item[key] for key in allowed if key in item}
        values.setdefault("id", -1)
        return BoxState(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.map.schema_version,
            "world_frame": self.map.world_frame,
            "map_sha256": self.map.map_sha256,
            "tray_reference": self.map.tray_reference,
            "box_count": len(self.map.boxes),
            "boxes": [asdict(box) for box in self.map.boxes],
        }

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return resolved

    def add(self, box: BoxState) -> int:
        if box.id < 0:
            box.id = self._next_id
            self._next_id += 1
        elif self.get(box.id) is not None:
            raise ValueError(f"StackMap 中已存在箱体 ID={box.id}")
        self.map.boxes.append(box)
        self.recompute_relations()
        return box.id

    def add_result(self, result: dict[str, Any], *, timestamp: float = 0.0) -> BoxState:
        """把实时 pipeline 的结果转换为一个新的 StackMap 箱体。"""
        pose = result["pose_4dof"]
        # 3D StackMap 优先使用 RGB-D 顶面/平面拟合得到的实测尺寸；
        # 固定箱型尺寸只作为深度测量失败时的回退，不作为默认显示尺寸。
        size = result.get("measured_size_m") or result.get("box_size_prior_m")
        if not size:
            raise ValueError("识别结果缺少箱体尺寸")
        size_source = "measured_rgbd" if result.get("measured_size_m") else "configured_fallback"
        segmentation = result.get("segmentation", {})
        quality = result.get("quality", {})
        confidence = float(
            segmentation.get("confidence", quality.get("confidence", 0.0))
        )
        box = BoxState(
            id=-1,
            x=float(pose["x_m"]),
            y=float(pose["y_m"]),
            z=float(pose["z_m"]),
            yaw=float(pose["yaw_deg"]),
            length=float(size["length"]),
            width=float(size["width"]),
            height=float(size["height"]),
            confidence=confidence,
            confirmed=True,
            source="fixed_l515_yolo_depth",
            size_source=size_source,
            timestamp=float(timestamp),
        )
        self.add(box)
        return box

    def recompute_relations(self) -> None:
        graph = build_support_graph(self.map)
        for box in self.map.boxes:
            box.supported_by = list(graph["supported_by"].get(str(box.id), []))
            box.supports = list(graph["supports"].get(str(box.id), []))
            box.layer = int(graph["layers"].get(str(box.id), 1))

    def update(self, box_id: int, **kwargs: Any) -> bool:
        for box in self.map.boxes:
            if box.id == box_id:
                for key, value in kwargs.items():
                    if hasattr(box, key):
                        setattr(box, key, value)
                self.recompute_relations()
                return True
        return False

    def remove(self, box_id: int) -> bool:
        before = len(self.map.boxes)
        self.map.boxes = [box for box in self.map.boxes if box.id != box_id]
        changed = len(self.map.boxes) != before
        if changed:
            self.recompute_relations()
        return changed

    def get(self, box_id: int) -> BoxState | None:
        for box in self.map.boxes:
            if box.id == box_id:
                return box
        return None
