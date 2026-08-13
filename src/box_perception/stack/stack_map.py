"""增量式 StackMap：Box-based 地图的添加 / 更新 / 删除。"""

from __future__ import annotations

from ..core.types import BoxState, StackMap


class StackMapManager:
    def __init__(self) -> None:
        self.map = StackMap()
        self._next_id = 1

    def add(self, box: BoxState) -> int:
        if box.id < 0:
            box.id = self._next_id
            self._next_id += 1
        self.map.boxes.append(box)
        return box.id

    def update(self, box_id: int, **kwargs) -> bool:
        for box in self.map.boxes:
            if box.id == box_id:
                for key, value in kwargs.items():
                    if hasattr(box, key):
                        setattr(box, key, value)
                return True
        return False

    def get(self, box_id: int) -> BoxState | None:
        for box in self.map.boxes:
            if box.id == box_id:
                return box
        return None

