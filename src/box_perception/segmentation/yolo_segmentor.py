"""YOLO-Seg 实例分割封装。

职责：输入 RGB，输出 BoxInstance[]。重点是实例正确分离，而非极高的 mask IoU。
"""

from __future__ import annotations

from ..core.types import BoxInstance


class YOLOSegmentor:
    def __init__(self, weights: str, device: str | None = None, conf: float = 0.25):
        self.weights = weights
        self.device = device
        self.conf = conf
        raise NotImplementedError("TODO(M1): 接入 ultralytics YOLO-Seg")

    def segment(self, color_bgr) -> list[BoxInstance]:
        raise NotImplementedError

