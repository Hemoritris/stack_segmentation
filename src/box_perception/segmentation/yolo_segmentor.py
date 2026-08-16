"""YOLO-Seg 实例分割封装。

职责：输入 BGR 图像，输出 ``BoxInstance`` 列表。

``ultralytics`` 在构造时延迟导入，保证没有安装视觉依赖时，几何和
RGB-D 预检模块仍然可以正常导入。模型只负责实例分离，后续深度几何
仍然在 ``real_pipeline`` 中完成。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..core.types import BoxInstance


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
        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError("YOLO confidence 必须在 [0, 1] 内")
        if int(imgsz) < 32:
            raise ValueError("YOLO imgsz 过小")
        if not 0.0 < float(mask_threshold) < 1.0:
            raise ValueError("mask_threshold 必须在 (0, 1) 内")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "未找到 ultralytics。请在包含模型依赖的 box-seg 环境中运行，"
                "例如 /home/han/miniforge3/envs/box-seg/bin/python。"
            ) from exc

        self.weights = str(weight_path)
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.mask_threshold = float(mask_threshold)
        self.classes = None if classes is None else [int(value) for value in classes]
        self.model = YOLO(self.weights)

    def segment(self, color_bgr: np.ndarray) -> list[BoxInstance]:
        """对一帧 BGR 图像推理并返回原图尺寸的二值实例 mask。"""
        image = np.asarray(color_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("color_bgr 必须是 HxWx3 的 uint8 图像")
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
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        instances: list[BoxInstance] = []
        for index in range(min(len(masks), len(boxes))):
            mask = np.asarray(masks[index], dtype=np.float32)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            binary = np.ascontiguousarray(mask >= self.mask_threshold, dtype=bool)
            if not np.any(binary):
                continue
            instances.append(
                BoxInstance(
                    mask=binary,
                    bbox=np.asarray(boxes[index], dtype=np.float32),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                )
            )
        return instances
