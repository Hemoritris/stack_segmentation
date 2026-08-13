"""mask 处理工具：面积、IoU、与变化区域的重叠度。"""

from __future__ import annotations

import numpy as np


def mask_area(mask) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def mask_iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def change_overlap(instance_mask, change_mask) -> float:
    """实例 mask 与变化区域的重叠度 |M∩C| / |M|。"""
    m = np.asarray(instance_mask, dtype=bool)
    c = np.asarray(change_mask, dtype=bool)
    area = int(np.count_nonzero(m))
    if area == 0:
        return 0.0
    return int(np.count_nonzero(m & c)) / area

