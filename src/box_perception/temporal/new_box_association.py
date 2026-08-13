"""Change Mask 与 YOLO 实例关联，识别新增箱子。"""

from __future__ import annotations

import numpy as np

from ..core.types import BoxInstance, NewBoxObservation
from ..segmentation.mask_utils import change_overlap


def associate_new_box(
    instances: list[BoxInstance],
    change_mask,
    min_overlap: float = 0.2,
) -> NewBoxObservation | None:
    """选择与变化区域重叠度最高的实例作为新箱。

    关键：新箱 mask 取 instance_mask ∩ change_mask，而不是整个实例，
    避免紧贴 / 粘连实例把邻箱点云带进来。

    Returns:
        找到则返回 NewBoxObservation，否则返回 None（调用方需走兜底路径）。
    """
    best: BoxInstance | None = None
    best_score = -1.0
    for inst in instances:
        score = change_overlap(inst.mask, change_mask)
        if score > best_score:
            best_score = score
            best = inst

    if best is None or best_score < min_overlap:
        return None

    new_mask = np.asarray(best.mask, dtype=bool) & np.asarray(change_mask, dtype=bool)
    return NewBoxObservation(
        instance_mask=new_mask,
        roi=best.bbox,
        yolo_confidence=best.confidence,
        change_overlap=best_score,
    )

