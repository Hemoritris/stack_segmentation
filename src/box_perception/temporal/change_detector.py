"""Before / After 时序高度差变化检测（M3）。

职责：由 H_before、H_after 计算 ΔH，经高度/面积阈值与形态学处理得到 Change Mask。
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_change(
    h_before,
    h_after,
    grid_size_m: float = 0.005,
    min_height_diff_m: float = 0.01,
    min_area_m2: float = 0.005,
    morph_kernel: int = 3,
) -> np.ndarray:
    """返回布尔 Change Mask（形状与输入一致）。

    “变化”定义为：新表面出现，或高度明显升高。二者都用于定位刚放置的新箱。
    """
    b = np.asarray(h_before, dtype=np.float64)
    a = np.asarray(h_after, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError("h_before 与 h_after 形状不一致")

    appeared = np.isnan(b) & ~np.isnan(a)
    raised = ~np.isnan(a) & ~np.isnan(b) & ((a - b) > min_height_diff_m)
    change = (appeared | raised).astype(np.uint8)

    k = int(morph_kernel)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, kernel)
    change = cv2.morphologyEx(change, cv2.MORPH_OPEN, kernel)

    min_cells = max(1, int(round(min_area_m2 / (grid_size_m * grid_size_m))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(change, 8)
    mask = np.zeros(change.shape, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_cells:
            mask[labels == i] = True
    return mask

