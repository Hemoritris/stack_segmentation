"""Before / After 时序高度差变化检测。

职责：由 H_before、H_after 计算 ΔH，经高度/面积阈值与形态学处理得到 Change Mask。
"""

from __future__ import annotations


def detect_change(
    h_before,
    h_after,
    min_height_diff_m: float = 0.01,
    min_area_m2: float = 0.005,
    morph_kernel: int = 3,
):
    raise NotImplementedError("TODO(M3): 差分、阈值、形态学与连通域")

