"""多帧稳定性判断：连续采集 5~10 帧后统计均值 / 方差，candidate -> confirmed。"""

from __future__ import annotations


def is_stable(
    estimates,
    pos_std_max_m: float = 0.005,
    yaw_std_max_deg: float = 0.5,
) -> bool:
    raise NotImplementedError("TODO(M9): 计算 x/y/z/yaw 标准差并判断阈值")

