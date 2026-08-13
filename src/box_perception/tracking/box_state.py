"""BoxState 的确认、更新与失效逻辑（配合 core.types.BoxState 使用）。"""

from __future__ import annotations


def confirm_estimate(estimates):
    """从多帧估计生成一个稳定的 BoxState。"""
    raise NotImplementedError("TODO(M9): 取中位数 / 均值作为最终状态")

