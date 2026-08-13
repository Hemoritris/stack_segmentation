"""RGB 与 Depth 对齐。"""

from __future__ import annotations


class RGBDAligner:
    """把深度帧对齐到彩色帧（或反之），供实例 mask 与点云配准使用。"""

    def __init__(self, align_to: str = "color"):
        self.align_to = align_to
        raise NotImplementedError("TODO(M0): 接入 pyrealsense2 rs.align")

    def align(self, color, depth):
        raise NotImplementedError

