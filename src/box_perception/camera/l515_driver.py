"""L515 相机驱动封装。

职责：封装 pyrealsense2 pipeline，提供稳定的 RGB/Depth 帧获取与设备生命周期管理。
"""

from __future__ import annotations


class L515Driver:
    """L515 RGB-D 相机驱动。"""

    def __init__(self, config_path: str | None = None):
        raise NotImplementedError("TODO(M0): 接入 pyrealsense2 并加载 config/camera.yaml")

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def read_rgbd(self):
        """返回 (color_bgr, depth_uint16, timestamp)。"""
        raise NotImplementedError

