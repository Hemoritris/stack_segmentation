"""Latest-frame ROS 2 color source for interactive RGB collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ros_rgbd import decode_color_image, stamp_ns


@dataclass(frozen=True)
class ColorFrame:
    image_bgr: np.ndarray
    stamp_ns: int
    sequence: int


class ROSColorSource:
    """Subscribe to one color topic and expose only the newest decoded frame."""

    def __init__(self, color_topic: str) -> None:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python interfaces unavailable. Source /opt/ros/humble/setup.bash "
                "and run this tool with /usr/bin/python3."
            ) from exc
        self._rclpy = rclpy
        self._latest: ColorFrame | None = None
        self._sequence = 0
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = rclpy.create_node("stack_seg_fixed_l515_rgb_collector")
        self.node.create_subscription(Image, color_topic, self._on_color, qos_profile_sensor_data)

    def _on_color(self, message: Any) -> None:
        try:
            image = decode_color_image(message)
        except ValueError as exc:
            self.node.get_logger().error(str(exc))
            return
        self._sequence += 1
        self._latest = ColorFrame(image.copy(), stamp_ns(message), self._sequence)

    def poll(self, timeout_s: float = 0.01) -> ColorFrame | None:
        self._rclpy.spin_once(self.node, timeout_sec=float(timeout_s))
        return self._latest

    def read(self, timeout_s: float = 12.0) -> ColorFrame:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            frame = self.poll(min(0.1, max(0.0, deadline - time.monotonic())))
            if frame is not None:
                return frame
        raise TimeoutError("timed out waiting for fixed-L515 RGB frames")

    def close(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self) -> "ROSColorSource":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
