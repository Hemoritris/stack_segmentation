"""ROS 2 固定 L515 彩色/对齐深度同步源。

ROS imports are lazy so geometry tests remain usable outside a sourced ROS shell.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import CameraIntrinsics, validate_live_intrinsics


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    aligned_depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    color_stamp_ns: int
    depth_stamp_ns: int
    pair_offset_s: float


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_color_image(message: Any) -> np.ndarray:
    """Decode common sensor_msgs/Image color encodings without cv_bridge."""
    encoding = str(message.encoding).lower()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding {message.encoding!r}")
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding == "rgb8":
        image = image[..., ::-1]
    elif encoding == "bgra8":
        image = image[..., :3]
    elif encoding == "rgba8":
        image = image[..., [2, 1, 0]]
    return np.ascontiguousarray(image)


def decode_depth_image(message: Any, integer_scale_m: float = 0.001) -> np.ndarray:
    """Decode aligned depth to metres; RealSense ROS publishes 16UC1 in mm."""
    encoding = str(message.encoding).lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        row_values = int(message.step) // 2
        row = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_values)
        depth = row[:, : message.width].astype(np.float32) * float(integer_scale_m)
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        row_values = int(message.step) // 4
        row = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_values)
        depth = row[:, : message.width].astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return depth


class ROSAlignedRGBDSource:
    """Read timestamp-paired color and aligned depth from the existing ROS driver."""

    def __init__(
        self,
        configured_intrinsics: CameraIntrinsics,
        *,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        max_pair_offset_s: float = 0.05,
        depth_integer_scale_m: float = 0.001,
        intrinsics_tolerance: float = 1e-3,
    ) -> None:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python interfaces unavailable. Run "
                "'source /opt/ros/humble/setup.bash' and use /usr/bin/python3. "
                "Do not invoke this script with 'PYTHONPATH=src', because that replaces "
                "the ROS 2 Python search paths."
            ) from exc
        self._rclpy = rclpy
        self._configured = configured_intrinsics
        self._max_pair_ns = int(max_pair_offset_s * 1e9)
        self._depth_scale = float(depth_integer_scale_m)
        self._intrinsics_tolerance = float(intrinsics_tolerance)
        self._colors: deque[tuple[int, np.ndarray]] = deque(maxlen=12)
        self._depths: deque[tuple[int, np.ndarray]] = deque(maxlen=12)
        self._camera_info: Any | None = None
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = rclpy.create_node("stack_seg_fixed_l515_rgbd_source")
        self.node.create_subscription(Image, color_topic, self._on_color, qos_profile_sensor_data)
        self.node.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.node.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )

    def _on_color(self, message: Any) -> None:
        try:
            self._colors.append((stamp_ns(message), decode_color_image(message)))
        except ValueError as exc:
            self.node.get_logger().error(str(exc))

    def _on_depth(self, message: Any) -> None:
        try:
            self._depths.append((stamp_ns(message), decode_depth_image(message, self._depth_scale)))
        except ValueError as exc:
            self.node.get_logger().error(str(exc))

    def _on_camera_info(self, message: Any) -> None:
        self._camera_info = message

    def _take_pair(self) -> RGBDFrame | None:
        if not self._colors or not self._depths or self._camera_info is None:
            return None
        best: tuple[int, int, int] | None = None
        for color_index, (color_stamp, _color) in enumerate(self._colors):
            for depth_index, (depth_stamp, _depth) in enumerate(self._depths):
                delta = abs(color_stamp - depth_stamp)
                if best is None or delta < best[0]:
                    best = (delta, color_index, depth_index)
        assert best is not None
        delta, color_index, depth_index = best
        if delta > self._max_pair_ns:
            if self._colors[0][0] < self._depths[0][0]:
                self._colors.popleft()
            else:
                self._depths.popleft()
            return None
        color_stamp, color = self._colors[color_index]
        depth_stamp, depth = self._depths[depth_index]
        if color.shape[:2] != depth.shape:
            raise RuntimeError(
                f"aligned depth shape {depth.shape} does not match color {color.shape[:2]}"
            )
        info = self._camera_info
        validate_live_intrinsics(
            self._configured,
            width=info.width,
            height=info.height,
            k=info.k,
            distortion=info.d,
            frame_id=info.header.frame_id,
            tolerance=self._intrinsics_tolerance,
        )
        self._colors.clear()
        self._depths.clear()
        return RGBDFrame(
            color_bgr=color.copy(),
            aligned_depth_m=depth.copy(),
            intrinsics=self._configured,
            color_stamp_ns=color_stamp,
            depth_stamp_ns=depth_stamp,
            pair_offset_s=(depth_stamp - color_stamp) / 1e9,
        )

    def read(self, timeout_s: float = 10.0) -> RGBDFrame:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
            frame = self._take_pair()
            if frame is not None:
                return frame
        raise TimeoutError("timed out waiting for synchronized fixed-L515 color/aligned depth")

    def close(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self) -> "ROSAlignedRGBDSource":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
