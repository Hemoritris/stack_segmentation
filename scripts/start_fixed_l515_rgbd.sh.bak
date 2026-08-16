#!/usr/bin/env bash
# Start the physically connected L515 for stack_seg RGB-D perception.

set -euo pipefail
L515_ROS_PREFIX="/home/han/文档/segmentation/yp2orin/realsense-ros-l515-runtime/realsense2_camera"
L515_SDK_PREFIX="/home/han/文档/segmentation/yp2orin/librealsense-l515-runtime"
L515_ROS_DEPS="/home/han/文档/segmentation/yp2orin/realsense-ros-l515-deps/opt/ros/humble/lib"

set +u
source /opt/ros/humble/setup.bash
source /home/han/文档/segmentation/yp2orin/realsense-ros-l515-runtime/setup.sh
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${L515_SDK_PREFIX}/lib:${L515_ROS_DEPS}:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/stack_seg_l515_logs}"
mkdir -p "${ROS_LOG_DIR}"

# This host also has system librealsense 2.58.x installed. L515 operation is
# pinned to the locally built 2.54.2 SDK and matching ROS wrapper; fail closed
# instead of silently loading the unsupported system stack.
ACTUAL_ROS_PREFIX="$(ros2 pkg prefix realsense2_camera 2>/dev/null || true)"
if [[ "${ACTUAL_ROS_PREFIX}" != "${L515_ROS_PREFIX}" ]]; then
  echo "Wrong realsense2_camera package: ${ACTUAL_ROS_PREFIX:-not found}" >&2
  echo "Expected L515 package: ${L515_ROS_PREFIX}" >&2
  exit 1
fi
SDK_VERSION="$(${L515_SDK_PREFIX}/bin/rs-enumerate-devices --version 2>/dev/null | tr -d '\r' | grep -E 'version:' | tail -n 1)"
if [[ "${SDK_VERSION}" != *"2.54.2"* ]]; then
  echo "Wrong librealsense SDK: ${SDK_VERSION:-unknown}; expected 2.54.2" >&2
  exit 1
fi
echo "Using L515-compatible stack: RealSense ROS 4.54.1 + librealsense 2.54.2"
echo "  ROS package: ${ACTUAL_ROS_PREFIX}"
echo "  SDK: ${L515_SDK_PREFIX}"

if ! lsusb -d 8086:0b64 >/dev/null 2>&1; then
  echo "Intel RealSense L515 (8086:0b64) is not present on USB." >&2
  echo "Physically reconnect the L515, wait 3 seconds, then rerun this script." >&2
  exit 1
fi

# L515 accepts only one streaming process. Stop the calibration color-only launch
# or an earlier RGB-D launch before taking the USB device.
LAUNCH_PATTERN='^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch realsense2_camera rs_launch.py camera_name:=fixed_l515 '
if pgrep -f "${LAUNCH_PATTERN}" >/dev/null 2>&1; then
  echo "Stopping existing fixed_l515 driver..."
  pkill -INT -f "${LAUNCH_PATTERN}"
  for _ in $(seq 1 50); do
    pgrep -f "${LAUNCH_PATTERN}" >/dev/null 2>&1 || break
    sleep 0.2
  done
fi
if pgrep -f "${LAUNCH_PATTERN}" >/dev/null 2>&1; then
  echo "Existing fixed_l515 driver did not stop; refusing duplicate USB access." >&2
  exit 1
fi
# Give the kernel/libusb time to release both UVC interfaces before reopening.
sleep 2
if ! lsusb -d 8086:0b64 >/dev/null 2>&1; then
  echo "L515 disappeared while releasing the previous driver." >&2
  echo "Physically reconnect it before retrying; do not start a second driver." >&2
  exit 1
fi

exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:=fixed_l515 \
  serial_no:=f1180517 \
  config_file:="${ROOT_DIR}/config/fixed_l515_rgbd.yaml" \
  rgb_camera.profile:=1280,720,30 \
  depth_module.profile:=1024,768,30 \
  enable_color:=true \
  enable_depth:=true \
  enable_confidence:=false \
  enable_pose:=false \
  enable_fisheye1:=false \
  enable_fisheye2:=false \
  enable_infra1:=false \
  enable_infra2:=false \
  pointcloud.enable:=false \
  align_depth.enable:=true \
  enable_sync:=true \
  publish_tf:=true
