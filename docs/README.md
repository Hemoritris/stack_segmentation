# 快速运行

## 终端 1：启动固定 L515

```bash
cd /home/han/文档/segmentation/stack_seg
./scripts/start_fixed_l515_rgbd.sh
```

默认使用彩色 `1280x720@30`、深度 `1024x768@30`。物理重新插拔相机后需要重启该脚本；如果本机负载导致卡顿，可用 `L515_COLOR_FPS=15 ./scripts/start_fixed_l515_rgbd.sh` 临时降低彩色帧率。

## 终端 2：实时识别与垛堆建模

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/home/han/venvs/stack-live/bin/python scripts/live_l515_yolo_test.py \
  --mode reuse_tray \
  --tray-reference record/tray_reference/tray_reference.json \
  --box-size 0.40 0.30 0.30 \
  --start-frames 5 \
  --after-frames 5 \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --yolo-device 0 \
  --yolo-conf 0.35 \
  --show-live-yolo \
  --live-yolo-hz 2 \
  --output record/live_stack_model_measured
```

窗口内会叠加 YOLO 分割 mask、边界框和置信度；按 `K` 记录一个新放置箱体，按 `Q` 保存并退出。不要设置 `PYTHONPATH=src`；脚本会自行加载项目源码，同时保留 ROS 2 的 Python 路径。

## 终端 3：交互式 3D 查看

```bash
cd /home/han/文档/segmentation/stack_seg
export MPLCONFIGDIR=/tmp/matplotlib_stack
mkdir -p "$MPLCONFIGDIR"

/usr/bin/python3 scripts/view_stack_map_3d.py \
  record/live_stack_model_measured/stack_map.json
```

## RGB-D 稳定性检查

```bash
source /opt/ros/humble/setup.bash
cd /home/han/文档/segmentation/stack_seg

/usr/bin/python3 scripts/check_real_rgbd.py \
  --frames 30 \
  --output record/rgbd_preflight
```

输出应包含同步时间差、实际吞吐和最大帧间隔，且所有检查均为 `[PASS]`。

## 独立最高层跟踪测试

启动 L515 后，在另一个终端运行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/home/han/venvs/stack-live/bin/python scripts/live_top_layer_tracker_test.py \
  --tray-reference record/tray_reference/tray_reference.json \
  --box-size 0.40 0.30 0.30 \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --yolo-device 0 \
  --inference-hz 2 \
  --output record/top_layer_tracking_test
```

该脚本不会修改正式 StackMap；红色候选表示 YOLO 检出但未通过 RGB-D 完整箱体校验。
完整冻结历史需要从第一层持续运行；中途重启时使用 `--resume-state` 加载测试状态。
