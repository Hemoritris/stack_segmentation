# box_perception

基于固定挂顶 RGB-D 相机（Intel RealSense L515）的箱体 4DoF 位姿估计与垛堆增量建模。

项目目标、技术路线与分阶段验收标准见仓库根目录下的 [`rgbd_box_4dof_stack_development_plan.md`](./rgbd_box_4dof_stack_development_plan.md)。

## 定位

- **V1**：固定尺寸箱子的“新增箱体”识别，输出 `[x, y, z, yaw]`。
- **V2**：垛堆增量式可视化建模（`StackMap`）。
- **V3**：多尺寸箱子识别与 SKU / 箱型匹配。

核心思路：**不重新理解整个垛堆**，而是利用“机器人每次只新增一个箱子”的时序先验，只识别刚放置的新箱子，再把历史结果累积到 `StackMap`。

## 目录结构

```text
stack_seg/
├── pyproject.toml              # 项目元数据与依赖
├── README.md
├── rgbd_box_4dof_stack_development_plan.md
├── config/                     # 相机、工作区、箱型配置
├── docs/                       # 补充文档
├── scripts/                    # 运行入口与一次性实验脚本
├── tests/                      # 单元测试
└── src/
    └── box_perception/
        ├── cli.py              # 命令行入口
        ├── core/               # 公共数据类型与常量
        ├── camera/             # L515 驱动、标定、RGB-D 对齐
        ├── segmentation/       # YOLO-Seg 实例分割
        ├── temporal/           # 高度图、时序变化检测、新箱关联
        ├── geometry/           # 点云、平面、矩形初始化、位姿优化
        ├── tracking/           # 多帧稳定与箱体状态
        ├── evaluation/         # 精度评估与基准脚本
        └── stack/              # StackMap、支撑关系、可视化
```

## 环境与安装

需要 Python 3.10+。

```bash
# 基础依赖
python -m pip install -e ".[dev]"

# 如要跑分割与相机（按实际环境安装）
python -m pip install -e ".[vision]"

# 如要 3D 可视化
python -m pip install -e ".[viz]"
```

`pyrealsense2` 与 `ultralytics` 依赖具体平台与加速库（CUDA / TensorRT），建议按目标部署机（如 Orin）单独确认版本，不要盲目升级。

## 固定 L515 真机 RGB-D 链路

真机链路复用现有 ROS 2 RealSense 驱动，不再由 Python 直接占用 USB。当前默认配置为：

- 彩色：`1280x720@30`；
- 原生深度：`1024x768@30`；
- 感知输入：对齐到彩色图的 `1280x720` 深度；
- 彩色和深度统一使用 L515 硬件时钟，ROS wrapper 开启软件最近帧同步；
- 对齐深度反投影使用厂家彩色 K/D；
- 世界外参读取 `two_camera` 当前 map2 的 `_filtered.json`，并校验地图 SHA256；
- 输出世界坐标系：`slamware_map`。

本机连接固定 L515，终端 1 启动 RGB-D 驱动：

```bash
cd /home/han/文档/segmentation/stack_seg
./scripts/start_fixed_l515_rgbd.sh
```

如果 30 Hz 在当前负载下出现卡顿，可临时降低为 15 Hz：
`L515_COLOR_FPS=15 ./scripts/start_fixed_l515_rgbd.sh`。

本机同时安装了系统 librealsense `2.58.3` 和 L515 专用 librealsense `2.54.2`。
该脚本固定使用专用 `2.54.2` 与 RealSense ROS `4.54.1`，启动时会打印并校验实际路径；
不要用 `/usr/bin/realsense-viewer`（系统 `2.58.3`）启动这台 L515。

该脚本会停止同名的标定彩色-only 驱动，避免两个进程争抢 L515。它应发布：

```text
/fixed_l515/color/image_raw
/fixed_l515/color/camera_info
/fixed_l515/aligned_depth_to_color/image_raw
```

若输出 `L515 is not present` 或驱动出现连续 `No such device`，说明设备已从 USB 总线掉线。
停止驱动、物理重新插拔 L515、等待约 3 秒后再启动；不要同时启动 `two_camera` 的
`start_fixed_l515.sh`。

专用启动脚本会关闭 librealsense 的热插拔监视、错误轮询和主机温度轮询，避免它们阻塞
高分辨率 RGB-D 传输；相机物理重连后必须重启该脚本。启动阶段或长时间运行中偶发一条
`control_transfer ... index: 768` 不等于图像中断，但不应再以约 1.2 秒的固定周期连续出现。
以三个目标话题存在且下方预检输出全部 `[PASS]` 为准；若预检超时或出现 `No such device`，
再按 USB 掉线处理。

终端 2 检查一帧完整链路：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/check_real_rgbd.py \
  --frames 30 \
  --output record/rgbd_preflight
```

`--frames 30` 会额外打印 RGB-D 实际吞吐、帧间隔和同步时间差，适合判断是否仍有卡顿。

这里不要写 `PYTHONPATH=src`。该脚本会自行加入项目的 `src` 目录；手动使用
`PYTHONPATH=src` 会覆盖 ROS 2 设置的 Python 路径，导致 `rclpy` 无法导入。

程序会严格检查实时 CameraInfo 是否仍为标定时的 `1280x720`、K/D 和
`fixed_l515_color_optical_frame`，随后完成：

```text
aligned depth [m]
→ 去畸变反投影到 fixed_l515_color_optical_frame
→ 乘 slamware_map_T_fixed_l515_color_optical_frame
→ slamware_map 世界点云
```

连续录制真实 RGB-D 数据：

```bash
/usr/bin/python3 scripts/record_rgbd.py \
  --output-dir record/before \
  --frames 30 \
  --interval 0.2
```

每帧保存无损彩色 PNG、米制对齐深度 NPY 和时间戳；`manifest.json` 同时冻结内参、外参、
地图哈希和坐标系。`record/` 已被 Git 忽略。

### YOLO RGB 图片交互采集

需要为 YOLO 标注或微调采集彩色图片时，终端 1 仍按上文运行
`./scripts/start_fixed_l515_rgbd.sh`。终端 2 运行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/collect_yolo_rgb.py \
  --output-dir record/yolo_sessions/train_20260815
```

先用鼠标点击图像窗口，让它获得键盘焦点：

- 按 `K`：保存当前一帧原始 RGB 图片；
- 按 `Q`：结束本次采集并退出；
- `Ctrl+C`：也可安全结束。

输出目录只会生成 `rgb_000000.png`、`rgb_000001.png` 等无损 RGB 图片，不保存深度、
相机参数或预览文字。再次使用同一目录会从已有最大编号继续，不会覆盖旧图片。每次调整箱子
数量、位置、角度、遮挡和光照后再按一次 `K`；训练、验证和测试图片建议分别保存到不同目录。

空托盘采集完成后，必须先识别并冻结托盘参考系：

```bash
cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/run_real_pipeline.py \
  --before record/before \
  --tray-only \
  --output record/tray_reference
```

先检查 `record/tray_reference/tray_overlay.png`，确认托盘轮廓、中心和 `tray +X/+Y` 正确。
确认后，`tray_reference.json` 作为当前托盘和当前地图的冻结参考；文件中带有地图 SHA256，
地图不一致时程序会拒绝加载。

采集完 `record/after` 后，对当前 `40 × 30 × 30 cm` 箱体运行真实数据离线 pipeline：

```bash
cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/run_real_pipeline.py \
  --before record/before \
  --after record/after \
  --tray-reference record/tray_reference/tray_reference.json \
  --box-size 0.40 0.30 0.30 \
  --output record/real_result
```

如果要启用固定 L515 专用的顶层箱体 YOLO-Seg 模型，使用包含
`ultralytics` 和 `scipy` 的 Python 环境运行，并追加权重参数：

```bash
cd /home/han/文档/segmentation/stack_seg
PYTHONPATH=src /path/to/python scripts/run_real_pipeline.py \
  --before record/before \
  --after record/after \
  --tray-reference record/tray_reference/tray_reference.json \
  --box-size 0.40 0.30 0.30 \
  --output record/real_result_yolo \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --yolo-device 0 \
  --yolo-conf 0.35 \
  --yolo-imgsz 768
```

当前机器的训练环境可使用：

```bash
/home/han/miniforge3/envs/box-seg/bin/python
```

该模式先用深度前后帧得到变化区域，再将 YOLO 实例与变化区域按重叠度关联，实际用于点云
拟合的 mask 为 `YOLO mask ∩ depth change mask`。如果没有传入 `--yolo-weights`，仍使用原来的
最大深度变化连通区域作为兜底。调试结果会额外保存 `yolo_image_mask.png`，并在
`result.json` 的 `segmentation` 字段记录实际使用的方法、置信度和重叠度。

### 实时 YOLO + RGB-D 测试

新增脚本 `scripts/live_l515_yolo_test.py`。它持续显示固定 L515 RGB 画面、托盘轮廓、
整个垛堆的箱体轮廓、世界/托盘 4DoF 和 FPS。加入 `--show-live-yolo` 后，画面会叠加
YOLO 实例 mask、边界框、置信度、实例数和推理耗时；`--live-yolo-hz` 独立限制推理频率，
不会要求每一张预览帧都执行 YOLO。每次成功后会把本次结果作为下一次 Before 状态，
同时将新箱加入 `StackMap`，自动分配箱体 ID、计算层级和支撑关系。

模式一：启动时重新识别托盘并保存新的托盘参考：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/home/han/venvs/stack-live/bin/python scripts/live_l515_yolo_test.py \
  --mode reidentify_tray \
  --box-size 0.40 0.30 0.30 \
  --tray-frames 5 \
  --start-frames 5 \
  --after-frames 5 \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --yolo-device 0 \
  --yolo-conf 0.35 \
  --show-live-yolo \
  --live-yolo-hz 2 \
  --output record/live_reidentify
```

模式二：不重新识别托盘，直接复用已有托盘参考：

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
  --output record/live_reuse
```

`reidentify_tray` 会先用 `--tray-frames` 帧更新托盘 4DoF，再用 `--start-frames` 帧建立
箱体 Before 状态；`reuse_tray` 会校验托盘参考中的地图 SHA256，然后只建立当前 Before 状态。
输出目录中保存 `tray_reference.json`（模式一）和每次 K 成功后的 `result_001.json`、
`result_002.json` 等，并持续更新 `stack_map.json`。`stack_map.json` 是参数化垛堆地图，
保存每个箱体的世界位姿、实测尺寸、层号、`supported_by` 和 `supports`。
3D 显示默认使用 `measured_size_m` 的 RGB-D 测量结果，不使用预设 CAD 尺寸；只有
测量尺寸缺失时才会记录并显示 `configured_fallback` 回退尺寸。箱体标签会显示当前使用的
长×宽×高（mm）。托盘同样使用 `tray_reference.json` 中的实测长宽高。

继续已有垛堆建模时，使用：

```bash
source /opt/ros/humble/setup.bash
cd /home/han/文档/segmentation/stack_seg
/home/han/venvs/stack-live/bin/python scripts/live_l515_yolo_test.py \
  --mode reuse_tray \
  --tray-reference record/tray_reference/tray_reference.json \
  --resume-stack-map record/live_reuse/stack_map.json \
  --output record/live_reuse
```

如果地图 SHA256 不一致，程序会拒绝加载旧 StackMap，避免把不同地图中的箱体混在一起。

### 最高活动层冻结与 4DoF 跟踪测试

`scripts/live_top_layer_tracker_test.py` 是独立测试程序，不修改正式 StackMap。它根据托盘
顶面与箱体高度把最高有效顶面量化为层号；稳定检测到更高一层后冻结所有旧层，只更新最高
活动层箱体的世界坐标 `x/y/z/yaw`。YOLO 只提供候选 mask，候选还必须通过有效深度比例、
水平顶面 RANSAC、平面误差、实测长宽、顶面面积、矩形填充率、层高和托盘范围检查。
因此只露出一部分但仍被 YOLO 检出的箱体会以红色 `REJECT` 显示，不会进入跟踪状态。

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

显示颜色：绿色为当前活动层确认轨迹，蓝色为冻结层，黄色为暂时丢失后保持的活动轨迹，
红色为未通过几何完整性校验的 YOLO 候选。按 `S` 保存测试状态，按 `Q` 保存并退出。
测试状态只写入 `top_layer_tracker_state.json`；需要断点续测时追加：

```bash
--resume-state record/top_layer_tracking_test/top_layer_tracker_state.json
```

若要得到完整的历史冻结层，应从第一层开始持续运行，或使用 `--resume-state` 接续上次状态。
如果第一次启动时现场已经堆到第三层，程序会正确跟踪第三层，但不会凭被遮挡的画面虚构第一、
二层箱体位置。

开始该功能前的正式文件备份位于
`backups/top_layer_tracking_before_20260815.tar.gz`。

### 交互式 3D 垛堆查看

使用 `scripts/view_stack_map_3d.py` 查看参数化垛堆模型。程序默认持续监视
`stack_map.json`，但只有文件发生变化时才重绘；实时测试程序每次按 `K` 后，三维窗口会自动刷新。
默认坐标范围为 `X=0.5～2.0 m`、`Y=-1.5～0 m`、`Z=0～1.6 m`。

```bash
cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/view_stack_map_3d.py \
  record/live_stack_model/stack_map.json
```

窗口操作：

- 鼠标左键拖动：环绕旋转，支持 360° 查看；
- 鼠标滚轮：缩放；
- `R`：恢复默认视角；
- `Q` 或 `Esc`：退出。

画面会显示托盘、托盘顶面中心原点、与实时画面一致的 `tray +X/tray +Y` 坐标轴、
箱体 ID、层号和支撑虚线。只查看一次、不持续监视时添加
`--no-watch`；保存截图可添加 `--save record/live_stack_model/stack_3d.png`。
如需调整范围，可使用 `--x-min`、`--x-max`、`--y-min`、`--y-max` 和 `--z-max`。

该脚本必须在同一个 Python 环境中同时运行 ROS 2、YOLO 和几何依赖；不能直接用当前只含
`ultralytics` 的 `box-seg` Conda 环境，因为它没有 `rclpy`。如果只使用离线录制流程，仍按
前面的 `run_real_pipeline.py` 使用 `box-seg` 环境。

程序首先从 `before` 深度中分离高于地面的最大水平连通平面，得到托盘在 `slamware_map`
下的 4DoF；随后对前后各 30 帧深度取像素中值，自动定位最大的新增深度区域，构建局部
世界高度图，拟合箱体顶面和固定尺寸矩形，最终同时输出箱体世界 4DoF 与托盘局部 4DoF。
结果保存在
`record/real_result/result.json`，`overlay.png` 显示图像变化区域及估计结果。yaw 定义为箱体
长边相对世界 `+X` 轴、绕世界 `+Z` 轴逆时针的角度；箱体具有 180° 对称性，因此结果统一
到 `[-90°, 90°)`。叠加图中蓝色为世界 `+X`，红色为箱体长轴/yaw，绿色为短轴，紫色为
固定尺寸顶面轮廓，青色点为箱体中心。

托盘局部坐标系原点位于托盘顶面中心，`+X` 沿检测到的长边，`+Y` 沿短边，`+Z` 与世界
向上一致。`result.json` 中的 `tray.pose_4dof` 是托盘世界位姿，
`box_pose_in_tray_4dof` 是箱体相对托盘的位姿，其中 `z_m` 是箱体中心相对托盘顶面的高度，
`bottom_z_m` 是箱底与托盘顶面的间隙。后续放置规划应先在托盘系指定目标，再通过
`pose_4dof_reference_to_world()` 转换到 `slamware_map`，不能直接把托盘局部 XY 当作世界 XY。

调试输出还包括：

- `tray_overlay.png`：空托盘识别轮廓、托盘坐标轴和世界 `+X`；
- `tray_image_mask.png`：托盘图像掩膜；
- `tray_top_points_world.npy`：托盘顶面世界点云；
- `overlay.png`：托盘轮廓、箱体轮廓、两个位姿及箱体相对托盘结果。

不提供 `--tray-reference` 时，完整脚本会为兼容调试流程而从 `before` 自动检测托盘；正式连续
堆叠应始终使用冻结文件。托盘接近正方形时，单靠外接矩形可能发生 90° 轴交换。应检查
`tray.quality.yaw_stable_from_shape` 和 `axis_ratio`；开始堆叠后托盘会被遮挡，因此应在第一只
箱子识别前检测一次并冻结该次托盘位姿，除非确认托盘发生移动。

当前外参只适用于地图 SHA256
`b3cb8f4e94190f047eb447bd8adcf07d730efce8a1245d4ed9814bbe70502a29` 和当前固定相机安装
位置。地图或相机移动后，参数加载会失败或必须重新标定，不能直接修改配置绕过检查。

## 无相机自测

在相机未安装 / 未标定时，可用合成场景验证 M2~M8 的几何链路：

```bash
# 端到端跑一次，打印真值 / 估计 / 误差
python scripts/run_pipeline.py

# 在噪声、紧贴邻箱、两层叠放等退化场景下评估精度
python scripts/benchmark_synthetic.py

# 回归测试
python -m pytest
```

合成相机采用**正俯视正交投影**近似（对高挂顶相机是合理简化，且避免了透视带来的箱边地板/顶面伪重叠）；真实 L515 为透视成像，换真机后需单独验证箱边的视差 / 遮挡效应。

## 开发约定

- **分支模型**：`main` 保持可用；功能开发在 `feature/*` 分支；阶段性可交付打 `release/v*` 标签（对应 V1/V2/V3 验收）。
- **提交信息**：建议遵循 Conventional Commits，如 `feat(temporal): add height-map temporal differencing`、`fix(geometry): ...`、`docs: ...`。
- **合并前**：跑 `python -m pytest`，并确认 `python -m pip install -e ".[dev]"` 无报错。
- **大文件**：模型权重、录包、点云、图像数据默认不入库（见 `.gitignore`），需要版本化时使用 Git LFS 或外部对象存储。

## 当前状态

V1 开发中。M0 已通过固定 L515 真机验收；M2、M3、M5～M8 已使用一组真实 Before/After
数据完成单个新增箱体闭环验证。当前 `40 × 30 × 30 cm` 箱体估计尺寸约为
`39.2 × 30.6 × 31.1 cm`，结果位于 `record/real_result/`（该目录不纳入 Git）。

托盘预识别与托盘局部放置参考系也已通过同一组真机数据验证：托盘顶面约
`0.922 × 0.885 m`，世界 yaw 约 `-88.02°`，箱体相对托盘中心约为
`(-0.175, 0.076) m`、相对 yaw 约 `55.67°`。

当前已经接入固定 L515 专用 YOLO-Seg 模型：M1 的模型封装和 M4 的
“YOLO mask × 深度变化区域”关联已实现；没有提供模型权重时仍保留深度连通区域兜底。
实时 ROS RGB-D 读取、复杂多箱场景和多帧稳定性仍需要真机验证，尚不能视为 V1 最终验收。
模块完成度与验收项以 `rgbd_box_4dof_stack_development_plan.md` 中的 checklist 为准。
