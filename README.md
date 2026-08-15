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

真机链路复用现有 ROS 2 RealSense 驱动，不再由 Python 直接占用 USB。当前配置固定为：

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

L515 固件 `1.5.4.1` 可能周期性打印 `control_transfer ... index: 768 ...
Resource temporarily unavailable`。该警告本身不等于图像中断；以三个目标话题存在且下方
预检输出全部 `[PASS]` 为准。若预检超时或日志出现 `No such device`，再按 USB 掉线处理。

终端 2 检查一帧完整链路：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg
/usr/bin/python3 scripts/check_real_rgbd.py \
  --output record/rgbd_preflight
```

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

V1 开发中。M0 的 ROS RGB-D 读取、内外参加载、对齐深度反投影和世界点云代码已经完成，
待固定 L515 真机运行验收；M1 YOLO-Seg 和 M9 多帧稳定仍未实现。模块完成度与验收项以
`rgbd_box_4dof_stack_development_plan.md` 中的 checklist 为准。
