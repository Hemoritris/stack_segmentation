# box_perception

基于固定挂顶 RGB-D 相机（Intel RealSense L515）的箱体 4DoF 位姿估计与垛堆增量建模。

## 简介

面向机器人码垛场景：识别机器人每次新放置的箱子，输出单个箱子的 4DoF 位姿 `[x, y, z, yaw]`，
并把历史结果累积成一个垛堆模型（StackMap）。

**核心思路**：不重新理解整个垛堆，而是利用“机器人每次只新增一个箱子”的时序先验——
只识别刚放置的新箱子，再把历史结果累积起来。

**三阶段规划**：

| 版本 | 目标 | 状态 |
|---|---|---|
| V1 | 固定尺寸新箱的 4DoF 识别 | 核心链路已闭环 |
| V2 | 垛堆增量式可视化建模（StackMap） | 已实现，真机验证中 |
| V3 | 多尺寸识别与 SKU / 箱型匹配 | 未开始 |

## 实现思路

```text
L515 RGB-D
  ├─ RGB ────→ YOLO-Seg ────→ 实例 mask
  └─ Depth ──→ 世界高度图 ──→ 时序变化 ──→ 变化 mask
                         ↓
                 新箱关联（YOLO mask ∩ 深度变化）
                         ↓
                 单箱点云 → 顶面拟合 → 矩形拟合
                         ↓
                 尺寸先验约束 → [x, y, z, yaw]
                         ↓
                 多帧稳定 → StackMap（层号 / 支撑关系 / 可视化）
```

关键设计：

- **YOLO 只提供候选 mask**，箱体必须通过 RGB-D 几何校验（顶面水平、实测尺寸、顶面面积、
  矩形填充率、层高）才会被确认；只露出一部分或被遮挡的候选会被拒绝。
- **层冻结**：识别到更高一层的箱子时冻结前一层，位置按实测锁定；中途打开时，看不到的层
  按标准位置补全。
- **深度兜底**：YOLO 对个别箱子漏检时，用深度图找“高于托盘顶面”的区域，减去 YOLO 已识别的
  mask，对剩余区域做连通域，得到漏检的候选，再走同样的几何校验。
- **尺寸/层高容差**：短轴（宽度）方向受透视与深度噪声影响、在最底层易测偏小，单独放宽；
  层高容差放宽以适配托盘约 70mm 的边框/垫板结构。

## 代码结构

```text
stack_seg/
├── README.md
├── pyproject.toml                     # 项目元数据与依赖
├── config/                            # 相机、工作区、箱型配置
│   ├── camera.yaml                    # 固定 L515 内外参、ROS 话题
│   ├── fixed_l515_rgbd.yaml           # L515 驱动参数
│   ├── box_types.yaml                 # 箱型库（V3 预留）
│   └── workspace.yaml
├── scripts/                           # 运行入口
│   ├── stack_box_mapper.py            # 正式版垛堆建图（单文件，推荐）
│   ├── start_fixed_l515_rgbd.sh       # 启动固定 L515 驱动
│   ├── check_real_rgbd.py             # 一帧真机 RGB-D 链路检查
│   ├── record_rgbd.py                 # 连续录制 RGB-D
│   ├── run_real_pipeline.py           # 离线真实数据 pipeline
│   ├── run_pipeline.py                # 合成场景端到端自测
│   ├── benchmark_synthetic.py         # 合成场景精度基准
│   ├── benchmark_tilt.py              # 倾斜视角基准
│   ├── collect_yolo_rgb.py            # YOLO 训练图片交互采集
│   ├── live_l515_yolo_test.py         # 实时 YOLO + RGB-D 测试（旧）
│   ├── live_top_layer_tracker_test.py # 顶层冻结跟踪测试（旧）
│   └── view_stack_map_3d.py           # 交互式 3D 垛堆查看（旧）
├── src/box_perception/
│   ├── camera/                        # L515 标定、ROS RGB-D 读取、RGB 采集
│   ├── segmentation/                  # YOLO-Seg 实例分割
│   ├── temporal/                      # 高度图、时序变化、新箱关联
│   ├── geometry/                      # 点云、顶面/矩形拟合、托盘检测、位姿优化
│   ├── tracking/                      # 多帧稳定（预留，当前未使用）
│   ├── stack/                         # StackMap、支撑关系、可视化
│   ├── evaluation/                    # 精度评估
│   ├── core/                          # 共享数据类型与常量
│   ├── pipeline.py / real_pipeline.py # 合成 / 真实 pipeline 串联
│   └── cli.py                         # 命令行入口
└── tests/                             # 单元测试
```

## 正式版：垛堆箱体建图（`scripts/stack_box_mapper.py`）

单文件独立运行，**不依赖仓库内其它 Python 模块**，面向双箱型四层码垛。

- 箱型（顺序 长×宽×高）：
  - A：`0.40 × 0.30 × 0.30 m`
  - B：`0.42 × 0.27 × 0.21 m`
- 层序：第 1、2 层为 A，第 3、4 层为 B；
- 每层 6 箱：箱子长轴沿托盘短边（`tray +Y`，3D 里的青色轴），短轴沿托盘长边（`tray +X`）；
  因此长边方向排 3 个、短边方向排 2 个；
- 编号按“标准目标区域”划分（与放置先后无关）：俯视托盘，`-Y` 侧一行 1、2、3，
  `+Y` 侧一行 4、5、6。

### 两个模式

- **`update_tray`**：检测空托盘，直接更新托盘参考文件（托盘移动后重跑一次即可）；
- **`map_stack`**：加载已有托盘参考，可在码垛任意时刻打开。

### 运行指令

终端 1 启动固定 L515 驱动：

```bash
cd /home/han/文档/segmentation/stack_seg
./scripts/start_fixed_l515_rgbd.sh
```

终端 2 建图（模式 1 空托盘时先跑一次，模式 2 常规建图）：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/han/文档/segmentation/yp2orin/arm_control/cyclonedds_local.xml

cd /home/han/文档/segmentation/stack_seg

# 模式 1：空托盘时更新托盘参考
/home/han/venvs/stack-live/bin/python scripts/stack_box_mapper.py \
  --mode update_tray \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --tray-reference record/tray_reference/tray_reference.json

# 模式 2：建图（任意时刻打开）
/home/han/venvs/stack-live/bin/python scripts/stack_box_mapper.py \
  --mode map_stack \
  --tray-reference record/tray_reference/tray_reference.json \
  --yolo-weights /home/han/文档/segmentation/yolo_train/runs/fixed_l515_top_box_seg/fixed_l515_top_box_yolo26s_768/weights/best.pt \
  --yolo-device 0 \
  --inference-hz 10 \
  --output record/stack_box_map
```

结果持续写入 `record/stack_box_map/boxmap.json`。可视化：OpenCV 2D 窗口显示活动层箱子；
matplotlib 3D 窗口显示所有箱子（绿=活动层实测、蓝=冻结、橙=标准补全），并保留手动旋转/缩放。
按 `Q` 保存并退出，`S` 立即保存状态。

## 其它脚本速览

| 脚本 | 用途 |
|---|---|
| `check_real_rgbd.py` | 检查一帧完整 RGB-D 链路（同步、吞吐、内参/外参校验） |
| `record_rgbd.py` | 连续录制真实 RGB-D（彩色 PNG + 米制深度 NPY + manifest） |
| `run_real_pipeline.py` | 离线 Before/After 真实数据 pipeline（托盘检测 + 新箱 4DoF） |
| `collect_yolo_rgb.py` | YOLO 标注/微调图片交互采集（按 `K` 存图、`Q` 退出） |
| `run_pipeline.py` / `benchmark_synthetic.py` | 无相机时用合成场景自测与基准 |
| `live_l515_yolo_test.py` | 实时 YOLO + RGB-D 测试，按 `K` 累积 StackMap（旧版） |
| `live_top_layer_tracker_test.py` | 顶层冻结 + 4DoF 跟踪测试（旧版） |
| `view_stack_map_3d.py` | 独立 3D 查看器，监视 stack_map.json（旧版） |

## 运行注意事项

- **Python 环境**：正式版与实时脚本用 `/home/han/venvs/stack-live/bin/python`
  （同时含 `rclpy`、`ultralytics`、`matplotlib`）。离线 pipeline 可用
  `/home/han/miniforge3/envs/box-seg/bin/python`。
- **ROS 环境变量**：运行前必须 `source /opt/ros/humble/setup.bash` 并 export
  `ROS_DOMAIN_ID=0`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`CYCLONEDDS_URI`。
- **不要设置 `PYTHONPATH=src`**：会覆盖 ROS 2 的 Python 路径，导致 `rclpy` 无法导入；
  脚本会自行加载 `src`。
- **L515 驱动**：`./scripts/start_fixed_l515_rgbd.sh` 固定使用专用 librealsense `2.54.2`；
  物理重新插拔相机后需重启该脚本；不要用系统 `realsense-viewer`（2.58.3）启动这台 L515。
- **托盘参考**：`record/tray_reference/tray_reference.json` 需在**空托盘**时用模式 1 生成；
  文件带地图 SHA256，地图不一致时程序拒绝加载。
- **`mpl_toolkits`**：`stack-live` 环境的 `mpl_toolkits` 曾因缺顶层 `__init__.py` 被系统旧版
  namespace 劫持，正式版已在内部规避（`_preload_mplot3d`），无需手动处理。
- **大文件不入库**：模型权重、录包、点云、图像、`record/` 数据均被 `.gitignore` 忽略。

## 开发约定

- **分支模型**：`main` 保持可用；功能开发在 `feature/*` 分支；阶段性可交付打 `release/v*` 标签。
- **提交信息**：遵循 Conventional Commits，如 `feat(geometry): ...`、`fix(camera): ...`、`docs: ...`。
- **合并前**：跑 `python -m pytest`，并确认 `python -m pip install -e ".[dev]"` 无报错。
- **大文件**：需要版本化时使用 Git LFS 或外部对象存储。

## 当前状态

- 正式版 `scripts/stack_box_mapper.py` 已完成（tag `stack-mapper-v1`），支持 A/B 双箱型四层码垛、
  标准目标区域编号、托盘更新与加载双模式、按实测位置冻结、活动层短暂丢失保持、深度兜底与
  3D 可视化，并已在真机上完成连续码垛验证。
- V1 核心几何链路（M2/M3/M5~M8）已用真实 Before/After 数据闭环验证；多帧稳定（M9）与
  V1/V2 正式验收仍在进行。
- 仓库无相机自测：`python scripts/run_pipeline.py`、`python scripts/benchmark_synthetic.py`、
  `python -m pytest`。
