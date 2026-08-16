# stack_segmentation

基于固定挂顶 RGB-D 相机（Intel RealSense L515）的箱体 4DoF 位姿估计与垛堆增量建模。

## 简介

面向机器人码垛场景：识别机器人每次新放置的箱子，输出单个箱子的 4DoF 位姿
`[x, y, z, yaw]`，并把历史结果累积成一个垛堆模型（BoxMap）。

核心思路：不重新理解整个垛堆，而是利用“每次只新增一个箱子”的时序先验，只识别刚放置的
新箱子，再把它累积进垛堆模型。

## 箱型与布局

- 箱型（顺序 长×宽×高）：
  - A：`0.40 × 0.30 × 0.30 m`
  - B：`0.42 × 0.27 × 0.21 m`
- 层序：第 1、2 层为 A，第 3、4 层为 B；
- 每层 6 箱：箱子长轴沿托盘短边（`tray +Y`，3D 可视化里的青色轴），短轴沿托盘长边
  （`tray +X`）；因此长边方向排 3 个、短边方向排 2 个；
- 编号按“标准目标区域”划分（与放置先后无关）：俯视托盘，`-Y` 侧一行 1、2、3，
  `+Y` 侧一行 4、5、6。

## 实现思路

```text
L515 RGB-D
  ├─ RGB ────→ YOLO-Seg ────→ 实例 mask
  └─ Depth ───→ 世界点云 ────→ 高于托盘的候选区域
                         ↓
                 深度兜底（深度区域 − YOLO mask）
                         ↓
                 单箱点云 → 顶面拟合 → 矩形拟合 → [x, y, z, yaw]
                         ↓
                 层号判定 / 编号匹配 → BoxMap
```

关键设计：

- **YOLO 只提供候选 mask**：箱体必须通过 RGB-D 几何校验（顶面水平、实测尺寸、顶面面积、
  矩形填充率、层高）才会被确认；只露出一部分或被遮挡的候选会被拒绝。
- **深度兜底**：YOLO 对个别箱子漏检时，用深度图找“高于托盘顶面”的区域，减去 YOLO 已识别的
  mask，对剩余区域做连通域，得到漏检候选，再走同样的几何校验。
- **层冻结**：识别到更高一层的箱子时冻结前一层，位置按实测锁定；中途打开时，看不到的层
  按标准位置补全。
- **短暂丢失保持**：活动层箱子连续 10 帧未识别才移除，容忍上层箱子/机械臂的短暂遮挡。
- **尺寸/层高容差**：短轴（宽度）方向受透视与深度噪声影响、在最底层易测偏小，单独放宽；
  层高容差放宽以适配托盘约 70mm 的边框/垫板结构。

## 代码结构

```text
.
├── README.md
├── LICENSE                            # MIT 许可
├── requirements.txt                   # Python 依赖
├── config/                            # 相机、外参、驱动与 DDS 配置
│   ├── camera.yaml
│   ├── fixed_l515_world_extrinsics_filtered.json
│   ├── fixed_l515_rgbd.yaml
│   └── cyclonedds_local.xml
├── models/
│   └── best.pt                        # YOLO-Seg 权重（已随仓库分发）
└── scripts/
    ├── stack_box_mapper.py            # 入口脚本（参数解析 + 两个模式主循环）
    ├── stack_mapper/                  # 功能模块包
    │   ├── config.py                  # 任务配置（箱型、层序、容差常量）
    │   ├── types.py                   # 数据结构
    │   ├── geometry.py                # 几何（反投影、顶面/矩形拟合、中值、投影）
    │   ├── camera.py                  # 相机标定、ROS RGB-D 读取、坐标变换
    │   ├── detect.py                  # 检测（YOLO、托盘、单箱 4DoF、深度兜底）
    │   ├── boxmap.py                  # 垛堆模型（层高/编号、更新与保存）
    │   └── visualize.py               # 2D 与 3D 可视化
    └── start_fixed_l515_rgbd.sh       # 启动固定 L515 驱动
```

## 快速配置（首次部署）

1. **系统依赖**：ROS 2 Humble（含 `rclpy`）、Python 3.10+、L515 专用运行时
   （librealsense 2.54.2 + RealSense ROS 4.54.1，安装目录通过
   `scripts/start_fixed_l515_rgbd.sh` 顶部的 `L515_RUNTIME_DIR` 指定）。

2. **Python 环境**：需要一个**同时含 `rclpy`、`ultralytics`、`matplotlib`** 的解释器
   （`rclpy` 随 ROS 2 安装，`ultralytics`/`matplotlib` 用 pip 装）。推荐用
   `--system-site-packages` 建 venv 以继承 ROS 2 的 `rclpy`：

   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **相机标定**：内参、外参已随仓库分发。若更换相机或移动相机，需重新标定，
   并更新 `config/camera.yaml`（相机 serial、`expected_map_sha256`）。

4. **托盘参考**：首次运行前，在空托盘状态下执行模式 1，生成
   `record/tray_reference/tray_reference.json`。

## 运行指令

终端 1 启动固定 L515 驱动（脚本内部已配置好 ROS / CycloneDDS 环境变量）：

```bash
./scripts/start_fixed_l515_rgbd.sh
```

终端 2 建图（先 `source` ROS 2 并激活上面的 venv）：

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
cd stack_segmentation

# 模式 1：空托盘时更新托盘参考
python scripts/stack_box_mapper.py \
  --mode update_tray \
  --yolo-weights models/best.pt \
  --tray-reference record/tray_reference/tray_reference.json

# 模式 2：建图（任意时刻打开）
python scripts/stack_box_mapper.py \
  --mode map_stack \
  --tray-reference record/tray_reference/tray_reference.json \
  --yolo-weights models/best.pt \
  --yolo-device 0 \
  --inference-hz 20 \
  --output record/stack_box_map
```

结果持续写入 `record/stack_box_map/boxmap.json`。

可视化：OpenCV 2D 窗口显示活动层箱子；matplotlib 3D 窗口显示所有箱子
（绿=活动层实测、蓝=冻结、橙=标准补全），并保留手动旋转/缩放。按 `Q` 保存并退出，
`S` 立即保存状态。

## 运行注意事项

- **ROS 环境**：运行前先 `source` ROS 2 的 `setup.bash`；`start_fixed_l515_rgbd.sh` 会自动
  export `ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、`CYCLONEDDS_URI`（指向仓库内
  `config/cyclonedds_local.xml`）。
- **不要设置 `PYTHONPATH`**：会覆盖 ROS 2 的 Python 路径，导致 `rclpy` 无法导入。
- **L515 驱动**：专用脚本固定使用 librealsense 2.54.2，物理重插相机后需重启；不要用系统
  `realsense-viewer`（2.58.x）启动 L515。
- **托盘参考**：需空托盘时生成；文件带地图 SHA256，地图不一致时拒绝加载。
- **`mpl_toolkits`**：某些环境 `mpl_toolkits` 可能被系统旧版 namespace 劫持，程序已在内部
  规避，无需手动处理。
- **YOLO 权重**：已随仓库分发于 `models/best.pt`；换模型时用 `--yolo-weights` 覆盖。
- **运行数据不入库**：`record/` 下的点云、图像、结果均被 `.gitignore` 忽略。

## 许可

本项目采用 [MIT License](./LICENSE)。
