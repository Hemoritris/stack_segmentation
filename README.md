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

## 开发约定

- **分支模型**：`main` 保持可用；功能开发在 `feature/*` 分支；阶段性可交付打 `release/v*` 标签（对应 V1/V2/V3 验收）。
- **提交信息**：建议遵循 Conventional Commits，如 `feat(temporal): add height-map temporal differencing`、`fix(geometry): ...`、`docs: ...`。
- **合并前**：跑 `python -m pytest`，并确认 `python -m pip install -e ".[dev]"` 无报错。
- **大文件**：模型权重、录包、点云、图像数据默认不入库（见 `.gitignore`），需要版本化时使用 Git LFS 或外部对象存储。

## 当前状态

V1 开发中。模块完成度与验收项以 `rgbd_box_4dof_stack_development_plan.md` 中的 checklist 为准。
