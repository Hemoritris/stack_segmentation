# 基于 RGB-D 实例分割与时序深度的箱体 4DoF 位姿估计与垛堆建模方案

## 1. 项目目标

本项目面向机器人码垛场景，使用固定挂顶 RGB-D 相机对机器人新放置的纸箱进行识别，并输出单个箱子的 4DoF 位姿：

\[
[x,\ y,\ z,\ yaw]
\]

其中：

- `x, y`：箱子在机器人世界坐标系下的平面位置；
- `z`：箱子几何中心或顶面中心高度；
- `yaw`：箱子绕世界坐标系 Z 轴的旋转角；
- 默认箱体在放置完成后近似水平，因此不重点估计 roll / pitch。

整体开发分为三个版本：

- **V1：固定单一尺寸箱子的新增箱体识别与 4DoF 位姿估计**
- **V2：在 V1 基础上增加垛堆增量式可视化建模**
- **V3：扩展到多尺寸箱子，并根据尺寸先验进行 SKU / 箱型匹配和约束拟合**

核心思想是：

> 不重新理解整个垛堆，而是利用机器人“每次只新增一个箱子”的时序先验，识别刚刚放置的新箱子，再把历史结果累积到 StackMap 中。

---

# 2. 总体技术路线

整体数据流：

```text
L515 RGB-D
     │
     ├──────────────────────────┐
     ↓                          ↓
 RGB图像                     Depth
     ↓                          ↓
YOLO-Seg                  深度预处理
     ↓                          ↓
Instance Masks            World Height Map
     │                          │
     │                    t-1 / t 时序差分
     │                          ↓
     │                     Change Mask
     │                          │
     └────────────┬─────────────┘
                  ↓
             新箱关联模块
                  ↓
             New Box Mask
                  ↓
            单箱 RGB-D ROI
                  ↓
             单箱3D点云
                  ↓
        顶面提取 + 异常点过滤
                  ↓
             XY / BEV投影
                  ↓
          几何矩形初始估计
                  ↓
       箱子尺寸先验约束优化
                  ↓
          x, y, z, yaw
                  ↓
           多帧稳定性判断
                  ↓
             Box State
```

最终系统围绕一个稳定的“单箱 4DoF 感知核心”进行扩展：

\[
V1\ 单箱4DoF
\rightarrow
V2\ 垛堆增量建模
\rightarrow
V3\ 多尺寸识别
\]


## 开发进度 Checklist

> 说明：完成模块后将 `- [ ]` 改为 `- [x]`。  
> 当前仅将已经明确具备可用结果的前置项勾选；算法开发模块默认保留为待完成。

### 基础阶段

- [x] **P0 固定 L515 内外参与 World 外参标定**
- [ ] **M0 RGB-D 读取、对齐、点云生成与 World 坐标数据链路**（代码与离线测试完成，待真机验收）
- [ ] **M1 YOLO-Seg 挂顶多箱场景验证**

### V1：固定尺寸新箱 4DoF

- [ ] **M2 World Height Map**
- [ ] **M3 Before / After 深度时序变化检测**
- [ ] **M4 Change Mask × YOLO 实例关联，识别新增箱子**
- [ ] **M5 新箱 Mask + Depth → 单箱点云**
- [ ] **M6 RANSAC 顶面提取**
- [ ] **M7 minAreaRect 4DoF Baseline**
- [ ] **M8 固定尺寸先验约束拟合**
- [ ] **M9 多帧稳定性判断与 Confidence**
- [ ] **V1 验收：固定尺寸新箱 4DoF 精度达标**

### V2：增量垛堆建模

- [ ] **M10 StackMap**
- [ ] **M11 箱子 ID、添加与位姿更新机制**
- [ ] **M12 3D 垛堆可视化**
- [ ] **M13 箱体支撑关系 / Support Graph**
- [ ] **V2 验收：连续码垛过程中模型能够正确增量更新**

### V3：多尺寸箱体

- [ ] **M14 箱体自由尺寸粗估**
- [ ] **M15 SKU / 箱型尺寸匹配**
- [ ] **M16 多尺寸 YOLO-Seg 数据补充与微调**
- [ ] **M17 多尺寸先验约束拟合**
- [ ] **V3 验收：多尺寸箱体 4DoF 与尺寸识别达到目标精度**


---


# 3. 基础坐标系与数据链路

## 3.1 固定相机外参

> **完成提示：** [x] 固定 L515 到 World 的标定结果已具备，可继续作为后续开发基础。

固定 L515 已完成到机器人世界坐标系的外参标定：

\[
{}^{W}T_C
\]

相机点：

\[
P_C
\]

转换到世界坐标：

\[
P_W={}^WT_CP_C
\]

建议统一世界坐标定义：

```text
World:
X = 托盘长方向
Y = 托盘宽方向
Z = 向上
yaw = 绕 +Z
```

所有后续点云和箱体几何运算统一在 `world` 坐标系下进行。

---

## 3.2 RGB-D 基础模块

当前实现采用 ROS 2 `realsense2_camera` 输出，而不是在 Python 中再次打开 USB：

```text
/fixed_l515/color/image_raw
/fixed_l515/color/camera_info
/fixed_l515/aligned_depth_to_color/image_raw
```

固定 L515 原生深度为 `1024x768`，驱动将其对齐到 `1280x720` 彩色图。对齐后的深度必须使用
彩色相机 K/D 反投影，不能使用原生深度 K。实时 CameraInfo 会与厂家参数逐项比较；世界外参
从以下冻结结果加载并校验地图哈希：

```text
../two_camera/data/results/fixed_l515_capture_20260814_map2/
  fixed_l515_world_extrinsics_filtered.json
```

转换链为：

```text
depth pixel + color K/D
  -> P_fixed_l515_color_optical_frame
  -> slamware_map_T_fixed_l515_color_optical_frame
  -> P_slamware_map
```

实现位置：

- `camera/calibration.py`：厂家内参、过滤版世界外参、地图/Frame 校验；
- `camera/ros_rgbd.py`：ROS 彩色/对齐深度时间配对和米制深度解码；
- `geometry/pointcloud.py`：带畸变处理的对齐深度反投影；
- `scripts/check_real_rgbd.py`：一帧真机闭环检查；
- `scripts/record_rgbd.py`：真实 Before/After 数据录制。

建议首先完成以下模块：

| 模块 | 输入 | 输出 |
|---|---|---|
| camera_driver | L515 | RGB、Depth、timestamp |
| depth_filter | Raw Depth | Filtered Depth |
| rgb_depth_align | RGB + Depth | 对齐后的 Depth |
| pointcloud_generator | Depth + Intrinsics | 相机坐标点云 |
| world_transform | 点云 + 外参 | World 点云 |
| roi_filter | World 点云 | 托盘 / 垛堆 ROI |

基础阶段重点验证：

- RGB 与 Depth 对齐；
- 世界坐标转换正确；
- 托盘平面不明显倾斜；
- 静态目标连续采样稳定；
- ROI 裁剪正确；
- 相机飞点和空洞在可接受范围内。

---

# 4. V1：固定尺寸单箱 4DoF

V1 是整个项目最关键的一版。

目标：

> 在每次机器人完成放箱后，从 RGB-D 数据中识别“刚刚新增的箱子”，并输出稳定的 \(x,y,z,yaw\)。

V1 暂时假设箱子尺寸固定：

\[
L=L_0,\quad W=W_0,\quad H=H_0
\]

---

# 5. V1-A：YOLO-Seg 实例分割

## 5.1 输入输出

输入：

```text
RGB Image
```

输出：

```text
BoxInstance[]
    mask
    bbox
    confidence
```

YOLO-Seg 的主要任务不是直接给出精确几何，而是：

> 将不同纸箱实例分开，并给出单个箱子的 RGB mask。

---

## 5.2 现有模型验证

目前模型主要使用单箱多面视角训练，因此第一步应直接在挂顶实际场景测试：

- 单箱；
- 两箱分离；
- 两箱紧贴；
- 多箱紧贴；
- 第一层基本铺满；
- 第二层刚放入一个新箱；
- 相邻箱子顶面共面；
- 少量机械臂遮挡；
- 不同光照和阴影。

重点统计：

- 漏检；
- 多箱粘连成一个实例；
- 一个箱子被拆成多个实例；
- mask 边界明显错误；
- 错误识别背景。

对于本项目：

> 实例是否正确分离，比单纯追求非常高的 mask IoU 更重要。

---

## 5.3 推荐补充数据

若现有模型在垛堆场景中泛化不足，第一阶段建议增加：

\[
300\sim500
\]

张真实挂顶 RGB 图像。

建议覆盖：

- 多箱同时出现；
- 同高度紧贴；
- 不同 yaw；
- 不同垛高；
- 不同光照；
- 少量机械臂遮挡；
- 空托盘和背景负样本。

建议每张图包含多个实例，使总实例数量达到约：

\[
2000\sim3000
\]

个。

训练集、验证集、测试集应按照完整码垛实验划分，不要随机拆连续视频帧。

---

# 6. V1-B：Depth 时序变化检测

YOLO 负责“箱子是谁”，Depth 时序负责“哪个箱子是刚刚新增的”。

---

## 6.1 Before / After

机器人放置前保存稳定深度状态：

\[
D_{before}
\]

机器人放置完成并撤离后保存：

\[
D_{after}
\]

不建议直接进行原始深度图差分。

更推荐转换成世界坐标下的高度图。

---

## 6.2 World Height Map

将点云投影到世界坐标 XY 网格：

\[
H(x,y)=z
\]

推荐第一版 XY 栅格尺寸：

\[
5\sim10\text{ mm/cell}
\]

对每个栅格的 Z 使用：

- median；
- percentile；
- 或经过离群点过滤后的稳定值。

得到：

\[
H_{before}(x,y)
\]

和：

\[
H_{after}(x,y)
\]

计算：

\[
\Delta H(x,y)
=
H_{after}(x,y)-H_{before}(x,y)
\]

经过：

- 高度阈值；
- 面积阈值；
- 形态学处理；
- Connected Components；

得到：

```text
Change Mask
```

代表当前场景中新增加的三维区域。

---

# 7. V1-C：YOLO 与时序 Depth 融合

设 YOLO 得到多个实例：

\[
M_1,M_2,\dots,M_n
\]

时序变化区域为：

\[
C
\]

可以计算每个实例与变化区域的重叠：

\[
Score_i=
\frac{|M_i\cap C|}
{|M_i|}
\]

选择：

\[
i^*=\arg\max_i Score_i
\]

对应：

\[
M_{new}=M_{i^*}
\]

作为当前刚放置的新箱子。

建议输出数据结构：

```cpp
struct NewBoxObservation {
    cv::Mat instance_mask;
    cv::Rect roi;

    float yolo_confidence;
    float change_overlap;

    double timestamp;
};
```

这一层的设计非常重要，因为后续 V2、V3 都可以继续复用。

---

# 8. V1-D：新箱 Mask + Depth 转单箱点云

只提取：

\[
(u,v)\in M_{new}
\]

对应的深度数据。

由相机内参：

\[
X=(u-c_x)Z/f_x
\]

\[
Y=(v-c_y)Z/f_y
\]

得到相机坐标点云：

\[
P_C
\]

再通过：

\[
P_W={}^WT_CP_C
\]

得到：

\[
P_{box}^{world}
\]

建议后续所有几何运算都使用世界坐标点云。

建议接口：

```cpp
struct BoxPointCloud {
    std::vector<Eigen::Vector3d> points;
};
```

---

# 9. V1-E：箱子顶面提取

由于固定 L515 是挂顶相机，主要观测箱子顶面。

第一版不建议直接做复杂完整 6D Cuboid Pose，而是先提取顶面。

处理流程：

```text
Box Point Cloud
      ↓
无效深度过滤
      ↓
离群点去除
      ↓
高区域筛选
      ↓
RANSAC Plane
      ↓
法向量检查
      ↓
Top Surface
```

顶面法向量应近似：

\[
n\approx[0,0,1]
\]

可以设置：

\[
n\cdot[0,0,1]>\tau
\]

过滤明显不是水平顶面的错误平面。

建议输出：

```cpp
struct BoxTopPlane {
    Eigen::Vector3d normal;

    double height;

    std::vector<Eigen::Vector3d> points;

    double plane_rmse;
};
```

顶面高度：

\[
z_{top}
\]

可以通过：

- 顶面点中位数；
- RANSAC 平面；
- 局部稳健平面拟合；

获得。

---

# 10. V1-F：4DoF Baseline

先实现一个简单但完整的 baseline。

将顶面点：

\[
(x,y,z)
\]

投影到 XY：

\[
(x,y,z)\rightarrow(x,y)
\]

流程：

```text
Top Points
    ↓
XY Projection
    ↓
Convex Hull
    ↓
minAreaRect
    ↓
cx, cy, width, height, angle
```

得到：

\[
\hat x,\hat y,\hat L,\hat W,\hat\psi
\]

V1 中箱高已知，因此箱体中心 Z：

\[
z=z_{top}-\frac{H_0}{2}
\]

最终 baseline：

\[
(\hat x,\hat y,\hat z,\hat\psi)
\]

该结果主要作为：

- 几何拟合初始值；
- 后续约束优化的 baseline；
- Debug 对照。

---

# 11. V1-G：固定尺寸先验约束拟合

V1 真正的核心不是直接使用 `minAreaRect` 输出，而是使用实际箱体尺寸：

\[
L_0,W_0,H_0
\]

作为先验。

固定：

\[
L=L_0
\]

\[
W=W_0
\]

只优化：

\[
x,y,\psi
\]

可以构造：

\[
\min_{x,y,\psi}
E_{point}
+
\lambda E_{edge}
+
\mu E_{mask}
\]

其中：

- \(E_{point}\)：顶面点到旋转矩形的几何误差；
- \(E_{edge}\)：拟合矩形边缘与观测边缘的一致性；
- \(E_{mask}\)：矩形投影与 YOLO mask 的一致性。

第一版可以先只实现：

\[
E_{point}
\]

形成一个可工作的几何优化器，再逐步增加其他约束。

优化变量只有：

\[
[x,y,\psi]
\]

因此计算规模很小。

最终：

\[
z=z_{top}-H_0/2
\]

获得：

\[
\boxed{x,y,z,yaw}
\]

---

# 12. V1-H：多帧稳定性判断

机器人刚释放箱子时可能存在：

- 箱体轻微晃动；
- 机械臂未完全退出；
- 深度短时波动；
- 图像瞬时遮挡。

因此不建议单帧直接输出最终结果。

推荐流程：

```text
Robot Release
      ↓
Robot Retreat
      ↓
等待场景稳定
      ↓
连续采集 5~10 帧
      ↓
每帧估计 4DoF
      ↓
统计均值 / 中位数 / 方差
      ↓
Confirmed
```

对：

\[
x_i,y_i,z_i,\psi_i
\]

计算：

\[
\sigma_x,\sigma_y,\sigma_z,\sigma_\psi
\]

满足阈值后：

```text
candidate → confirmed
```

建议最终输出接口：

```cpp
struct BoxEstimate {
    int id;

    double x;
    double y;
    double z;
    double yaw;

    double length;
    double width;
    double height;

    double position_std;
    double yaw_std;

    double plane_rmse;
    double fit_error;

    double yolo_confidence;
    double change_overlap;

    bool valid;
};
```

建议从 V1 就固定这类接口，使 V2 / V3 直接复用。

---

# 13. V1 验收方案

建议建立标准测试集，对箱子设置 50～100 个测试位姿。

覆盖：

| 变量 | 测试范围 |
|---|---|
| X | 托盘左、中、右 |
| Y | 前、中、后 |
| Z | 不同堆高 |
| yaw | 例如 -10° ~ +10° |
| 邻箱 | 单独、贴合 |
| 光照 | 正常、偏亮、偏暗 |
| 遮挡 | 无、少量 |
| 位置 | 图像中心和边缘 |

统计：

\[
e_{xy}
=
\sqrt{(x-x_{gt})^2+(y-y_{gt})^2}
\]

\[
e_z
=
|z-z_{gt}|
\]

\[
e_{yaw}
=
|\psi-\psi_{gt}|
\]

建议报告：

- RMS；
- Median；
- P95；
- Max。

---

## 13.1 V1 推荐验收指标

| 指标 | 第一目标 | 理想目标 |
|---|---:|---:|
| XY RMS | < 10 mm | < 5 mm |
| Z RMS | < 10 mm | < 5 mm |
| Yaw RMS | < 1° | < 0.5° |
| 新箱识别率 | >95% | >99% |
| P95 XY | <15 mm | <8 mm |

如果 V1 的几何精度尚不稳定，不建议过早进入 V2。

---

# 14. V2：增量式垛堆建模

V2 原则：

> 不修改 V1 的单箱感知核心，只增加历史状态维护与可视化。

每次 V1 输出：

```text
BoxEstimate
```

后：

```text
BoxEstimate
      ↓
BoxManager
      ↓
StackMap
      ↓
Visualization
```

---

# 15. StackMap 数据结构

建议使用 Box-based Map，而不是 Layer-based Map。

例如：

```cpp
struct BoxState {
    int id;

    double x;
    double y;
    double z;
    double yaw;

    double length;
    double width;
    double height;

    double confidence;

    bool confirmed;
};

struct StackMap {
    std::vector<BoxState> boxes;
};
```

一次放置：

```text
Before
  ↓
Place
  ↓
After
  ↓
V1 Detect
  ↓
BoxEstimate
  ↓
New ID
  ↓
加入 StackMap
```

如果机器人随后对箱子进行微调：

```text
Box007 old pose
      ↓
机器人推动
      ↓
重新识别
      ↓
Box007 new pose
      ↓
Update StackMap
```

因此 StackMap 始终保存箱子的最终状态。

---

# 16. V2 可视化

不建议长期保存完整历史点云作为主要地图。

建议保存参数化箱子：

\[
B_i=(x,y,z,yaw,L,W,H)
\]

然后按需要实时构造 Cuboid。

可视化可使用：

- RViz Marker；
- Open3D；
- Web 3D。

如果系统本身基于 ROS，优先推荐：

```text
RViz Marker
```

优势：

- 开发成本低；
- 容易与机器人 TF 集成；
- 容易显示箱体 ID；
- 容易显示目标位姿和实际位姿。

---

# 17. V2 支撑关系

V2 后期可增加箱子之间的支撑关系。

例如：

```text
Box007
supported_by:
    Box003
    Box004
```

基本思路：

1. 计算当前箱子的底面；
2. 找高度接近其底面的历史箱子顶面；
3. 计算 XY 投影 overlap；
4. 超过阈值则认为存在支撑关系。

形成：

```text
Box 7
 ↙   ↘
Box3 Box4
 ↓    ↓
Box1 Box2
```

未来可用于：

- 垛堆稳定性判断；
- 悬空比例判断；
- 支撑面积判断；
- 微调可行性判断；
- 强化学习 observation。

---

# 18. V3：多尺寸箱体

V3 的目标是扩展：

\[
L,W,H=\text{固定}
\]

为：

\[
L,W,H=\text{多种}
\]

V1 的以下模块保持不变：

```text
YOLO-Seg
Depth Temporal Difference
New Box Association
Point Cloud
Top Plane
```

主要替换固定尺寸拟合模块。

---

# 19. V3 尺寸识别流程

推荐：

```text
Single Box Point Cloud
        ↓
自由尺寸粗估
        ↓
L_hat, W_hat, H_hat
        ↓
SKU / 箱型库匹配
        ↓
获得先验 L, W, H
        ↓
调用 V1 固定尺寸约束拟合
        ↓
精确 x, y, z, yaw
```

例如：

```yaml
box_types:
  A:
    length: 0.600
    width: 0.400
    height: 0.350

  B:
    length: 0.500
    width: 0.400
    height: 0.300

  C:
    length: 0.400
    width: 0.300
    height: 0.250
```

先估计：

\[
\hat L,\hat W,\hat H
\]

然后：

\[
SKU^*
=
\arg\min_k
D((\hat L,\hat W,\hat H),S_k)
\]

获得对应的真实尺寸：

\[
L_k,W_k,H_k
\]

最后重新调用 V1 的固定尺寸优化器。

这样 V3 不需要重新设计整个 4DoF 算法。

---

# 20. 为什么不依赖固定层高和层数

长期系统不应该假设：

\[
z_k=z_0+kH
\]

因为 V3 中：

- 箱子高度可能不同；
- 每层高度可能不同；
- 不一定存在严格整齐的“层”；
- 层数也可能不固定。

因此建议系统内部统一使用：

```text
Box-based 3D Map
```

而不是：

```text
Layer-based Map
```

“第几层”最多作为可视化辅助信息，不作为核心状态。

---

# 21. 建议开发顺序

下面这部分可直接作为开发过程中的主进度清单：

- [ ] **M0** RGB-D 读取、RGB/Depth 对齐、点云与 World 坐标
- [ ] **M1** YOLO-Seg 实际挂顶多箱场景验证
- [ ] **M2** World Height Map
- [ ] **M3** Before / After 变化检测
- [ ] **M4** Change Mask × YOLO → 新箱关联
- [ ] **M5** 新箱 Mask → 单箱点云
- [ ] **M6** RANSAC 顶面
- [ ] **M7** minAreaRect baseline
- [ ] **M8** 固定尺寸先验约束拟合
- [ ] **M9** 多帧稳定 + Confidence
- [ ] **V1 验收**

---

- [ ] **M10** StackMap
- [ ] **M11** 箱子 ID / 添加 / 更新机制
- [ ] **M12** 3D 可视化
- [ ] **M13** 支撑关系
- [ ] **V2 验收**

---

- [ ] **M14** 自由尺寸粗估
- [ ] **M15** SKU 匹配
- [ ] **M16** 多尺寸 YOLO 数据补充
- [ ] **M17** 多尺寸约束拟合
- [ ] **V3 验收**

其中 V1 最关键的是：

\[
\boxed{M3,\ M4,\ M6,\ M8}
\]

即：

```text
时序新箱检测
    ↓
RGB / Depth 实例关联
    ↓
顶面提取
    ↓
尺寸先验几何拟合
```

# 22. 推荐代码结构

```text
box_perception/
│
├── camera/
│   ├── l515_driver.py
│   ├── calibration.py
│   └── rgbd_align.py
│
├── segmentation/
│   ├── yolo_segmentor.py
│   └── mask_utils.py
│
├── temporal/
│   ├── height_map.py
│   ├── change_detector.py
│   └── new_box_association.py
│
├── geometry/
│   ├── pointcloud.py
│   ├── plane_fitting.py
│   ├── rectangle_init.py
│   └── box_optimizer.py
│
├── tracking/
│   ├── temporal_filter.py
│   └── box_state.py
│
├── stack/
│   ├── stack_map.py
│   ├── support_graph.py
│   └── visualization.py
│
├── config/
│   ├── camera.yaml
│   ├── workspace.yaml
│   └── box_types.yaml
│
└── evaluation/
    ├── pose_eval.py
    ├── perception_eval.py
    └── benchmark.py
```

---

# 23. 各版本的核心变化

| 模块 | V1 | V2 | V3 |
|---|---|---|---|
| RGB-D | ✓ | ✓ | ✓ |
| YOLO-Seg | ✓ | ✓ | ✓，补充多尺寸数据 |
| Depth 时序检测 | ✓ | ✓ | ✓ |
| 新箱识别 | ✓ | ✓ | ✓ |
| 单箱点云 | ✓ | ✓ | ✓ |
| 顶面提取 | ✓ | ✓ | ✓ |
| 4DoF | ✓ | ✓ | ✓ |
| 尺寸 | 单一固定 | 单一固定 | 多尺寸 |
| 几何拟合 | 固定尺寸 | 固定尺寸 | SKU / 多尺寸约束 |
| StackMap | — | ✓ | ✓ |
| 3D 可视化 | — | ✓ | ✓ |
| 支撑关系 | — | 可选 | ✓ |
| 固定层高 | 不依赖 | 不依赖 | 不依赖 |
| 固定层数 | 不依赖 | 不依赖 | 不依赖 |

---

# 24. 最终推荐的开发策略

整个项目不要一开始同时解决：

```text
多尺寸
+
完整垛堆建模
+
稳定性判断
+
多箱6D位姿
+
微调决策
```

第一阶段最重要的是证明：

\[
\boxed{
\text{固定尺寸新箱}
\rightarrow
\text{稳定可靠的 }x,y,z,yaw
}
\]

V1 成功后：

V2 只是：

\[
Box_t\rightarrow StackMap
\]

V3 只是把：

\[
\text{固定尺寸}
\]

改成：

\[
\text{尺寸粗估}
\rightarrow
\text{SKU匹配}
\rightarrow
\text{固定尺寸约束拟合}
\]

因此建议优先把 V1 的以下链路做到足够稳定：

```text
Depth时序变化
      ↓
YOLO实例关联
      ↓
单箱点云
      ↓
顶面提取
      ↓
固定尺寸几何拟合
      ↓
多帧稳定
      ↓
4DoF
```

只要这一核心感知链路稳定，后续垛堆可视化建模和多尺寸扩展都可以较自然地叠加，而不需要推翻已有系统。
