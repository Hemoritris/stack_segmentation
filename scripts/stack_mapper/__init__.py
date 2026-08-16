"""垛堆箱体建图（stack_box_mapper）的功能模块包。

入口脚本为 scripts/stack_box_mapper.py，本包内的各模块按功能拆分：

- config    任务配置（箱型、层序、容差常量）
- types     数据结构
- geometry  几何运算（反投影、顶面/矩形拟合、中值、投影）
- camera    相机标定、ROS RGB-D 读取、坐标变换
- detect    YOLO 分割、托盘检测、单箱 4DoF 估计、深度兜底
- boxmap    层高/编号/标准位置、垛堆模型更新与保存
- visualize 2D 与 3D 可视化
"""
