"""任务配置：箱型、层序与容差常量。"""

# 箱型：长、宽、高（米），顺序与需求一致（长×宽×高）。
BOX_TYPES: dict[str, dict[str, float]] = {
    "A": {"length": 0.40, "width": 0.30, "height": 0.30},
    "B": {"length": 0.42, "width": 0.27, "height": 0.21},
}

# 层号 -> 箱型（第 1、2 层 A，第 3、4 层 B）。
LAYER_BOX_TYPES: list[str | None] = [None, "A", "A", "B", "B"]

# 总层数与每层箱子数。
LAYER_COUNT = 4
BOXES_PER_LAYER = 6

# 尺寸校验容差：箱子存在制造/贴合误差，长宽允许偏离标准值 ±25%。
SIZE_RATIO_MIN = 0.75
SIZE_RATIO_MAX = 1.25
# 短轴（宽度）方向受透视与深度噪声影响、在最底层易测偏小，单独放宽下限。
SIZE_RATIO_WIDTH_MIN = 0.5

# 活动层箱子连续未识别多少帧后才从 boxmap 移除（容忍短暂遮挡，如上层箱子/机械臂挡一下）。
# 10 帧约等于 1 秒（inference-hz=10），覆盖摆放上层箱子期间对下层的短暂遮挡。
MISSED_FRAMES_BEFORE_REMOVE = 10

# 层高判定容差（米）：观测顶面高度与标准层顶面高度的最大偏差。
# 托盘存在约 70mm 的边框/垫板结构，箱子实际放置面高于托盘顶面中心，
# 因此放宽到 100mm；层号本身仍由“最近邻”判定，不受此容差影响。
LAYER_HEIGHT_TOLERANCE = 0.10
