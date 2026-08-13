import numpy as np

from box_perception.temporal.change_detector import detect_change


def test_detect_raised_box():
    hb = np.zeros((20, 20), dtype=float)
    ha = hb.copy()
    ha[4:12, 4:12] = 0.35
    mask = detect_change(hb, ha, grid_size_m=0.05, min_height_diff_m=0.01, min_area_m2=0.005)
    # 内部区域应稳定检出；角落可能被形态学轻微圆化。
    assert mask[5:11, 5:11].all()
    assert not mask[0:3, :].any()
    assert not mask[13:, :].any()


def test_appeared_surface():
    hb = np.full((16, 16), np.nan)
    ha = np.full((16, 16), np.nan)
    ha[3:13, 3:13] = 0.3
    mask = detect_change(hb, ha, grid_size_m=0.05, min_height_diff_m=0.01, min_area_m2=0.001)
    assert mask[6:10, 6:10].all()
    assert not mask[0:2, 0:2].any()

