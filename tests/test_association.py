import numpy as np
import pytest

from box_perception.core.types import BoxInstance
from box_perception.segmentation.mask_utils import change_overlap, mask_iou
from box_perception.temporal.new_box_association import associate_new_box


def test_mask_iou():
    a = np.zeros((10, 10), dtype=bool)
    a[:5, :5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:8, 3:8] = True
    assert mask_iou(a, b) == pytest.approx(4 / 46)


def test_change_overlap():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    c = np.zeros((10, 10), dtype=bool)
    c[2:6, 2:6] = True
    assert change_overlap(m, c) == pytest.approx(1.0)


def test_associate_returns_intersection_mask():
    # 一个粘连实例包含左右两个箱子，变化区域只覆盖右侧新箱。
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1:5] = True
    mask[1:5, 6:10] = True
    change = np.zeros((10, 10), dtype=bool)
    change[1:5, 6:10] = True

    inst = BoxInstance(mask=mask, bbox=np.array([1, 1, 10, 5]), confidence=0.9)
    obs = associate_new_box([inst], change, min_overlap=0.2)
    assert obs is not None
    np.testing.assert_array_equal(obs.instance_mask, change)
    assert obs.change_overlap == pytest.approx(0.5)

