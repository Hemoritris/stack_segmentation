import cv2
import numpy as np

from box_perception.real_pipeline import (
    estimate_box_from_world_clouds,
    normalize_box_yaw_deg,
)


def _synthetic_floor_and_box(yaw_deg=25.0):
    xs = np.arange(-0.7, 0.705, 0.005)
    ys = np.arange(-0.6, 0.605, 0.005)
    xx, yy = np.meshgrid(xs, ys)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.30)])
    center = np.array([0.10, -0.05])
    theta = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = (floor[:, :2] - center) @ rotation
    inside = (np.abs(local[:, 0]) <= 0.20) & (np.abs(local[:, 1]) <= 0.15)
    after = floor.copy()
    after[inside, 2] = 0.60
    return floor, after


def test_real_world_cloud_estimator_recovers_known_box():
    before, after = _synthetic_floor_and_box()
    result, artifacts = estimate_box_from_world_clouds(
        before,
        after,
        box_size=(0.40, 0.30, 0.30),
        roi=((-0.35, 0.55), (-0.50, 0.40)),
    )
    pose = result["pose_4dof"]
    measured = result["measured_size_m"]
    assert abs(pose["x_m"] - 0.10) < 0.01
    assert abs(pose["y_m"] + 0.05) < 0.01
    assert abs(pose["z_m"] - 0.45) < 0.01
    assert abs(normalize_box_yaw_deg(pose["yaw_deg"] - 25.0)) < 1.0
    assert abs(measured["length"] - 0.40) < 0.015
    assert abs(measured["width"] - 0.30) < 0.015
    assert artifacts.world_change_mask.any()


def test_box_yaw_normalization_has_180_degree_symmetry():
    assert normalize_box_yaw_deg(147.5) == -32.5
    assert normalize_box_yaw_deg(-32.5) == -32.5
    assert normalize_box_yaw_deg(327.5) == -32.5

