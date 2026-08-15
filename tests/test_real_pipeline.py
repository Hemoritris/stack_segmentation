import cv2
import numpy as np

from box_perception.real_pipeline import (
    estimate_box_from_world_clouds,
    normalize_box_yaw_deg,
    pose_4dof_reference_to_world,
    pose_4dof_world_to_reference,
    project_world_points_to_image,
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


def test_world_point_projection_uses_world_t_camera_convention():
    k = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    pixels = project_world_points_to_image(
        np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 1.0], [0.0, 0.0, -1.0]]),
        np.eye(4),
        k,
    )
    np.testing.assert_allclose(pixels[:2], [[320.0, 240.0], [330.0, 220.0]])
    assert np.isnan(pixels[2]).all()


def test_reference_pose_transform_round_trip():
    tray_world = {"x_m": 1.2, "y_m": -0.7, "z_m": 0.3, "yaw_deg": 30.0}
    box_in_tray = {"x_m": 0.2, "y_m": -0.1, "z_m": 0.15, "yaw_deg": 20.0}
    box_world = pose_4dof_reference_to_world(box_in_tray, tray_world)
    recovered = pose_4dof_world_to_reference(box_world, tray_world)
    np.testing.assert_allclose(
        [recovered[key] for key in ("x_m", "y_m", "z_m", "yaw_deg")],
        [box_in_tray[key] for key in ("x_m", "y_m", "z_m", "yaw_deg")],
        atol=1e-9,
    )
