import numpy as np

from box_perception.geometry.pointcloud import depth_to_pointcloud
from box_perception.geometry.rectangle_init import fit_min_area_rect


def test_axis_aligned_rectangle():
    xs = [0.0, 0.6, 0.6, 0.0, 0.3, 0.15, 0.45]
    ys = [0.0, 0.0, 0.4, 0.4, 0.2, 0.1, 0.3]
    pts = np.stack([xs, ys], axis=1)
    cx, cy, length, width, yaw = fit_min_area_rect(pts)
    assert abs(cx - 0.3) < 1e-3
    assert abs(cy - 0.2) < 1e-3
    assert abs(length - 0.6) < 1e-2
    assert abs(width - 0.4) < 1e-2
    assert yaw < 1e-3 or abs(yaw - 180.0) < 1e-3


def test_rotated_rectangle_yaw():
    w, h = 0.6, 0.4
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    theta = np.deg2rad(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = corners @ R.T
    cx, cy, length, width, yaw = fit_min_area_rect(pts)
    assert abs(length - w) < 1e-2
    assert abs(width - h) < 1e-2
    assert min(abs(yaw - 30.0), abs(yaw - 120.0)) < 1.0


def test_depth_to_pointcloud():
    depth = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    fx = fy = 100.0
    cx, cy = 0.5, 0.5
    pts = depth_to_pointcloud(depth, fx, fy, cx, cy)
    assert pts.shape == (3, 3)
    assert np.all(pts[:, 2] == 1.0)

