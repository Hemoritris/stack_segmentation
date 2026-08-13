import numpy as np

from box_perception.geometry.box_optimizer import BoxOptimizer
from box_perception.geometry.rectangle_init import fit_min_area_rect
from box_perception.synthetic import box_top_points


def test_optimize_recovers_pose():
    length, width = 0.6, 0.4
    center = (0.12, -0.08)
    yaw_deg = 25.0
    pts = box_top_points(
        center=(center[0], center[1], 0.35),
        length=length,
        width=width,
        yaw_deg=yaw_deg,
        spacing=0.02,
        noise=0.0,
        seed=0,
    )
    xy = pts[:, :2]
    cx, cy, _, _, yaw_deg_init = fit_min_area_rect(xy)
    init = np.array([cx, cy, np.deg2rad(yaw_deg_init)])

    x, y, yaw, cost = BoxOptimizer(length, width).optimize(xy, init)
    assert abs(x - center[0]) < 0.002
    assert abs(y - center[1]) < 0.002
    d = abs(np.rad2deg(yaw) - yaw_deg) % 180.0
    assert min(d, 180.0 - d) < 1.0
    assert cost < 1e-6

