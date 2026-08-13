import numpy as np

from box_perception.geometry.plane_fitting import fit_top_plane, is_horizontal
from box_perception.synthetic import box_top_points


def test_fit_horizontal_top_plane():
    pts = box_top_points(
        center=(0.1, 0.2, 0.35),
        length=0.6,
        width=0.4,
        yaw_deg=10.0,
        spacing=0.03,
        noise=0.001,
        seed=1,
    )
    plane = fit_top_plane(pts, distance_threshold=0.003)
    assert is_horizontal(plane.normal, threshold=0.99)
    assert abs(plane.height - 0.35) < 0.003
    assert plane.plane_rmse < 0.003


def test_robust_to_outliers():
    pts = box_top_points(center=(0.0, 0.0, 0.3), spacing=0.03, noise=0.0, seed=0)
    outliers = np.array([[0.0, 0.0, 0.5], [0.1, 0.1, 0.55], [0.2, 0.2, 0.0]])
    pts = np.vstack([pts, outliers])
    plane = fit_top_plane(pts, distance_threshold=0.005)
    assert is_horizontal(plane.normal, threshold=0.99)
    assert abs(plane.height - 0.3) < 0.005

