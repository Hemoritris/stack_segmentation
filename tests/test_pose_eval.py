import numpy as np

from box_perception.evaluation.pose_eval import (
    summarize,
    xy_error,
    yaw_error_deg,
    z_error,
)


def test_pose_metrics():
    est = np.array([[0.01, 0.0, 0.35], [0.0, -0.02, 0.30]])
    gt = np.array([[0.0, 0.0, 0.35], [0.0, 0.0, 0.30]])
    assert xy_error(est, gt).tolist() == [0.01, 0.02]
    assert z_error(est, gt).tolist() == [0.0, 0.0]


def test_yaw_error_wraps_at_180():
    assert yaw_error_deg(np.array([179.0]), np.array([1.0])).tolist() == [2.0]


def test_summarize():
    err = np.array([0.0, 0.02])
    s = summarize(err)
    assert s["max"] == 0.02
    assert s["median"] == 0.01

