import numpy as np

from box_perception.pipeline import run_synthetic_demo


def test_end_to_end_recovers_pose():
    result = run_synthetic_demo()
    est = np.asarray(result["est"])
    true = np.asarray(result["true"])
    e_xy = float(np.hypot(est[0] - true[0], est[1] - true[1]))
    e_z = abs(est[2] - true[2])
    d = abs(est[3] - true[3]) % 180.0
    e_yaw = min(d, 180.0 - d)
    assert e_xy < 0.008
    assert e_z < 0.008
    assert e_yaw < 1.0

