import numpy as np

from box_perception.pipeline import run_synthetic_demo


def _errors(result: dict) -> tuple[float, float, float]:
    est = np.asarray(result["est"])
    true = np.asarray(result["true"])
    e_xy = float(np.hypot(est[0] - true[0], est[1] - true[1]))
    e_z = abs(est[2] - true[2])
    d = abs(est[3] - true[3]) % 180.0
    e_yaw = min(d, 180.0 - d)
    return e_xy * 1000.0, e_z * 1000.0, e_yaw


def test_end_to_end_recovers_pose():
    result = run_synthetic_demo()
    e_xy, e_z, e_yaw = _errors(result)
    assert e_xy < 0.008
    assert e_z < 0.008
    assert e_yaw < 1.0


def test_tight_packing_with_neighbor():
    result = run_synthetic_demo(
        box_center=(-0.05, 0.0),
        yaw_deg=0.0,
        depth_noise=0.005,
        existing_boxes=[(0.55, 0.0, 0.6, 0.4, 0.35, 0.0)],
    )
    e_xy, e_z, e_yaw = _errors(result)
    assert e_xy < 15.0
    assert e_z < 5.0
    assert e_yaw < 2.0


def test_two_layer_stack():
    result = run_synthetic_demo(
        box_center=(0.02, 0.0),
        box_size=(0.5, 0.35, 0.35),
        yaw_deg=5.0,
        base_z=0.35,
        depth_noise=0.005,
        existing_boxes=[(0.0, 0.0, 0.6, 0.4, 0.35, 0.0)],
    )
    e_xy, e_z, e_yaw = _errors(result)
    assert e_xy < 10.0
    assert e_z < 5.0
    assert e_yaw < 2.0
