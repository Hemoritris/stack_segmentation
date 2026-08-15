import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from box_perception.camera.calibration import (
    cam_to_world,
    load_intrinsics,
    load_world_calibration,
    validate_live_intrinsics,
)


def _write_config(tmp_path: Path):
    result = {
        "calibration_scope": "fixed_l515_world_extrinsics_only",
        "frames": {"M": "slamware_map", "Cf": "fixed_optical"},
        "dataset": {"map_sha256": "abc123"},
        "transforms": {
            "slamware_map_T_fixed": {
                "matrix": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    config = {
        "intrinsics": {
            "color": {
                "width": 4,
                "height": 3,
                "frame_id": "fixed_optical",
                "distortion_model": "plumb_bob",
                "k": [[100.0, 0.0, 1.5], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]],
                "distortion": [0.1, -0.2, 0.0, 0.0, 0.1],
            }
        },
        "extrinsics": {
            "result_json": "result.json",
            "transform_key": "slamware_map_T_fixed",
            "world_frame": "slamware_map",
            "camera_frame": "fixed_optical",
            "expected_map_sha256": "abc123",
        },
    }
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_frozen_intrinsics_and_world_extrinsics(tmp_path):
    path = _write_config(tmp_path)
    intrinsics = load_intrinsics(path)
    calibration = load_world_calibration(path)
    assert intrinsics.width == 4
    assert intrinsics.fx == 100.0
    assert calibration.map_sha256 == "abc123"
    point = cam_to_world(np.array([[0.0, 0.0, 1.0]]), calibration.world_T_camera)
    np.testing.assert_allclose(point, [[1.0, 2.0, 4.0]])


def test_live_intrinsics_reject_wrong_frame(tmp_path):
    configured = load_intrinsics(_write_config(tmp_path))
    with pytest.raises(ValueError, match="frame"):
        validate_live_intrinsics(
            configured,
            width=4,
            height=3,
            k=configured.k,
            distortion=configured.distortion,
            frame_id="wrong_frame",
        )
