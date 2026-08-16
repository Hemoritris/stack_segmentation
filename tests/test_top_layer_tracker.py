import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/live_top_layer_tracker_test.py"
SPEC = importlib.util.spec_from_file_location("live_top_layer_tracker_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate(layer: int, x: float, yaw: float = 0.0):
    return MODULE.Candidate4DoF(
        accepted=True,
        reasons=[],
        mask=np.ones((4, 4), dtype=bool),
        bbox=np.array([0, 0, 3, 3]),
        yolo_confidence=0.9,
        layer=layer,
        x=x,
        y=0.0,
        z=0.15 + 0.30 * (layer - 1),
        yaw=yaw,
        top_z=0.30 * layer,
        length=0.40,
        width=0.30,
        height=0.30,
        geometry_score=0.9,
    )


def test_layer_height_quantization() -> None:
    layer, error = MODULE.layer_from_top_height(0.907, 0.307, 0.30)
    assert layer == 2
    assert error == pytest.approx(0.0, abs=0.008)


def test_partial_rectangle_is_rejected() -> None:
    reasons = MODULE.geometry_rejection_reasons(
        length=0.24,
        width=0.18,
        expected_length=0.40,
        expected_width=0.30,
        top_area_ratio=0.36,
        rectangle_fill_ratio=0.82,
        top_inlier_ratio=0.50,
        min_size_ratio=0.82,
        max_size_ratio=1.22,
        min_top_area_ratio=0.65,
        min_rectangle_fill=0.60,
        min_top_inlier_ratio=0.15,
    )
    assert any(reason.startswith("L_ratio") for reason in reasons)
    assert any(reason.startswith("W_ratio") for reason in reasons)
    assert any(reason.startswith("top_area") for reason in reasons)


def test_previous_layer_freezes_after_stable_upper_layer() -> None:
    tracker = MODULE.LayerFreezeTracker(
        switch_cycles=2,
        confirm_cycles=2,
        smoothing_alpha=0.5,
    )
    lower = _candidate(1, 1.0)
    tracker.update([lower], 1.0)
    tracker.update([lower], 2.0)
    assert tracker.active_layer == 1
    tracker.update([_candidate(1, 1.02)], 3.0)
    assert tracker.tracks[0].confirmed
    assert tracker.tracks[0].x == pytest.approx(1.01)

    upper = _candidate(2, 1.1, 5.0)
    tracker.update([upper], 4.0)
    assert tracker.active_layer == 1
    tracker.update([upper], 5.0)
    assert tracker.active_layer == 2
    lower_track = next(track for track in tracker.tracks if track.layer == 1)
    assert lower_track.frozen
    frozen_x = lower_track.x

    # Lower-layer observations can no longer alter the frozen pose.
    tracker.update([_candidate(1, 1.25), upper], 6.0)
    assert lower_track.x == frozen_x


# Imported late so the standalone-script loader above stays explicit.
import pytest  # noqa: E402
