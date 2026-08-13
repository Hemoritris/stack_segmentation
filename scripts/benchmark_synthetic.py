#!/usr/bin/env python3
"""在多种合成退化场景下评估 M2~M8 的精度。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from box_perception.pipeline import run_synthetic_demo  # noqa: E402


def _err(result: dict) -> tuple[float, float, float]:
    est = np.asarray(result["est"])
    true = np.asarray(result["true"])
    e_xy = float(np.hypot(est[0] - true[0], est[1] - true[1]))
    e_z = abs(est[2] - true[2])
    d = abs(est[3] - true[3]) % 180.0
    e_yaw = min(d, 180.0 - d)
    return e_xy * 1000.0, e_z * 1000.0, e_yaw


def main() -> int:
    scenarios = [
        ("单箱 无噪声", {}),
        ("单箱 噪声2mm", {"depth_noise": 0.002}),
        ("单箱 噪声5mm", {"depth_noise": 0.005}),
        ("单箱 噪声10mm", {"depth_noise": 0.010}),
        (
            "紧贴邻箱 噪声5mm",
            {
                "box_center": (-0.05, 0.0),
                "yaw_deg": 0.0,
                "depth_noise": 0.005,
                "existing_boxes": [(0.55, 0.0, 0.6, 0.4, 0.35, 0.0)],
            },
        ),
        (
            "两层叠放 噪声5mm",
            {
                "box_center": (0.02, 0.0),
                "box_size": (0.5, 0.35, 0.35),
                "yaw_deg": 5.0,
                "base_z": 0.35,
                "depth_noise": 0.005,
                "existing_boxes": [(0.0, 0.0, 0.6, 0.4, 0.35, 0.0)],
            },
        ),
    ]

    print(f"{'场景':<16} {'xy(mm)':>8} {'z(mm)':>8} {'yaw(deg)':>9}")
    print("-" * 44)
    for name, kwargs in scenarios:
        result = run_synthetic_demo(**kwargs)
        e_xy, e_z, e_yaw = _err(result)
        print(f"{name:<16} {e_xy:>8.2f} {e_z:>8.2f} {e_yaw:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
