#!/usr/bin/env python3
"""在不同相机倾斜角度下评估 M2~M8 的精度退化。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from box_perception.pipeline import run_tilted_demo  # noqa: E402


def _err(result: dict) -> tuple[float, float, float]:
    est = np.asarray(result["est"])
    true = np.asarray(result["true"])
    e_xy = float(np.hypot(est[0] - true[0], est[1] - true[1]))
    e_z = abs(est[2] - true[2])
    d = abs(est[3] - true[3]) % 180.0
    e_yaw = min(d, 180.0 - d)
    return e_xy * 1000.0, e_z * 1000.0, e_yaw


def main() -> int:
    print(f"{'tilt(deg)':<10} {'xy(mm)':>8} {'z(mm)':>8} {'yaw(deg)':>9}")
    print("-" * 42)
    for tilt in [0, 5, 10, 15, 20, 30]:
        try:
            result = run_tilted_demo(tilt_deg=tilt, depth_noise=0.005)
            e_xy, e_z, e_yaw = _err(result)
            print(f"{tilt:<10} {e_xy:>8.2f} {e_z:>8.2f} {e_yaw:>9.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{tilt:<10} 失败: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
