#!/usr/bin/env python3
"""端到端合成 pipeline 演示：M2~M8 串联。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from box_perception.pipeline import run_synthetic_demo  # noqa: E402


def main() -> int:
    result = run_synthetic_demo()
    est = result["est"]
    true = result["true"]
    e_xy = float(np.hypot(est[0] - true[0], est[1] - true[1]))
    e_z = abs(est[2] - true[2])
    d = abs(est[3] - true[3]) % 180.0
    e_yaw = min(d, 180.0 - d)

    print("真值  [x, y, z, yaw]:", [round(v, 4) for v in true])
    print("估计  [x, y, z, yaw]:", [round(v, 4) for v in est])
    print(f"误差  xy={e_xy * 1000:.2f} mm   z={e_z * 1000:.2f} mm   yaw={e_yaw:.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

