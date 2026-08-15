#!/usr/bin/env python3
"""Estimate one newly placed box from recorded fixed-L515 before/after RGB-D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.real_pipeline import (  # noqa: E402
    load_recording,
    run_recorded_real_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "record/before")
    parser.add_argument("--after", type=Path, default=ROOT / "record/after")
    parser.add_argument("--output", type=Path, default=ROOT / "record/real_result")
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        default=(0.40, 0.30, 0.30),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--cloud-stride", type=int, default=2)
    parser.add_argument("--grid-size", type=float, default=0.005)
    parser.add_argument("--min-depth-change", type=float, default=0.05)
    parser.add_argument("--min-height-change", type=float, default=0.05)
    parser.add_argument("--min-change-area", type=float, default=0.02)
    parser.add_argument("--roi-margin", type=float, default=0.15)
    return parser.parse_args()


def _save_debug(output: Path, after_color: np.ndarray, result: dict, artifacts) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output / "height_before.npy", artifacts.height_before)
    np.save(output / "height_after.npy", artifacts.height_after)
    np.save(output / "top_points_world.npy", artifacts.top_points_world)
    cv2.imwrite(
        str(output / "image_change_mask.png"),
        artifacts.image_change_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output / "world_change_mask.png"),
        artifacts.world_change_mask.astype(np.uint8) * 255,
    )

    overlay = after_color.copy()
    contours, _ = cv2.findContours(
        artifacts.image_change_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    pose = result["pose_4dof"]
    measured = result["measured_size_m"]
    lines = [
        f"world xyz=({pose['x_m']:.3f}, {pose['y_m']:.3f}, {pose['z_m']:.3f}) m",
        f"yaw={pose['yaw_deg']:.2f} deg",
        f"measured LWH=({measured['length']:.3f}, {measured['width']:.3f}, {measured['height']:.3f}) m",
    ]
    for index, line in enumerate(lines):
        y = 36 + index * 34
        cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output / "overlay.png"), overlay)


def main() -> int:
    args = parse_args()
    if min(args.box_size) <= 0 or args.cloud_stride < 1 or args.grid_size <= 0:
        raise ValueError("box size, cloud stride, and grid size must be positive")
    before = load_recording(args.before, args.max_frames)
    after = load_recording(args.after, args.max_frames)
    result, artifacts = run_recorded_real_pipeline(
        before,
        after,
        box_size=tuple(args.box_size),
        cloud_stride=args.cloud_stride,
        grid_size_m=args.grid_size,
        min_depth_change_m=args.min_depth_change,
        min_height_change_m=args.min_height_change,
        min_change_area_m2=args.min_change_area,
        roi_margin_m=args.roi_margin,
    )
    _save_debug(args.output.expanduser().resolve(), after.display_color_bgr, result, artifacts)

    pose = result["pose_4dof"]
    measured = result["measured_size_m"]
    error = result["size_error_m"]
    surface = result["surface"]
    print("========== fixed L515 recorded real-box result ==========")
    print(f"frame: {result['frame']}")
    print(
        "pose: "
        f"x={pose['x_m']:.4f} m, y={pose['y_m']:.4f} m, z={pose['z_m']:.4f} m, "
        f"yaw={pose['yaw_deg']:.2f} deg"
    )
    print(
        "measured L/W/H: "
        f"{measured['length']:.4f}/{measured['width']:.4f}/{measured['height']:.4f} m"
    )
    print(
        "size error L/W/H: "
        f"{error['length'] * 1000:+.1f}/{error['width'] * 1000:+.1f}/"
        f"{error['height'] * 1000:+.1f} mm"
    )
    print(
        f"support/top: {surface['support_z_m']:.4f}/{surface['top_z_m']:.4f} m; "
        f"plane RMSE={surface['plane_rmse_m'] * 1000:.2f} mm"
    )
    print(f"result: {args.output.expanduser().resolve() / 'result.json'}")
    print(f"overlay: {args.output.expanduser().resolve() / 'overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

