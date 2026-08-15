#!/usr/bin/env python3
"""Record timestamped fixed-L515 aligned RGB-D frames for real-data development."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.camera.calibration import (  # noqa: E402
    load_intrinsics,
    load_world_calibration,
)
from box_perception.camera.ros_rgbd import ROSAlignedRGBDSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()
    if args.frames < 1 or args.interval < 0.0:
        parser.error("--frames must be positive and --interval must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ros = config["ros"]
    camera = config["camera"]
    records = []

    with ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros["color_topic"],
        depth_topic=ros["aligned_depth_topic"],
        camera_info_topic=ros["camera_info_topic"],
        max_pair_offset_s=float(ros["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros["intrinsics_tolerance"]),
    ) as source:
        for index in range(1, args.frames + 1):
            frame = source.read(args.timeout)
            stem = f"frame_{index:04d}"
            cv2.imwrite(str(output / f"{stem}_color.png"), frame.color_bgr)
            np.save(output / f"{stem}_aligned_depth_m.npy", frame.aligned_depth_m)
            records.append(
                {
                    "index": index,
                    "color_file": f"{stem}_color.png",
                    "depth_file": f"{stem}_aligned_depth_m.npy",
                    "color_stamp_ns": frame.color_stamp_ns,
                    "depth_stamp_ns": frame.depth_stamp_ns,
                    "pair_offset_s": frame.pair_offset_s,
                }
            )
            print(
                f"saved {index}/{args.frames}: pair offset="
                f"{frame.pair_offset_s * 1000.0:+.2f} ms"
            )
            if index < args.frames and args.interval:
                time.sleep(args.interval)

    manifest = {
        "schema_version": 1,
        "purpose": "stack_seg fixed-L515 aligned RGB-D recording",
        "config": str(config_path),
        "world_frame": calibration.world_frame,
        "camera_frame": calibration.camera_frame,
        "map_sha256": calibration.map_sha256,
        "external_calibration": str(calibration.result_path),
        "world_T_camera": calibration.world_T_camera.tolist(),
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "k": intrinsics.k.tolist(),
            "distortion": intrinsics.distortion.tolist(),
            "distortion_model": intrinsics.distortion_model,
        },
        "frames": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"capture complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
