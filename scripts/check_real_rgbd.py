#!/usr/bin/env python3
"""Read one fixed-L515 RGB-D pair and verify the camera-to-world data path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.camera.calibration import (  # noqa: E402
    cam_to_world,
    load_intrinsics,
    load_world_calibration,
)
from box_perception.camera.ros_rgbd import ROSAlignedRGBDSource  # noqa: E402
from box_perception.geometry.pointcloud import aligned_depth_to_pointcloud  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--output", type=Path, help="optional directory for one RGB-D sample")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    ros = config["ros"]
    camera = config["camera"]

    print("========== fixed L515 real RGB-D preflight ==========")
    print(f"intrinsics: {intrinsics.width}x{intrinsics.height}, fx/fy={intrinsics.fx:.6f}/{intrinsics.fy:.6f}")
    print(f"external calibration: {calibration.result_path}")
    print(f"frames: {calibration.world_frame} <- {calibration.camera_frame}")
    print(f"map SHA256: {calibration.map_sha256}")
    with ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros["color_topic"],
        depth_topic=ros["aligned_depth_topic"],
        camera_info_topic=ros["camera_info_topic"],
        max_pair_offset_s=float(ros["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros["intrinsics_tolerance"]),
    ) as source:
        frame = source.read(args.timeout)

    points_camera = aligned_depth_to_pointcloud(
        frame.aligned_depth_m,
        frame.intrinsics.k,
        frame.intrinsics.distortion,
        stride=args.stride,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
    )
    points_world = cam_to_world(points_camera, calibration.world_T_camera)
    valid_depth = frame.aligned_depth_m[np.isfinite(frame.aligned_depth_m)]
    if len(points_world) == 0:
        raise RuntimeError("aligned depth contains no valid points in the configured range")
    print(f"[PASS] synchronized pair offset={frame.pair_offset_s * 1000.0:+.2f} ms")
    print(
        f"[PASS] depth valid={len(valid_depth)}/{frame.aligned_depth_m.size}, "
        f"range={float(valid_depth.min()):.3f}..{float(valid_depth.max()):.3f} m"
    )
    print(
        "[PASS] world cloud points="
        f"{len(points_world)}, xyz min={np.min(points_world, axis=0).round(4).tolist()}, "
        f"max={np.max(points_world, axis=0).round(4).tolist()}"
    )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / "color.png"), frame.color_bgr)
        np.save(output / "aligned_depth_m.npy", frame.aligned_depth_m)
        np.save(output / "points_world.npy", points_world)
        metadata = {
            "color_stamp_ns": frame.color_stamp_ns,
            "depth_stamp_ns": frame.depth_stamp_ns,
            "pair_offset_s": frame.pair_offset_s,
            "camera_frame": calibration.camera_frame,
            "world_frame": calibration.world_frame,
            "map_sha256": calibration.map_sha256,
            "external_calibration": str(calibration.result_path),
            "world_T_camera": calibration.world_T_camera.tolist(),
            "k": intrinsics.k.tolist(),
            "distortion": intrinsics.distortion.tolist(),
            "point_stride": args.stride,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"saved sample: {output}")

    if args.show:
        depth_vis = np.nan_to_num(frame.aligned_depth_m, nan=0.0)
        scale = max(float(np.nanpercentile(depth_vis[depth_vis > 0], 95)), 1e-6)
        depth_vis = cv2.applyColorMap(
            np.clip(depth_vis / scale * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        cv2.imshow("fixed L515 color", frame.color_bgr)
        cv2.imshow("fixed L515 aligned depth", depth_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
