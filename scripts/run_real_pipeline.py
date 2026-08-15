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
    project_world_points_to_image,
    run_recorded_real_pipeline,
)
from box_perception.geometry.tray_detection import detect_tray_from_depth  # noqa: E402


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
    parser.add_argument("--tray-min-elevation", type=float, default=0.10)
    parser.add_argument("--tray-max-elevation", type=float, default=0.55)
    parser.add_argument("--tray-min-area-pixels", type=int, default=5000)
    parser.add_argument("--tray-plane-threshold", type=float, default=0.008)
    parser.add_argument(
        "--tray-only",
        action="store_true",
        help="detect and freeze the empty tray from --before; do not process a box",
    )
    parser.add_argument(
        "--tray-reference",
        type=Path,
        help="reuse a frozen tray_reference.json instead of detecting the tray again",
    )
    return parser.parse_args()


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _draw_pose_axes(overlay: np.ndarray, result: dict, manifest: dict) -> None:
    pose = result["pose_4dof"]
    size = result["box_size_prior_m"]
    top_z = float(result["surface"]["top_z_m"])
    center = np.array([pose["x_m"], pose["y_m"], top_z], dtype=np.float64)
    yaw = np.deg2rad(float(pose["yaw_deg"]))
    length_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    width_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    half_l = float(size["length"]) / 2.0
    half_w = float(size["width"]) / 2.0
    corners = np.array(
        [
            center + sx * half_l * length_axis + sy * half_w * width_axis
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    world_points = np.vstack(
        [
            center,
            center + np.array([0.25, 0.0, 0.0]),
            center - half_l * length_axis,
            center + half_l * length_axis,
            center - half_w * width_axis,
            center + half_w * width_axis,
            corners,
        ]
    )
    intrinsics = manifest["intrinsics"]
    pixels = project_world_points_to_image(
        world_points,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(intrinsics["k"], dtype=np.float64),
        np.asarray(intrinsics["distortion"], dtype=np.float64),
    )
    if not np.all(np.isfinite(pixels)):
        return
    p = [_pixel(point) for point in pixels]
    center_px, world_x_px = p[0], p[1]
    length_a, length_b = p[2], p[3]
    width_a, width_b = p[4], p[5]
    corner_pixels = np.asarray(p[6:10], dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(overlay, [corner_pixels], True, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(overlay, center_px, world_x_px, (255, 80, 0), 3, cv2.LINE_AA, tipLength=0.12)
    cv2.line(overlay, length_a, length_b, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.arrowedLine(overlay, center_px, length_b, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.12)
    cv2.line(overlay, width_a, width_b, (0, 200, 0), 4, cv2.LINE_AA)
    cv2.circle(overlay, center_px, 7, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(overlay, "+X world", world_x_px, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)
    cv2.putText(overlay, "+L / yaw", length_b, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(overlay, "W", width_b, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)


def _draw_tray_pose(
    overlay: np.ndarray,
    result: dict,
    manifest: dict,
    *,
    labels: bool,
) -> None:
    tray = result["tray"]
    pose = tray["pose_4dof"]
    size = tray["measured_size_m"]
    center = np.array([pose["x_m"], pose["y_m"], pose["z_m"]], dtype=np.float64)
    yaw = np.deg2rad(float(pose["yaw_deg"]))
    x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    half_l = float(size["length"]) / 2.0
    half_w = float(size["width"]) / 2.0
    corners = np.array(
        [
            center + sx * half_l * x_axis + sy * half_w * y_axis
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    points = np.vstack(
        [
            center,
            center + np.array([0.30, 0.0, 0.0]),
            center + min(0.35, half_l) * x_axis,
            center + min(0.35, half_w) * y_axis,
            corners,
        ]
    )
    intrinsics = manifest["intrinsics"]
    pixels = project_world_points_to_image(
        points,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(intrinsics["k"], dtype=np.float64),
        np.asarray(intrinsics["distortion"], dtype=np.float64),
    )
    if not np.all(np.isfinite(pixels)):
        return
    p = [_pixel(point) for point in pixels]
    center_px, world_x_px, tray_x_px, tray_y_px = p[:4]
    corners_px = np.asarray(p[4:8], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [corners_px], True, (0, 165, 255), 4, cv2.LINE_AA)
    cv2.arrowedLine(overlay, center_px, tray_x_px, (255, 255, 0), 4, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(overlay, center_px, tray_y_px, (0, 255, 255), 4, cv2.LINE_AA, tipLength=0.12)
    cv2.circle(overlay, center_px, 8, (255, 255, 255), -1, cv2.LINE_AA)
    if labels:
        cv2.arrowedLine(
            overlay, center_px, world_x_px, (255, 80, 0), 3, cv2.LINE_AA, tipLength=0.12
        )
        cv2.putText(
            overlay, "+X world", world_x_px, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2
        )
        cv2.putText(
            overlay, "tray +X", tray_x_px, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
        )
        cv2.putText(
            overlay, "tray +Y", tray_y_px, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )


def _save_debug(
    output: Path,
    before_color: np.ndarray,
    after_color: np.ndarray,
    result: dict,
    artifacts,
    manifest: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output / "height_before.npy", artifacts.height_before)
    np.save(output / "height_after.npy", artifacts.height_after)
    np.save(output / "top_points_world.npy", artifacts.top_points_world)
    np.save(output / "tray_top_points_world.npy", artifacts.tray_top_points_world)
    cv2.imwrite(
        str(output / "image_change_mask.png"),
        artifacts.image_change_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output / "world_change_mask.png"),
        artifacts.world_change_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output / "tray_image_mask.png"),
        artifacts.tray_image_mask.astype(np.uint8) * 255,
    )

    _save_tray_debug(output, before_color, result["tray"], artifacts, manifest)

    overlay = after_color.copy()
    contours, _ = cv2.findContours(
        artifacts.image_change_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    _draw_tray_pose(overlay, result, manifest, labels=False)
    _draw_pose_axes(overlay, result, manifest)
    pose = result["pose_4dof"]
    pose_in_tray = result["box_pose_in_tray_4dof"]
    measured = result["measured_size_m"]
    lines = [
        f"world xyz=({pose['x_m']:.3f}, {pose['y_m']:.3f}, {pose['z_m']:.3f}) m",
        f"yaw={pose['yaw_deg']:.2f} deg",
        f"tray xyzyaw=({pose_in_tray['x_m']:.3f}, {pose_in_tray['y_m']:.3f}, "
        f"{pose_in_tray['z_m']:.3f}, {pose_in_tray['yaw_deg']:.2f}deg)",
        f"measured LWH=({measured['length']:.3f}, {measured['width']:.3f}, {measured['height']:.3f}) m",
    ]
    for index, line in enumerate(lines):
        y = 36 + index * 34
        cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output / "overlay.png"), overlay)


def _save_tray_debug(
    output: Path,
    before_color: np.ndarray,
    tray: dict,
    artifacts,
    manifest: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tray_overlay = before_color.copy()
    tray_mask = getattr(artifacts, "tray_image_mask", None)
    if tray_mask is None:
        tray_mask = artifacts.image_mask
    tray_contours, _ = cv2.findContours(
        tray_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(tray_overlay, tray_contours, -1, (0, 255, 0), 3)
    _draw_tray_pose(tray_overlay, {"tray": tray}, manifest, labels=True)
    tray_pose = tray["pose_4dof"]
    tray_size = tray["measured_size_m"]
    tray_lines = [
        f"tray world xyz=({tray_pose['x_m']:.3f}, {tray_pose['y_m']:.3f}, {tray_pose['z_m']:.3f}) m",
        f"tray yaw={tray_pose['yaw_deg']:.2f} deg",
        f"tray LWH=({tray_size['length']:.3f}, {tray_size['width']:.3f}, "
        f"{tray_size['top_height_above_ground']:.3f}) m",
    ]
    for index, line in enumerate(tray_lines):
        y = 36 + index * 34
        cv2.putText(tray_overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(tray_overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output / "tray_overlay.png"), tray_overlay)


def _detect_and_save_tray_only(args: argparse.Namespace, before) -> int:
    manifest = before.manifest
    intrinsics = manifest["intrinsics"]
    tray, artifacts = detect_tray_from_depth(
        before.median_depth_m,
        np.asarray(intrinsics["k"], dtype=np.float64),
        np.asarray(intrinsics["distortion"], dtype=np.float64),
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        frame=str(manifest["world_frame"]),
        min_elevation_m=args.tray_min_elevation,
        max_elevation_m=args.tray_max_elevation,
        min_area_pixels=args.tray_min_area_pixels,
        plane_distance_threshold_m=args.tray_plane_threshold,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "frozen tray placement reference",
        "map_sha256": str(manifest["map_sha256"]),
        "source_recording": str(before.root),
        "tray": tray,
    }
    (output / "tray_reference.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cv2.imwrite(
        str(output / "tray_image_mask.png"), artifacts.image_mask.astype(np.uint8) * 255
    )
    np.save(output / "tray_top_points_world.npy", artifacts.top_points_world)
    _save_tray_debug(output, before.display_color_bgr, tray, artifacts, manifest)
    pose = tray["pose_4dof"]
    size = tray["measured_size_m"]
    quality = tray["quality"]
    print("========== frozen empty-tray reference ==========")
    print(
        f"frame={tray['frame']}; x={pose['x_m']:.4f} m, y={pose['y_m']:.4f} m, "
        f"top_z={pose['z_m']:.4f} m, yaw={pose['yaw_deg']:.2f} deg"
    )
    print(
        f"measured L/W/H={size['length']:.4f}/{size['width']:.4f}/"
        f"{size['top_height_above_ground']:.4f} m; axis_ratio={quality['axis_ratio']:.3f}; "
        f"yaw_stable={quality['yaw_stable_from_shape']}"
    )
    print(f"reference: {output / 'tray_reference.json'}")
    print(f"overlay: {output / 'tray_overlay.png'}")
    return 0


def _load_frozen_tray(path: Path, manifest: dict) -> dict:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("map_sha256") != manifest.get("map_sha256"):
        raise ValueError(
            f"tray reference map SHA mismatch: {payload.get('map_sha256')} != "
            f"{manifest.get('map_sha256')}"
        )
    tray = payload.get("tray")
    if not isinstance(tray, dict):
        raise ValueError(f"{resolved}: missing tray object")
    return tray


def main() -> int:
    args = parse_args()
    if min(args.box_size) <= 0 or args.cloud_stride < 1 or args.grid_size <= 0:
        raise ValueError("box size, cloud stride, and grid size must be positive")
    if args.tray_only and args.tray_reference is not None:
        raise ValueError("--tray-only and --tray-reference cannot be used together")
    before = load_recording(args.before, args.max_frames)
    if args.tray_only:
        return _detect_and_save_tray_only(args, before)
    after = load_recording(args.after, args.max_frames)
    tray_reference = (
        _load_frozen_tray(args.tray_reference, after.manifest)
        if args.tray_reference is not None
        else None
    )
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
        tray_min_elevation_m=args.tray_min_elevation,
        tray_max_elevation_m=args.tray_max_elevation,
        tray_min_area_pixels=args.tray_min_area_pixels,
        tray_plane_distance_threshold_m=args.tray_plane_threshold,
        tray_reference=tray_reference,
    )
    _save_debug(
        args.output.expanduser().resolve(),
        before.display_color_bgr,
        after.display_color_bgr,
        result,
        artifacts,
        after.manifest,
    )

    pose = result["pose_4dof"]
    measured = result["measured_size_m"]
    error = result["size_error_m"]
    surface = result["surface"]
    tray = result["tray"]
    tray_pose = tray["pose_4dof"]
    tray_size = tray["measured_size_m"]
    tray_quality = tray["quality"]
    pose_in_tray = result["box_pose_in_tray_4dof"]
    print("========== fixed L515 recorded real-box result ==========")
    print(f"frame: {result['frame']}")
    print(
        "tray world pose: "
        f"x={tray_pose['x_m']:.4f} m, y={tray_pose['y_m']:.4f} m, "
        f"top_z={tray_pose['z_m']:.4f} m, yaw={tray_pose['yaw_deg']:.2f} deg"
    )
    print(
        "tray measured L/W/H: "
        f"{tray_size['length']:.4f}/{tray_size['width']:.4f}/"
        f"{tray_size['top_height_above_ground']:.4f} m; "
        f"axis_ratio={tray_quality['axis_ratio']:.3f}, "
        f"yaw_stable={tray_quality['yaw_stable_from_shape']}"
    )
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
        "box in tray frame: "
        f"x={pose_in_tray['x_m']:.4f} m, y={pose_in_tray['y_m']:.4f} m, "
        f"z={pose_in_tray['z_m']:.4f} m, yaw={pose_in_tray['yaw_deg']:.2f} deg, "
        f"bottom={pose_in_tray['bottom_z_m']:.4f} m"
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
    print(f"tray overlay: {args.output.expanduser().resolve() / 'tray_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
