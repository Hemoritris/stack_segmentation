"""Recorded fixed-L515 before/after RGB-D -> world-frame box 4DoF."""

from __future__ import annotations

import json
import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera.calibration import cam_to_world
from .geometry.box_optimizer import BoxOptimizer
from .geometry.plane_fitting import fit_top_plane, is_horizontal
from .geometry.pointcloud import aligned_depth_to_pointcloud
from .geometry.rectangle_init import fit_min_area_rect
from .geometry.tray_detection import TrayDetectionArtifacts, detect_tray_from_depth
from .segmentation.yolo_segmentor import YOLOSegmentor
from .temporal.change_detector import detect_change
from .temporal.height_map import HeightMap
from .temporal.new_box_association import associate_new_box


@dataclass(frozen=True)
class RecordedRGBD:
    root: Path
    manifest: dict[str, Any]
    median_depth_m: np.ndarray
    display_color_bgr: np.ndarray


@dataclass(frozen=True)
class RealPipelineArtifacts:
    image_change_mask: np.ndarray
    tray_image_mask: np.ndarray
    height_before: np.ndarray
    height_after: np.ndarray
    world_change_mask: np.ndarray
    world_roi: tuple[tuple[float, float], tuple[float, float]]
    top_points_world: np.ndarray
    tray_top_points_world: np.ndarray
    yolo_image_mask: np.ndarray | None = None


def _median_depth(depths: list[np.ndarray]) -> np.ndarray:
    if not depths:
        raise ValueError("recording contains no depth frames")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(depths), axis=0).astype(np.float32)


def load_recording(root: str | Path, max_frames: int | None = None) -> RecordedRGBD:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"recording manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames", []))
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"recording has no frames: {root}")

    depths: list[np.ndarray] = []
    color: np.ndarray | None = None
    expected_shape = (
        int(manifest["intrinsics"]["height"]),
        int(manifest["intrinsics"]["width"]),
    )
    for frame in frames:
        depth_path = root / frame["depth_file"]
        color_path = root / frame["color_file"]
        if not depth_path.is_file() or not color_path.is_file():
            raise FileNotFoundError(f"recording frame file missing: {depth_path} or {color_path}")
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        if depth.shape != expected_shape:
            raise ValueError(f"unexpected depth shape {depth.shape}; expected {expected_shape}")
        depths.append(depth)
        if color is None:
            color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if color is None or color.shape[:2] != expected_shape:
                raise ValueError(f"failed to read matching color frame: {color_path}")
    assert color is not None
    return RecordedRGBD(root, manifest, _median_depth(depths), color)


def validate_recording_pair(before: RecordedRGBD, after: RecordedRGBD) -> None:
    for key in ("world_frame", "camera_frame", "map_sha256"):
        if before.manifest.get(key) != after.manifest.get(key):
            raise ValueError(f"before/after {key} mismatch")
    numeric_fields = (
        ("world_T_camera",),
        ("intrinsics", "k"),
        ("intrinsics", "distortion"),
    )
    for path in numeric_fields:
        left: Any = before.manifest
        right: Any = after.manifest
        for key in path:
            left, right = left[key], right[key]
        if not np.allclose(left, right, rtol=0.0, atol=1e-9):
            raise ValueError(f"before/after {'/'.join(path)} mismatch")


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise RuntimeError("no connected change component found")
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = 1 + int(np.argmax(areas))
    if int(stats[index, cv2.CC_STAT_AREA]) < int(min_area):
        raise RuntimeError(
            f"largest change component has only {int(stats[index, cv2.CC_STAT_AREA])} cells/pixels; "
            f"minimum is {min_area}"
        )
    return labels == index


def normalize_box_yaw_deg(yaw_deg: float) -> float:
    """Normalize a rectangular box's 180-degree-symmetric yaw to [-90, 90)."""
    return float((yaw_deg + 90.0) % 180.0 - 90.0)


def pose_4dof_world_to_reference(
    pose_world: dict[str, float], reference_world: dict[str, float]
) -> dict[str, float]:
    """Express a world-frame ``x, y, z, yaw`` pose in a planar reference frame."""
    yaw_reference = np.deg2rad(float(reference_world["yaw_deg"]))
    dx = float(pose_world["x_m"]) - float(reference_world["x_m"])
    dy = float(pose_world["y_m"]) - float(reference_world["y_m"])
    c, s = np.cos(yaw_reference), np.sin(yaw_reference)
    return {
        "x_m": float(c * dx + s * dy),
        "y_m": float(-s * dx + c * dy),
        "z_m": float(pose_world["z_m"]) - float(reference_world["z_m"]),
        "yaw_deg": normalize_box_yaw_deg(
            float(pose_world["yaw_deg"]) - float(reference_world["yaw_deg"])
        ),
    }


def pose_4dof_reference_to_world(
    pose_reference: dict[str, float], reference_world: dict[str, float]
) -> dict[str, float]:
    """Transform a planar-reference ``x, y, z, yaw`` pose back to world."""
    yaw_reference = np.deg2rad(float(reference_world["yaw_deg"]))
    c, s = np.cos(yaw_reference), np.sin(yaw_reference)
    x_local = float(pose_reference["x_m"])
    y_local = float(pose_reference["y_m"])
    return {
        "x_m": float(reference_world["x_m"]) + c * x_local - s * y_local,
        "y_m": float(reference_world["y_m"]) + s * x_local + c * y_local,
        "z_m": float(reference_world["z_m"]) + float(pose_reference["z_m"]),
        "yaw_deg": normalize_box_yaw_deg(
            float(reference_world["yaw_deg"]) + float(pose_reference["yaw_deg"])
        ),
    }


def project_world_points_to_image(
    points_world: np.ndarray,
    world_t_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
    """Project world-frame XYZ points into the original distorted color image.

    Points behind the optical camera are returned as ``[nan, nan]``.
    ``world_t_camera`` follows the project convention
    ``P_world = world_T_camera @ P_camera``.
    """
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(world_t_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("world_t_camera must have shape (4, 4)")
    camera_t_world = np.linalg.inv(transform)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_camera = (homogeneous @ camera_t_world.T)[:, :3]
    visible = points_camera[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(visible):
        k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        d = np.asarray([] if distortion is None else distortion, dtype=np.float64).reshape(-1)
        projected, _ = cv2.projectPoints(
            points_camera[visible],
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            k,
            d if len(d) else None,
        )
        pixels[visible] = projected.reshape(-1, 2)
    return pixels


def estimate_box_from_world_clouds(
    points_before: np.ndarray,
    points_after: np.ndarray,
    *,
    box_size: tuple[float, float, float],
    roi: tuple[tuple[float, float], tuple[float, float]],
    grid_size_m: float = 0.005,
    min_height_change_m: float = 0.05,
    min_change_area_m2: float = 0.02,
    plane_distance_threshold_m: float = 0.008,
    changed_world_points: np.ndarray | None = None,
) -> tuple[dict[str, Any], RealPipelineArtifacts]:
    length, width, expected_height = (float(value) for value in box_size)
    height_map = HeightMap(roi[0], roi[1], grid_size_m, aggregation="median")
    h_before = height_map.build(points_before)
    h_after = height_map.build(points_after)
    if changed_world_points is None:
        change = detect_change(
            h_before,
            h_after,
            grid_size_m=grid_size_m,
            min_height_diff_m=min_height_change_m,
            min_area_m2=min_change_area_m2,
            morph_kernel=3,
        )
        min_cells = max(1, int(round(min_change_area_m2 / grid_size_m**2)))
        component = _largest_component(change, min_cells)
        ix, iy, valid = height_map.world_to_indices(np.asarray(points_after)[:, :2])
        keep = np.zeros(len(points_after), dtype=bool)
        keep[valid] = component[iy[valid], ix[valid]]
        changed_points = np.asarray(points_after, dtype=np.float64)[keep]
    else:
        changed_points = np.asarray(changed_world_points, dtype=np.float64)
        if changed_points.ndim != 2 or changed_points.shape[1] != 3:
            raise ValueError("changed_world_points must have shape (N, 3)")
        if len(changed_points) < 100:
            raise RuntimeError(
                f"only {len(changed_points)} YOLO/depth-selected world points are available"
            )
        ix, iy, valid = height_map.world_to_indices(changed_points[:, :2])
        component = np.zeros_like(h_after, dtype=bool)
        component[iy[valid], ix[valid]] = True

    support_values = h_before[component & np.isfinite(h_before)]
    if len(support_values) < 20:
        expanded = cv2.dilate(component.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        support_values = h_before[expanded & np.isfinite(h_before)]
    if len(support_values) < 20:
        raise RuntimeError("insufficient before-frame support surface samples")
    support_z = float(np.median(support_values))

    if len(changed_points) < 100:
        raise RuntimeError(f"only {len(changed_points)} changed world points are available")

    plane = fit_top_plane(
        changed_points,
        distance_threshold=plane_distance_threshold_m,
        max_iter=400,
    )
    if not is_horizontal(plane.normal, 0.95):
        raise RuntimeError(f"detected plane is not horizontal enough: normal={plane.normal.tolist()}")

    cx, cy, measured_length, measured_width, yaw_init_deg = fit_min_area_rect(plane.points[:, :2])
    optimizer = BoxOptimizer(length, width)
    x, y, yaw_rad, fit_cost = optimizer.optimize(
        plane.points[:, :2],
        np.array([cx, cy, np.deg2rad(yaw_init_deg)], dtype=np.float64),
    )
    yaw_deg = normalize_box_yaw_deg(np.rad2deg(yaw_rad))
    measured_height = float(plane.height - support_z)
    z_center = float(plane.height - expected_height / 2.0)

    result: dict[str, Any] = {
        "frame": "world",
        "pose_4dof": {
            "x_m": x,
            "y_m": y,
            "z_m": z_center,
            "yaw_deg": yaw_deg,
        },
        "yaw_convention": {
            "reference_frame": "world",
            "axis": "box_length_axis",
            "zero_direction": "world_+X",
            "positive_direction": "counterclockwise_about_world_+Z",
            "range_deg": "[-90, 90)",
            "symmetry_deg": 180,
        },
        "box_size_prior_m": {"length": length, "width": width, "height": expected_height},
        "measured_size_m": {
            "length": measured_length,
            "width": measured_width,
            "height": measured_height,
        },
        "size_error_m": {
            "length": measured_length - length,
            "width": measured_width - width,
            "height": measured_height - expected_height,
        },
        "surface": {
            "support_z_m": support_z,
            "top_z_m": float(plane.height),
            "normal": plane.normal.tolist(),
            "plane_rmse_m": float(plane.plane_rmse),
        },
        "quality": {
            "top_plane_points": int(len(plane.points)),
            "changed_points": int(len(changed_points)),
            "optimizer_cost": float(fit_cost),
        },
        "world_roi": {
            "x_min": roi[0][0],
            "x_max": roi[0][1],
            "y_min": roi[1][0],
            "y_max": roi[1][1],
        },
    }
    artifacts = RealPipelineArtifacts(
        image_change_mask=np.empty((0, 0), dtype=bool),
        tray_image_mask=np.empty((0, 0), dtype=bool),
        height_before=h_before,
        height_after=h_after,
        world_change_mask=component,
        world_roi=roi,
        top_points_world=plane.points,
        tray_top_points_world=np.empty((0, 3), dtype=np.float64),
    )
    return result, artifacts


def run_recorded_real_pipeline(
    before: RecordedRGBD,
    after: RecordedRGBD,
    *,
    box_size: tuple[float, float, float],
    cloud_stride: int = 2,
    grid_size_m: float = 0.005,
    min_depth_change_m: float = 0.05,
    min_height_change_m: float = 0.05,
    min_change_area_m2: float = 0.02,
    roi_margin_m: float = 0.15,
    tray_min_elevation_m: float = 0.10,
    tray_max_elevation_m: float = 0.55,
    tray_min_area_pixels: int = 5000,
    tray_plane_distance_threshold_m: float = 0.008,
    tray_reference: dict[str, Any] | None = None,
    yolo_segmentor: YOLOSegmentor | None = None,
    yolo_min_overlap: float = 0.2,
) -> tuple[dict[str, Any], RealPipelineArtifacts]:
    validate_recording_pair(before, after)
    b = before.median_depth_m
    a = after.median_depth_m
    intrinsics = after.manifest["intrinsics"]
    k = np.asarray(intrinsics["k"], dtype=np.float64)
    distortion = np.asarray(intrinsics["distortion"], dtype=np.float64)
    world_t_camera = np.asarray(after.manifest["world_T_camera"], dtype=np.float64)

    world_frame = str(after.manifest["world_frame"])
    if tray_reference is None:
        # The tray is deliberately detected first from the empty Before recording.
        tray, tray_artifacts = detect_tray_from_depth(
            b,
            k,
            distortion,
            world_t_camera,
            frame=world_frame,
            min_elevation_m=tray_min_elevation_m,
            max_elevation_m=tray_max_elevation_m,
            min_area_pixels=tray_min_area_pixels,
            plane_distance_threshold_m=tray_plane_distance_threshold_m,
        )
    else:
        tray = copy.deepcopy(tray_reference)
        if tray.get("frame") != world_frame:
            raise ValueError(
                f"frozen tray frame {tray.get('frame')!r} does not match {world_frame!r}"
            )
        pose_values = [
            tray.get("pose_4dof", {}).get(key)
            for key in ("x_m", "y_m", "z_m", "yaw_deg")
        ]
        if not np.all(np.isfinite(np.asarray(pose_values, dtype=np.float64))):
            raise ValueError("frozen tray reference has an invalid pose_4dof")
        tray_artifacts = TrayDetectionArtifacts(
            image_mask=np.zeros(b.shape, dtype=bool),
            top_points_world=np.empty((0, 3), dtype=np.float64),
            world_z_image=np.empty((0, 0), dtype=np.float64),
        )

    valid = np.isfinite(b) & np.isfinite(a)
    image_change = valid & ((b - a) >= min_depth_change_m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_change = cv2.morphologyEx(image_change.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    image_change = cv2.morphologyEx(image_change, cv2.MORPH_OPEN, kernel)
    depth_component = _largest_component(image_change, min_area=500)
    image_component = depth_component
    yolo_mask: np.ndarray | None = None
    segmentation_info: dict[str, Any] = {
        "method": "depth_change_component",
        "yolo_enabled": yolo_segmentor is not None,
    }
    if yolo_segmentor is not None:
        instances = yolo_segmentor.segment(after.display_color_bgr)
        observation = associate_new_box(
            instances,
            image_change.astype(bool),
            min_overlap=float(yolo_min_overlap),
        )
        segmentation_info["instances"] = len(instances)
        if observation is not None and int(np.count_nonzero(observation.instance_mask)) >= 500:
            yolo_mask = np.asarray(observation.instance_mask, dtype=bool)
            image_component = yolo_mask
            segmentation_info.update(
                {
                    "method": "yolo_mask_intersection_with_depth_change",
                    "confidence": float(observation.yolo_confidence),
                    "change_overlap": float(observation.change_overlap),
                    "mask_pixels": int(np.count_nonzero(yolo_mask)),
                }
            )
        else:
            segmentation_info["fallback"] = "depth_change_component"

    changed_camera = aligned_depth_to_pointcloud(
        a, k, distortion, mask=image_component, min_depth_m=0.2, max_depth_m=6.0
    )
    changed_world = cam_to_world(changed_camera, world_t_camera)
    if len(changed_world) < 100:
        raise RuntimeError("image-space depth change did not yield enough valid world points")
    x_range = (
        float(np.min(changed_world[:, 0]) - roi_margin_m),
        float(np.max(changed_world[:, 0]) + roi_margin_m),
    )
    y_range = (
        float(np.min(changed_world[:, 1]) - roi_margin_m),
        float(np.max(changed_world[:, 1]) + roi_margin_m),
    )

    before_world = cam_to_world(
        aligned_depth_to_pointcloud(
            b, k, distortion, stride=cloud_stride, min_depth_m=0.2, max_depth_m=5.0
        ),
        world_t_camera,
    )
    after_world = cam_to_world(
        aligned_depth_to_pointcloud(
            a, k, distortion, stride=cloud_stride, min_depth_m=0.2, max_depth_m=5.0
        ),
        world_t_camera,
    )
    result, artifacts = estimate_box_from_world_clouds(
        before_world,
        after_world,
        box_size=box_size,
        roi=(x_range, y_range),
        grid_size_m=grid_size_m,
        min_height_change_m=min_height_change_m,
        min_change_area_m2=min_change_area_m2,
        changed_world_points=changed_world,
    )
    result["segmentation"] = segmentation_info
    result["frame"] = str(after.manifest["world_frame"])
    result["yaw_convention"]["reference_frame"] = result["frame"]
    box_in_tray = pose_4dof_world_to_reference(result["pose_4dof"], tray["pose_4dof"])
    box_in_tray["bottom_z_m"] = float(box_in_tray["z_m"] - box_size[2] / 2.0)
    result["tray"] = tray
    result["box_pose_in_tray_4dof"] = box_in_tray
    result["placement_reference"] = {
        "frame": "tray",
        "origin": "tray_top_surface_center",
        "note": "Use tray-relative x/y/z/yaw targets, then transform them with tray pose_4dof",
    }
    result["recordings"] = {"before": str(before.root), "after": str(after.root)}
    result["map_sha256"] = str(after.manifest["map_sha256"])
    artifacts = RealPipelineArtifacts(
        image_change_mask=image_component,
        tray_image_mask=tray_artifacts.image_mask,
        height_before=artifacts.height_before,
        height_after=artifacts.height_after,
        world_change_mask=artifacts.world_change_mask,
        world_roi=artifacts.world_roi,
        top_points_world=artifacts.top_points_world,
        tray_top_points_world=tray_artifacts.top_points_world,
        yolo_image_mask=yolo_mask,
    )
    return result, artifacts
