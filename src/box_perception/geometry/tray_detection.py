"""Detect the elevated tray top before recognizing newly placed boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..camera.calibration import cam_to_world
from .plane_fitting import fit_top_plane, is_horizontal
from .pointcloud import aligned_depth_to_pointcloud
from .rectangle_init import fit_min_area_rect


@dataclass(frozen=True)
class TrayDetectionArtifacts:
    image_mask: np.ndarray
    top_points_world: np.ndarray
    world_z_image: np.ndarray


def _largest_component(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        raise RuntimeError("no elevated tray component was found")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < min_area_pixels:
        raise RuntimeError(
            f"largest elevated component has {area} pixels; minimum is {min_area_pixels}"
        )
    return labels == index


def _estimate_ground_height(points_world: np.ndarray) -> float:
    z = np.asarray(points_world, dtype=np.float64)[:, 2]
    z = z[np.isfinite(z)]
    if len(z) < 100:
        raise RuntimeError("insufficient world points for ground-height estimation")
    lower, upper = np.percentile(z, [2.0, 75.0])
    ground = z[(z >= lower) & (z <= upper)]
    if len(ground) < 100:
        raise RuntimeError("ground-height robust subset is empty")
    return float(np.median(ground))


def detect_tray_from_depth(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    world_t_camera: np.ndarray,
    *,
    frame: str = "world",
    min_elevation_m: float = 0.10,
    max_elevation_m: float = 0.55,
    min_area_pixels: int = 5000,
    plane_distance_threshold_m: float = 0.008,
    max_plane_points: int = 12000,
    min_axis_ratio_for_stable_yaw: float = 1.03,
) -> tuple[dict[str, Any], TrayDetectionArtifacts]:
    """Estimate the largest elevated horizontal tray top in one median depth image.

    The tray frame origin is its top-surface center. Tray +X follows the measured
    long edge, +Y follows the short edge, and +Z is the world up direction.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    if not 0.0 < min_elevation_m < max_elevation_m:
        raise ValueError("tray elevation bounds are invalid")
    if min_area_pixels < 1 or max_plane_points < 100:
        raise ValueError("tray area and plane-point limits must be positive")

    valid = np.isfinite(depth) & (depth >= 0.2) & (depth <= 6.0)
    rows, cols = np.nonzero(valid)
    points_camera = aligned_depth_to_pointcloud(
        depth,
        camera_matrix,
        distortion,
        mask=valid,
        min_depth_m=0.2,
        max_depth_m=6.0,
    )
    points_world = cam_to_world(points_camera, world_t_camera)
    ground_z = _estimate_ground_height(points_world)

    world_z_image = np.full(depth.shape, np.nan, dtype=np.float64)
    world_z_image[rows, cols] = points_world[:, 2]
    elevated = (
        np.isfinite(world_z_image)
        & (world_z_image >= ground_z + min_elevation_m)
        & (world_z_image <= ground_z + max_elevation_m)
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    elevated = cv2.morphologyEx(elevated.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    elevated = cv2.morphologyEx(elevated, cv2.MORPH_OPEN, open_kernel)
    component = _largest_component(elevated, min_area_pixels)

    tray_points = points_world[component[rows, cols]]
    if len(tray_points) < 100:
        raise RuntimeError("elevated tray mask did not produce enough world points")
    if len(tray_points) > max_plane_points:
        rng = np.random.default_rng(0)
        plane_input = tray_points[rng.choice(len(tray_points), max_plane_points, replace=False)]
    else:
        plane_input = tray_points
    initial_plane = fit_top_plane(
        plane_input,
        distance_threshold=plane_distance_threshold_m,
        max_iter=400,
    )
    if not is_horizontal(initial_plane.normal, 0.95):
        raise RuntimeError(
            f"detected tray plane is not horizontal: normal={initial_plane.normal.tolist()}"
        )

    normal = initial_plane.normal
    plane_offset = -float(np.median(initial_plane.points @ normal))
    distances = np.abs(tray_points @ normal + plane_offset)
    top_points = tray_points[distances <= plane_distance_threshold_m]
    if len(top_points) < 100:
        raise RuntimeError("too few full-resolution tray top-plane inliers")
    top_z = float(np.median(top_points[:, 2]))
    plane_rmse = float(np.sqrt(np.mean(distances[distances <= plane_distance_threshold_m] ** 2)))
    cx, cy, length, width, yaw_0_180 = fit_min_area_rect(top_points[:, :2])
    yaw_deg = float((yaw_0_180 + 90.0) % 180.0 - 90.0)
    axis_ratio = float(length / width) if width > 0.0 else float("inf")

    result: dict[str, Any] = {
        "frame": str(frame),
        "pose_4dof": {
            "x_m": cx,
            "y_m": cy,
            "z_m": top_z,
            "yaw_deg": yaw_deg,
        },
        "frame_definition": {
            "origin": "tray_top_surface_center",
            "x_axis": "measured_long_edge",
            "y_axis": "measured_short_edge",
            "z_axis": "world_+Z",
            "yaw_zero": "world_+X",
            "yaw_positive": "counterclockwise_about_world_+Z",
            "yaw_range_deg": "[-90, 90)",
            "symmetry_deg": 180,
        },
        "measured_size_m": {
            "length": length,
            "width": width,
            "top_height_above_ground": top_z - ground_z,
        },
        "surface": {
            "ground_z_m": ground_z,
            "top_z_m": top_z,
            "normal": normal.tolist(),
            "plane_rmse_m": plane_rmse,
        },
        "quality": {
            "mask_pixels": int(component.sum()),
            "top_plane_points": int(len(top_points)),
            "axis_ratio": axis_ratio,
            "yaw_stable_from_shape": bool(axis_ratio >= min_axis_ratio_for_stable_yaw),
            "minimum_stable_axis_ratio": float(min_axis_ratio_for_stable_yaw),
        },
    }
    return result, TrayDetectionArtifacts(component, top_points, world_z_image)
