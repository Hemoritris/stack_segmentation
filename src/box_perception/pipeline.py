"""端到端 pipeline（合成场景）：把 M2~M8 串起来跑一遍。

在没有相机 / 标定时，用合成正俯视相机生成前后两帧深度图，验证整条几何链路。
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera.calibration import cam_to_world
from .core.types import BoxInstance
from .geometry.box_optimizer import BoxOptimizer
from .geometry.plane_fitting import fit_top_plane
from .geometry.pointcloud import depth_to_pointcloud
from .geometry.rectangle_init import fit_min_area_rect
from .synthetic import (
    make_overhead_camera,
    make_tilted_camera,
    ortho_depth_to_world,
    render_perspective_depth,
    render_scene_depth,
)
from .temporal.change_detector import detect_change
from .temporal.height_map import HeightMap
from .temporal.new_box_association import associate_new_box


def _estimate(pts_before, pts_after, roi, grid_size, box_center, box_size, base_z, yaw_deg):
    """M2~M8：给定前后两帧 world 点云，估计新箱 4DoF 并返回结果字典。"""
    length, width, height = box_size
    bx, by = box_center

    hm = HeightMap(x_range=roi[0], y_range=roi[1], grid_size_m=grid_size, aggregation="median")
    h_before = hm.build(pts_before)
    h_after = hm.build(pts_after)

    change = detect_change(
        h_before, h_after,
        grid_size_m=grid_size,
        min_height_diff_m=0.01,
        min_area_m2=0.01,
    )

    yolo = cv2.dilate(change.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    obs = associate_new_box(
        [BoxInstance(mask=yolo, bbox=None, confidence=0.9)],
        change,
        min_overlap=0.1,
    )
    if obs is None:
        raise RuntimeError("M4: 未关联到新箱")
    new_mask = np.asarray(obs.instance_mask, dtype=bool)

    ix, iy, valid = hm.world_to_indices(pts_after[:, :2])
    keep_cells = new_mask[iy[valid], ix[valid]]
    keep = np.zeros(len(pts_after), dtype=bool)
    keep[valid] = keep_cells
    box_pts = pts_after[keep]
    if len(box_pts) < 8:
        raise RuntimeError("M5: 单箱点云点数不足")

    plane = fit_top_plane(box_pts, distance_threshold=0.01)
    z_center = plane.height - height / 2.0

    xy = plane.points[:, :2]
    cx, cy, _, _, yaw_init_deg = fit_min_area_rect(xy)
    init = np.array([cx, cy, np.deg2rad(yaw_init_deg)])
    x, y, yaw_rad, cost = BoxOptimizer(length, width).optimize(xy, init)

    return {
        "est": (x, y, z_center, np.rad2deg(yaw_rad)),
        "true": (bx, by, base_z + height / 2.0, yaw_deg),
        "cost": cost,
    }


def run_synthetic_demo(
    box_center=(0.1, -0.05),
    box_size=(0.6, 0.4, 0.35),
    yaw_deg: float = 25.0,
    base_z: float = 0.0,
    existing_boxes=None,
    cam_height: float = 2.5,
    image_size=(240, 320),
    roi=((-0.9, 0.9), (-0.9, 0.9)),
    grid_size: float = 0.01,
    depth_noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """在合成场景跑一遍 M2~M8，返回 est / true 位姿与代价。

    existing_boxes: 已存在箱体列表，元素 (cx, cy, length, width, top_z, yaw_deg)。
    base_z: 新箱放置面的高度；新箱顶面高度 = base_z + height。
    """
    length, width, height = box_size
    bx, by = box_center
    new_top_z = base_z + height
    img_h, img_w = image_size

    intrinsics = make_overhead_camera(img_w, img_h, cam_height, pixel_size_m=0.005)

    # M0/M5 反投影半段：渲染前后深度，转 world 点云
    before_boxes = list(existing_boxes or [])
    after_boxes = before_boxes + [(bx, by, length, width, new_top_z, yaw_deg)]
    depth_before = render_scene_depth(
        img_w, img_h, cam_height, intrinsics,
        boxes=before_boxes, noise=depth_noise, seed=seed,
    )
    depth_after = render_scene_depth(
        img_w, img_h, cam_height, intrinsics,
        boxes=after_boxes, noise=depth_noise, seed=seed + 1,
    )
    pts_before = ortho_depth_to_world(depth_before, **intrinsics)
    pts_after = ortho_depth_to_world(depth_after, **intrinsics)

    return _estimate(pts_before, pts_after, roi, grid_size, box_center, box_size, base_z, yaw_deg)


def run_tilted_demo(
    box_center=(0.1, -0.05),
    box_size=(0.6, 0.4, 0.35),
    yaw_deg: float = 25.0,
    base_z: float = 0.0,
    existing_boxes=None,
    cam_height: float = 2.5,
    image_size=(240, 320),
    roi=((-0.9, 0.9), (-0.9, 0.9)),
    grid_size: float = 0.01,
    tilt_deg: float = 0.0,
    depth_noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """用可倾斜的透视相机跑一遍 M2~M8。"""
    length, width, height = box_size
    bx, by = box_center
    new_top_z = base_z + height
    img_h, img_w = image_size

    intrinsics, T_cam_world = make_tilted_camera(
        img_w, img_h, cam_height, fx=float(img_w), tilt_deg=tilt_deg
    )
    before_boxes = list(existing_boxes or [])
    after_boxes = before_boxes + [(bx, by, length, width, new_top_z, yaw_deg)]

    depth_before = render_perspective_depth(
        img_w, img_h, intrinsics, T_cam_world,
        boxes=before_boxes, noise=depth_noise, seed=seed,
    )
    depth_after = render_perspective_depth(
        img_w, img_h, intrinsics, T_cam_world,
        boxes=after_boxes, noise=depth_noise, seed=seed + 1,
    )

    pts_before = cam_to_world(depth_to_pointcloud(depth_before, **intrinsics), T_cam_world)
    pts_after = cam_to_world(depth_to_pointcloud(depth_after, **intrinsics), T_cam_world)

    return _estimate(pts_before, pts_after, roi, grid_size, box_center, box_size, base_z, yaw_deg)
