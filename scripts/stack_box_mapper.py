#!/usr/bin/env python3
"""正式版：固定 L515 垛堆箱体建图（双箱型 A/B，四层 AABB，标准网格编号）。

本文件是入口脚本，具体功能拆分在 ``stack_mapper`` 包内：
config / types / geometry / camera / detect / boxmap / visualize。

任务定义
========
- 两种箱子（顺序均为 长×宽×高）：
    A: 0.40 × 0.30 × 0.30 m
    B: 0.42 × 0.27 × 0.21 m
- 从下往上堆 4 层：第 1、2 层为 A，第 3、4 层为 B。
- 每层 6 个箱子：箱子长轴沿托盘短边（tray +Y），短轴沿托盘长边（tray +X）。
- 编号按“标准目标区域”划分（与放置先后无关）：-Y 侧一行 1、2、3，+Y 侧一行 4、5、6。

两个运行模式
============
1) ``--mode update_tray``：检测空托盘，更新托盘参考文件（tray_reference.json）。
2) ``--mode map_stack``：加载已有托盘参考，可在码垛任意时刻打开，实时识别活动层。

窗口按键：Q=保存并退出，S=立即保存状态。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# 保证无论从哪个目录运行，都能 import 同目录下的 stack_mapper 包。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack_mapper import config  # noqa: E402
from stack_mapper.boxmap import save_boxmap, update_boxmap  # noqa: E402
from stack_mapper.camera import (  # noqa: E402
    ROSAlignedRGBDSource,
    load_intrinsics,
    load_world_calibration,
)
from stack_mapper.detect import (  # noqa: E402
    YOLOSegmentor,
    deduplicate_by_slot,
    depth_fallback_instances,
    detect_tray_from_depth,
    estimate_candidate,
    load_tray_reference,
    merge_instances,
    save_tray_reference,
)
from stack_mapper.geometry import build_ray_lookup, median_depth_masked  # noqa: E402
from stack_mapper.types import BoxMap, CandidateBox, Intrinsics, WorldCalibration  # noqa: E402
from stack_mapper.visualize import (  # noqa: E402
    BoxMap3DViewer,
    _draw_candidate_masks,
    _draw_status,
    _draw_world_rectangle,
)

# 仓库根目录，用于默认路径。
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["update_tray", "map_stack"],
        required=True,
        help="update_tray: 检测空托盘并更新托盘参考文件；map_stack: 加载托盘并建图",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument(
        "--tray-reference", type=Path, default=ROOT / "record/tray_reference/tray_reference.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "record/stack_box_map")
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-conf", type=float, default=0.2)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.50)
    parser.add_argument("--inference-hz", type=float, default=10.0)
    parser.add_argument("--depth-median-frames", type=int, default=3)
    parser.add_argument("--read-timeout", type=float, default=8.0)
    parser.add_argument("--timing-every", type=int, default=10, help="每 N 次推理打印分段耗时，0 禁用")

    # 几何校验阈值。
    parser.add_argument("--min-mask-pixels", type=int, default=1200)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.55)
    parser.add_argument("--min-top-points", type=int, default=450)
    parser.add_argument("--max-geometry-points", type=int, default=6000)
    parser.add_argument("--plane-threshold", type=float, default=0.008)
    parser.add_argument("--min-normal-z", type=float, default=0.97)
    parser.add_argument("--max-plane-rmse", type=float, default=0.008)
    parser.add_argument("--min-top-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-rectangle-fill", type=float, default=0.50)
    parser.add_argument("--min-top-inlier-ratio", type=float, default=0.15)
    parser.add_argument(
        "--layer-height-tolerance", type=float, default=config.LAYER_HEIGHT_TOLERANCE
    )
    parser.add_argument("--show-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug-reject",
        action="store_true",
        help="打印每个被拒绝候选的完整拒绝原因，用于定位识别不到的问题",
    )
    parser.add_argument(
        "--show-3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 matplotlib 3D 窗口，显示所有箱子（冻结层+活动层）",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> tuple[Intrinsics, WorldCalibration, dict[str, Any]]:
    """加载相机配置、内参与世界外参。"""
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    return intrinsics, calibration, config


def _make_manifest(intrinsics: Intrinsics, calibration: WorldCalibration) -> dict[str, Any]:
    """构造可视化投影所需的内外参 manifest。"""
    return {
        "world_T_camera": calibration.world_T_camera.tolist(),
        "intrinsics": {
            "k": intrinsics.k.tolist(),
            "distortion": intrinsics.distortion.tolist(),
        },
    }


def _make_source(intrinsics: Intrinsics, config: dict[str, Any]) -> ROSAlignedRGBDSource:
    """根据配置创建 ROS RGB-D 源。"""
    ros_config = config["ros"]
    camera_config = config["camera"]
    return ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros_config["color_topic"],
        depth_topic=ros_config["aligned_depth_topic"],
        camera_info_topic=ros_config["camera_info_topic"],
        max_pair_offset_s=float(ros_config["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera_config["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros_config["intrinsics_tolerance"]),
    )


def mode_update_tray(args: argparse.Namespace) -> int:
    """模式 1：检测空托盘并更新托盘参考文件。"""
    intrinsics, calibration, config = _load_config(args)
    ray_x, ray_y = build_ray_lookup(intrinsics)
    print("========== update_tray mode ==========")
    print("将连续读取深度帧，取中值后检测空托盘，并更新托盘参考文件")
    print(f"托盘参考文件: {args.tray_reference.expanduser().resolve()}")

    with _make_source(intrinsics, config) as source:
        depth_frames: deque[np.ndarray] = deque(maxlen=args.depth_median_frames)
        color_bgr: np.ndarray | None = None
        while len(depth_frames) < args.depth_median_frames:
            frame = source.read(args.read_timeout)
            depth_frames.append(frame.aligned_depth_m)
            if color_bgr is None:
                color_bgr = frame.color_bgr

        median_depth = median_depth_masked(depth_frames)
        tray, _artifacts = detect_tray_from_depth(
            median_depth,
            ray_x,
            ray_y,
            calibration.world_T_camera,
            frame=calibration.world_frame,
        )
        save_tray_reference(args.tray_reference, tray, calibration.map_sha256)
        print("托盘识别结果:")
        print(f"  pose_4dof: x={tray['pose_4dof']['x_m']:.4f} y={tray['pose_4dof']['y_m']:.4f} "
              f"z={tray['pose_4dof']['z_m']:.4f} yaw={tray['pose_4dof']['yaw_deg']:.2f}°")
        print(f"  size: {tray['measured_size_m']['length']:.4f} x "
              f"{tray['measured_size_m']['width']:.4f} m")
        print(f"  axis_ratio={tray['quality']['axis_ratio']:.3f} "
              f"yaw_stable={tray['quality']['yaw_stable_from_shape']}")
        print(f"已写入: {args.tray_reference.expanduser().resolve()}")

        if color_bgr is not None:
            manifest = _make_manifest(intrinsics, calibration)
            display = color_bgr.copy()
            _draw_world_rectangle(
                display,
                x=float(tray["pose_4dof"]["x_m"]),
                y=float(tray["pose_4dof"]["y_m"]),
                top_z=float(tray["pose_4dof"]["z_m"]),
                length=float(tray["measured_size_m"]["length"]),
                width=float(tray["measured_size_m"]["width"]),
                yaw=float(tray["pose_4dof"]["yaw_deg"]),
                manifest=manifest,
                color=(0, 165, 255),
                label="TRAY",
                thickness=3,
            )
            cv2.namedWindow("tray detection", cv2.WINDOW_NORMAL)
            while True:
                cv2.imshow("tray detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            cv2.destroyAllWindows()
    return 0


def mode_map_stack(args: argparse.Namespace) -> int:
    """模式 2：加载托盘参考，实时识别活动层并增量更新 boxmap。"""
    intrinsics, calibration, config = _load_config(args)
    tray = load_tray_reference(args.tray_reference, calibration.map_sha256)
    tray_pose = tray["pose_4dof"]
    ray_x, ray_y = build_ray_lookup(intrinsics)
    manifest = _make_manifest(intrinsics, calibration)
    segmentor = YOLOSegmentor(
        str(args.yolo_weights),
        device=args.yolo_device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        mask_threshold=args.yolo_mask_threshold,
    )

    output = args.output.expanduser().resolve()
    boxmap_path = output / "boxmap.json"
    depth_frames: deque[np.ndarray] = deque(maxlen=args.depth_median_frames)
    candidates: list[CandidateBox] = []
    last_inference = float("-inf")
    inference_ms = 0.0
    last_frame_time = time.monotonic()
    display_fps = 0.0
    timing_ms = {"yolo": 0.0, "median": 0.0, "geom": 0.0, "save": 0.0}
    timing_cycles = 0
    boxmap = BoxMap()
    viewer = BoxMap3DViewer() if args.show_3d else None

    print("========== map_stack mode ==========")
    print(f"托盘位姿: x={tray_pose['x_m']:.4f} y={tray_pose['y_m']:.4f} "
          f"z={tray_pose['z_m']:.4f} yaw={tray_pose['yaw_deg']:.2f}°")
    print("冻结层将按标准位置补全，活动层实时识别")
    print(f"boxmap: {boxmap_path}")
    if viewer is not None:
        viewer.start()

    with _make_source(intrinsics, config) as source:
        cv2.namedWindow("stack box mapper", cv2.WINDOW_NORMAL)
        while True:
            try:
                frame = source.read(args.read_timeout)
            except TimeoutError as exc:
                print(f"WARNING: {exc}; waiting for RGB-D recovery")
                continue
            depth_frames.append(frame.aligned_depth_m)
            now = time.monotonic()
            if (
                len(depth_frames) == args.depth_median_frames
                and now - last_inference >= 1.0 / args.inference_hz
            ):
                t0 = time.monotonic()
                instances = segmentor.segment(frame.color_bgr)
                # 深度兜底：深度区域减去 YOLO 已识别 mask，找出 YOLO 遗漏的箱子。
                yolo_union = None
                if instances:
                    yolo_union = np.asarray(instances[0]["mask"], dtype=bool).copy()
                    for inst in instances[1:]:
                        yolo_union |= np.asarray(inst["mask"], dtype=bool)
                fallback = depth_fallback_instances(
                    frame.aligned_depth_m,
                    tray_pose,
                    ray_x,
                    ray_y,
                    calibration.world_T_camera,
                    exclude_mask=yolo_union,
                )
                instances = merge_instances(instances, fallback)
                t1 = time.monotonic()
                candidates = []
                if instances:
                    union_mask = np.asarray(instances[0]["mask"], dtype=bool).copy()
                    for inst in instances[1:]:
                        union_mask |= np.asarray(inst["mask"], dtype=bool)
                    median_depth = median_depth_masked(depth_frames, union_mask)
                    t2 = time.monotonic()
                    for inst in instances:
                        try:
                            candidates.append(
                                estimate_candidate(
                                    inst, median_depth, ray_x, ray_y,
                                    calibration.world_T_camera, tray_pose, args,
                                )
                            )
                        except Exception as exc:
                            candidates.append(
                                CandidateBox(
                                    accepted=False,
                                    reasons=[f"geometry:{type(exc).__name__}"],
                                    mask=np.asarray(inst["mask"], dtype=bool),
                                    bbox=np.asarray(inst["bbox"], dtype=np.float64),
                                    yolo_confidence=float(inst["confidence"]),
                                )
                            )
                    t3 = time.monotonic()
                else:
                    t2 = t1
                    t3 = t1
                if args.debug_reject:
                    for c in candidates:
                        if not c.accepted:
                            print(f"[reject] layer={c.layer} box_type={c.box_type} "
                                  f"reasons={c.reasons}")
                accepted = deduplicate_by_slot(candidates)
                boxmap = update_boxmap(boxmap, accepted, tray, calibration.map_sha256)
                save_boxmap(boxmap_path, boxmap)
                if viewer is not None:
                    viewer.update(boxmap)
                t4 = time.monotonic()
                inference_ms = (t4 - t0) * 1000.0
                last_inference = t4
                if args.timing_every > 0:
                    timing_ms["yolo"] += (t1 - t0) * 1000.0
                    timing_ms["median"] += (t2 - t1) * 1000.0
                    timing_ms["geom"] += (t3 - t2) * 1000.0
                    timing_ms["save"] += (t4 - t3) * 1000.0
                    timing_cycles += 1
                    if timing_cycles >= args.timing_every:
                        avg = {k: v / timing_cycles for k, v in timing_ms.items()}
                        print(
                            f"[timing] n={timing_cycles} | yolo={avg['yolo']:.1f}ms "
                            f"median={avg['median']:.1f}ms geom={avg['geom']:.1f}ms "
                            f"save={avg['save']:.1f}ms | total={inference_ms:.1f}ms"
                        )
                        timing_ms = {k: 0.0 for k in timing_ms}
                        timing_cycles = 0

            # 2D 画面：托盘 + 活动层实测箱子。
            display = frame.color_bgr.copy()
            _draw_candidate_masks(display, candidates, args.show_rejected)
            tray_size = tray.get("measured_size_m", {})
            if "length" in tray_size:
                _draw_world_rectangle(
                    display,
                    x=float(tray_pose["x_m"]),
                    y=float(tray_pose["y_m"]),
                    top_z=float(tray_pose["z_m"]),
                    length=float(tray_size["length"]),
                    width=float(tray_size["width"]),
                    yaw=float(tray_pose["yaw_deg"]),
                    manifest=manifest,
                    color=(0, 165, 255),
                    label="TRAY",
                    thickness=2,
                )
            for box in boxmap.boxes:
                if box.source != "measured":
                    continue  # 2D 画面只显示活动层实测箱子；冻结/标准补全层仅 3D 显示。
                _draw_world_rectangle(
                    display,
                    x=box.x,
                    y=box.y,
                    top_z=box.z + box.height * 0.5,
                    length=box.length,
                    width=box.width,
                    yaw=box.yaw,
                    manifest=manifest,
                    color=(40, 230, 40),
                    label=f"L{box.layer}-{box.id} {box.box_type}",
                    thickness=3,
                )
            current_time = time.monotonic()
            instant_fps = 1.0 / max(current_time - last_frame_time, 1e-6)
            last_frame_time = current_time
            display_fps = instant_fps if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant_fps
            _draw_status(display, boxmap.active_layer, len(boxmap.boxes), inference_ms, display_fps)
            cv2.imshow("stack box mapper", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                save_boxmap(boxmap_path, boxmap)
                print(f"saved boxmap: {boxmap_path}")
            if key in (ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()
    save_boxmap(boxmap_path, boxmap)
    print(f"finished; boxmap saved to {boxmap_path}")
    return 0


def main() -> int:
    """入口：按模式分发。"""
    args = parse_args()
    if args.mode == "update_tray":
        return mode_update_tray(args)
    return mode_map_stack(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
