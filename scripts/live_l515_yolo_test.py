#!/usr/bin/env python3
"""固定 L515 YOLO-Seg + RGB-D 箱体实时测试。

按 K：把当前画面后的新箱视为一次放置完成，运行 Before/After 几何估计；
按 Q：退出。窗口显示实时 RGB、托盘/整个垛堆箱体 4DoF 和 FPS。使用
``--show-live-yolo`` 时，会以受限频率叠加 YOLO 实例 mask、边界和置信度；不启用时
YOLO 仍只参与 K 触发后的内部计算。

本脚本需要同一个 Python 环境同时提供 rclpy、ultralytics、scipy、numpy 和
OpenCV。ROS 2 环境若没有 ultralytics，请先准备统一运行环境。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.camera.calibration import load_intrinsics, load_world_calibration  # noqa: E402
from box_perception.camera.ros_rgbd import RGBDFrame, ROSAlignedRGBDSource  # noqa: E402
from box_perception.geometry.tray_detection import detect_tray_from_depth  # noqa: E402
from box_perception.real_pipeline import (  # noqa: E402
    RecordedRGBD,
    project_world_points_to_image,
    run_recorded_real_pipeline,
)
from box_perception.segmentation.yolo_segmentor import YOLOSegmentor  # noqa: E402
from box_perception.stack.stack_map import StackMapManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("reidentify_tray", "reuse_tray"),
        default="reidentify_tray",
        help="reidentify_tray=启动时重新识别托盘；reuse_tray=加载已有托盘参考",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument(
        "--tray-reference",
        type=Path,
        default=ROOT / "record/tray_reference/tray_reference.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "record/live_yolo_test")
    parser.add_argument(
        "--resume-stack-map",
        type=Path,
        default=None,
        help="加载已有 StackMap 继续增量建模，并校验地图 SHA256",
    )
    parser.add_argument(
        "--stack-map-output",
        type=Path,
        default=None,
        help="StackMap 输出 JSON；默认保存到 output/stack_map.json",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        default=(0.40, 0.30, 0.30),
    )
    parser.add_argument("--start-frames", type=int, default=5)
    parser.add_argument("--after-frames", type=int, default=5)
    parser.add_argument("--tray-frames", type=int, default=5)
    parser.add_argument("--read-timeout", type=float, default=5.0)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-min-overlap", type=float, default=0.20)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.50)
    parser.add_argument(
        "--show-live-yolo",
        action="store_true",
        help="在实时 RGB 画面叠加最近一次 YOLO 分割 mask、边界和置信度",
    )
    parser.add_argument(
        "--live-yolo-hz",
        type=float,
        default=2.0,
        help="实时分割推理频率；显示帧率不受此值限制",
    )
    parser.add_argument(
        "--live-yolo-alpha",
        type=float,
        default=0.35,
        help="实时分割 mask 透明度，范围 (0, 1]",
    )
    parser.add_argument("--min-depth-change", type=float, default=0.05)
    parser.add_argument("--min-height-change", type=float, default=0.05)
    parser.add_argument("--min-change-area", type=float, default=0.02)
    parser.add_argument("--tray-min-elevation", type=float, default=0.10)
    parser.add_argument("--tray-max-elevation", type=float, default=0.55)
    parser.add_argument("--tray-min-area-pixels", type=int, default=5000)
    parser.add_argument("--tray-plane-threshold", type=float, default=0.008)
    return parser.parse_args()


def _median_depth(frames: list[RGBDFrame]) -> np.ndarray:
    if not frames:
        raise ValueError("没有可用 RGB-D 帧")
    # Pixels unseen in every frame legitimately remain NaN.  nanmedian emits a
    # RuntimeWarning for those pixels even though that is the representation we
    # want to keep for subsequent point-cloud processing.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(np.stack([frame.aligned_depth_m for frame in frames]), axis=0).astype(
            np.float32
        )


def _manifest_from_config(
    intrinsics: Any,
    calibration: Any,
) -> dict[str, Any]:
    return {
        "world_frame": calibration.world_frame,
        "camera_frame": calibration.camera_frame,
        "map_sha256": calibration.map_sha256,
        "world_T_camera": np.asarray(calibration.world_T_camera).tolist(),
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "k": np.asarray(intrinsics.k).tolist(),
            "distortion": np.asarray(intrinsics.distortion).tolist(),
        },
    }


def _load_tray_reference(path: Path, map_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("map_sha256") != map_sha256:
        raise ValueError(
            f"托盘参考地图 SHA256 不匹配: {payload.get('map_sha256')} != {map_sha256}"
        )
    tray = payload.get("tray")
    if not isinstance(tray, dict):
        raise ValueError(f"托盘参考缺少 tray: {resolved}")
    return tray


def _project(points: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    return project_world_points_to_image(
        points,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["k"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["distortion"], dtype=np.float64),
    )


def _draw_projected_pose(
    image: np.ndarray,
    pose: dict[str, float],
    length: float,
    width: float,
    top_z: float,
    manifest: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    center = np.array([pose["x_m"], pose["y_m"], top_z], dtype=np.float64)
    yaw = np.deg2rad(float(pose["yaw_deg"]))
    x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    points = np.vstack(
        [
            center,
            center + 0.25 * np.array([1.0, 0.0, 0.0]),
            center + 0.40 * x_axis,
            center + 0.25 * y_axis,
            *[
                center + sx * length / 2.0 * x_axis + sy * width / 2.0 * y_axis
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ],
        ]
    )
    pixels = _project(points, manifest)
    if not np.all(np.isfinite(pixels)):
        return
    p = [tuple(np.round(point).astype(int)) for point in pixels]
    center_px, world_x_px, x_px, y_px = p[:4]
    corners = np.asarray(p[4:], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [corners], True, color, 3, cv2.LINE_AA)
    cv2.arrowedLine(image, center_px, x_px, color, 3, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, center_px, y_px, (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, center_px, world_x_px, (255, 80, 0), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.circle(image, center_px, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(image, label, (center_px[0] + 8, center_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _draw_stack_map(image: np.ndarray, stack_manager: StackMapManager, manifest: dict[str, Any]) -> None:
    colors = [
        (255, 0, 255),
        (0, 255, 0),
        (255, 180, 0),
        (0, 200, 255),
        (180, 80, 255),
    ]
    for box in sorted(stack_manager.map.boxes, key=lambda item: (item.layer, item.id)):
        _draw_projected_pose(
            image,
            {"x_m": box.x, "y_m": box.y, "yaw_deg": box.yaw},
            box.length,
            box.width,
            box.z + box.height * 0.5,
            manifest,
            colors[(box.layer - 1) % len(colors)],
            f"B{box.id}/L{box.layer}",
        )


def _draw_status(
    image: np.ndarray,
    *,
    mode: str,
    status: str,
    fps: float,
    kept: int,
    result: dict[str, Any] | None,
    yolo_status: str | None = None,
) -> None:
    lines = [
        f"mode={mode}  FPS={fps:.1f}  kept={kept}",
        "K: capture placed box / Q: quit",
        status,
    ]
    if yolo_status is not None:
        lines.append(yolo_status)
    if result is not None:
        pose = result["pose_4dof"]
        tray_pose = result["box_pose_in_tray_4dof"]
        lines.extend(
            [
                f"world xyz=({pose['x_m']:.3f}, {pose['y_m']:.3f}, {pose['z_m']:.3f}) m yaw={pose['yaw_deg']:.2f} deg",
            f"tray xyz=({tray_pose['x_m']:.3f}, {tray_pose['y_m']:.3f}, {tray_pose['z_m']:.3f}) m yaw={tray_pose['yaw_deg']:.2f} deg",
            ]
        )
    for index, line in enumerate(lines):
        y = 30 + index * 30
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_live_yolo_instances(
    image: np.ndarray,
    instances: list[Any],
    alpha: float,
) -> None:
    """Overlay cached YOLO instances without changing the geometry pipeline."""
    if not instances:
        return
    palette = [
        (40, 220, 40),
        (255, 120, 30),
        (40, 180, 255),
        (220, 60, 220),
        (80, 220, 220),
    ]
    overlay = image.copy()
    valid_instances: list[tuple[Any, np.ndarray, tuple[int, int, int]]] = []
    for index, instance in enumerate(instances):
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != image.shape[:2] or not np.any(mask):
            continue
        color = palette[index % len(palette)]
        overlay[mask] = color
        valid_instances.append((instance, mask, color))
    if not valid_instances:
        return
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)
    for instance, mask, color in valid_instances:
        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        bbox = getattr(instance, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = np.round(np.asarray(bbox).reshape(-1)[:4]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"box {float(instance.confidence):.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _save_tray_reference(output: Path, map_sha256: str, tray: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "tray_reference.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "live L515 tray reference",
                "map_sha256": map_sha256,
                "tray": tray,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _collect_frames(source: ROSAlignedRGBDSource, count: int, timeout: float) -> list[RGBDFrame]:
    return [source.read(timeout) for _ in range(count)]


def main() -> int:
    args = parse_args()
    if min(args.box_size) <= 0 or min(args.start_frames, args.after_frames, args.tray_frames) < 1:
        raise ValueError("箱体尺寸和采集帧数必须为正")
    if args.live_yolo_hz <= 0.0:
        raise ValueError("--live-yolo-hz 必须大于 0")
    if not 0.0 < args.live_yolo_alpha <= 1.0:
        raise ValueError("--live-yolo-alpha 必须在 (0, 1] 内")

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(config_path, "color")
    calibration = load_world_calibration(config_path)
    ros_config = config["ros"]
    camera_config = config["camera"]
    manifest = _manifest_from_config(intrinsics, calibration)
    segmentor = YOLOSegmentor(
        str(args.yolo_weights),
        device=args.yolo_device,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        mask_threshold=args.yolo_mask_threshold,
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = "waiting for first RGB-D frame"
    tray: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    kept = 0

    print("========== live fixed-L515 YOLO test ==========")
    print(f"mode: {args.mode}")
    print(f"weights: {args.yolo_weights.expanduser().resolve()}")
    if args.show_live_yolo:
        print(
            f"live YOLO overlay: enabled at <= {args.live_yolo_hz:.2f} Hz, "
            f"alpha={args.live_yolo_alpha:.2f}"
        )
    else:
        print("live YOLO overlay: disabled (use --show-live-yolo to enable)")
    print("keys: K=process a newly placed box, Q=quit")

    with ROSAlignedRGBDSource(
        intrinsics,
        color_topic=ros_config["color_topic"],
        depth_topic=ros_config["aligned_depth_topic"],
        camera_info_topic=ros_config["camera_info_topic"],
        max_pair_offset_s=float(ros_config["max_pair_offset_s"]),
        depth_integer_scale_m=float(camera_config["depth_integer_scale_m"]),
        intrinsics_tolerance=float(ros_config["intrinsics_tolerance"]),
    ) as source:
        if args.mode == "reidentify_tray":
            print(f"Collecting {args.tray_frames} tray RGB-D frames...")
            tray_frames = _collect_frames(source, args.tray_frames, args.read_timeout)
            tray_depth = _median_depth(tray_frames)
            tray, _artifacts = detect_tray_from_depth(
                tray_depth,
                intrinsics.k,
                intrinsics.distortion,
                calibration.world_T_camera,
                frame=calibration.world_frame,
                min_elevation_m=args.tray_min_elevation,
                max_elevation_m=args.tray_max_elevation,
                min_area_pixels=args.tray_min_area_pixels,
                plane_distance_threshold_m=args.tray_plane_threshold,
            )
            tray_path = _save_tray_reference(output, calibration.map_sha256, tray)
            print(
                f"tray updated: x={tray['pose_4dof']['x_m']:.3f}, "
                f"y={tray['pose_4dof']['y_m']:.3f}, "
                f"yaw={tray['pose_4dof']['yaw_deg']:.2f} deg"
            )
            print(f"tray reference: {tray_path}")
        else:
            tray = _load_tray_reference(args.tray_reference, calibration.map_sha256)
            print(f"tray reference loaded: {args.tray_reference.expanduser().resolve()}")

        if args.resume_stack_map is not None:
            stack_manager = StackMapManager.load(
                args.resume_stack_map,
                expected_map_sha256=calibration.map_sha256,
            )
            if stack_manager.map.tray_reference is None:
                stack_manager.map.tray_reference = copy.deepcopy(tray)
            print(
                f"StackMap resumed: {args.resume_stack_map.expanduser().resolve()} "
                f"({len(stack_manager.map.boxes)} boxes)"
            )
        else:
            stack_manager = StackMapManager(
                world_frame=calibration.world_frame,
                map_sha256=calibration.map_sha256,
                tray_reference=copy.deepcopy(tray),
            )
        stack_map_output = (
            args.stack_map_output.expanduser().resolve()
            if args.stack_map_output is not None
            else output / "stack_map.json"
        )
        stack_manager.save(stack_map_output)
        print(f"StackMap output: {stack_map_output}")

        print(f"Collecting {args.start_frames} startup RGB-D frames...")
        start_frames = _collect_frames(source, args.start_frames, args.read_timeout)
        start_depth = _median_depth(start_frames)
        start_color = start_frames[-1].color_bgr.copy()
        before = RecordedRGBD(Path("live_before"), manifest, start_depth, start_color)
        cv2.namedWindow("fixed L515 live YOLO test", cv2.WINDOW_NORMAL)
        last_time = time.monotonic()
        fps = 0.0
        live_instances: list[Any] = []
        live_yolo_last_run = float("-inf")
        live_yolo_inference_ms = 0.0

        while True:
            try:
                frame = source.read(args.read_timeout)
            except TimeoutError as exc:
                status = f"RGB-D timeout; waiting for stream recovery: {exc}"
                print(f"WARNING: {status}")
                continue
            display = frame.color_bgr.copy()
            now = time.monotonic()
            if args.show_live_yolo and now - live_yolo_last_run >= 1.0 / args.live_yolo_hz:
                inference_start = time.monotonic()
                try:
                    live_instances = segmentor.segment(frame.color_bgr)
                    live_yolo_inference_ms = (time.monotonic() - inference_start) * 1000.0
                except Exception as exc:
                    status = f"live YOLO failed: {type(exc).__name__}: {exc}"
                    print(f"WARNING: {status}")
                live_yolo_last_run = time.monotonic()
            if args.show_live_yolo:
                _draw_live_yolo_instances(display, live_instances, args.live_yolo_alpha)
            if tray is not None:
                tray_pose = tray["pose_4dof"]
                tray_size = tray["measured_size_m"]
                _draw_projected_pose(
                    display,
                    tray_pose,
                    float(tray_size["length"]),
                    float(tray_size["width"]),
                    float(tray_pose["z_m"]),
                    manifest,
                    (0, 165, 255),
                    "TRAY",
                )
            _draw_stack_map(display, stack_manager, manifest)
            now = time.monotonic()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            yolo_status = None
            if args.show_live_yolo:
                yolo_status = (
                    f"YOLO instances={len(live_instances)}  "
                    f"inference={live_yolo_inference_ms:.1f} ms"
                )
            _draw_status(
                display,
                mode=args.mode,
                status=status,
                fps=fps,
                kept=kept,
                result=last_result,
                yolo_status=yolo_status,
            )
            cv2.imshow("fixed L515 live YOLO test", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key not in (ord("k"), ord("K")):
                continue

            try:
                status = f"capturing {args.after_frames} after frames..."
                cv2.waitKey(1)
                after_frames = _collect_frames(source, args.after_frames, args.read_timeout)
                after_depth = _median_depth(after_frames)
                after_color = after_frames[-1].color_bgr.copy()
                after = RecordedRGBD(
                    Path(f"live_after_{kept + 1:03d}"), manifest, after_depth, after_color
                )
                result, _artifacts = run_recorded_real_pipeline(
                    before,
                    after,
                    box_size=tuple(args.box_size),
                    min_depth_change_m=args.min_depth_change,
                    min_height_change_m=args.min_height_change,
                    min_change_area_m2=args.min_change_area,
                    tray_min_elevation_m=args.tray_min_elevation,
                    tray_max_elevation_m=args.tray_max_elevation,
                    tray_min_area_pixels=args.tray_min_area_pixels,
                    tray_plane_distance_threshold_m=args.tray_plane_threshold,
                    tray_reference=copy.deepcopy(tray),
                    yolo_segmentor=segmentor,
                    yolo_min_overlap=args.yolo_min_overlap,
                )
                kept += 1
                last_result = result
                before = after
                result_path = output / f"result_{kept:03d}.json"
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                box = stack_manager.add_result(result, timestamp=time.time())
                stack_manager.save(stack_map_output)
                pose = result["pose_4dof"]
                status = (
                    f"K accepted #{kept}: B{box.id}/L{box.layer} "
                    f"world=({pose['x_m']:.3f}, {pose['y_m']:.3f}, "
                    f"{pose['z_m']:.3f}) yaw={pose['yaw_deg']:.2f} deg"
                )
                print(status)
                print(f"saved: {result_path}")
                print(f"saved StackMap: {stack_map_output}")
            except Exception as exc:
                status = f"K failed: {type(exc).__name__}: {exc}"
                print(f"ERROR: {status}")

        cv2.destroyAllWindows()
        stack_manager.save(stack_map_output)
    print(f"Live test finished; saved {kept} result(s) to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; live test closed.")
        raise SystemExit(130)
