#!/usr/bin/env python3
"""交互式查看 StackMap 三维垛堆。

鼠标左键拖动：环绕旋转视角；滚轮：缩放；按 R：恢复默认视角；按 Q/Esc：退出。
默认持续监视 stack_map.json；只有文件发生变化时才重绘，因此不会持续刷屏。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from box_perception.stack.stack_map import StackMapManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stack_map", type=Path, help="StackMap JSON 文件")
    parser.add_argument("--interval", type=float, default=0.5, help="自动刷新间隔（秒）")
    parser.add_argument("--no-watch", action="store_true", help="只加载一次，不监视文件变化")
    parser.add_argument("--save", type=Path, default=None, help="保存当前视图 PNG")
    parser.add_argument("--elev", type=float, default=24.0, help="初始俯仰角")
    parser.add_argument("--azim", type=float, default=-58.0, help="初始水平旋转角")
    parser.add_argument("--x-min", type=float, default=0.5, help="X 轴最小值（米）")
    parser.add_argument("--x-max", type=float, default=2.0, help="X 轴最大值（米）")
    parser.add_argument("--y-min", type=float, default=-1.5, help="Y 轴最小值（米）")
    parser.add_argument("--y-max", type=float, default=0.0, help="Y 轴最大值（米）")
    parser.add_argument("--z-max", type=float, default=1.6, help="Z 轴最大值（米）")
    return parser.parse_args()


def _cuboid_vertices(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
) -> np.ndarray:
    length, width, height = size
    local = np.array(
        [
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=float,
    )
    angle = np.deg2rad(float(yaw_deg))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return local @ rotation.T + np.asarray(center, dtype=float)


def _draw_cuboid(ax: Any, vertices: np.ndarray, color: Any, alpha: float = 0.28) -> None:
    faces = [
        [vertices[index] for index in (0, 1, 2, 3)],
        [vertices[index] for index in (4, 5, 6, 7)],
        [vertices[index] for index in (0, 1, 5, 4)],
        [vertices[index] for index in (1, 2, 6, 5)],
        [vertices[index] for index in (2, 3, 7, 6)],
        [vertices[index] for index in (3, 0, 4, 7)],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=alpha,
        )
    )


def _box_vertices_from_state(box: Any) -> np.ndarray:
    return _cuboid_vertices(
        (box.x, box.y, box.z),
        (box.length, box.width, box.height),
        box.yaw,
    )


def _draw_scene(
    ax: Any,
    manager: StackMapManager,
    *,
    elev: float,
    azim: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_max: float,
) -> None:
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    ax.set_title(
        f"StackMap 3D | boxes={len(manager.map.boxes)} | "
        "left-drag: rotate, wheel: zoom, R: reset"
    )

    all_vertices: list[np.ndarray] = []
    tray = manager.map.tray_reference
    if isinstance(tray, dict):
        tray_pose = tray.get("pose_4dof", {})
        tray_size = tray.get("measured_size_m", {})
        if {"x_m", "y_m", "z_m", "yaw_deg"} <= tray_pose.keys() and {
            "length", "width"
        } <= tray_size.keys():
            # tray_detection 使用 top_height_above_ground 记录托盘厚度/顶面高度；
            # 兼容旧文件中可能使用的 height 字段。
            tray_height = float(
                tray_size.get("height", tray_size.get("top_height_above_ground", 0.10))
            )
            vertices = _cuboid_vertices(
                (
                    float(tray_pose["x_m"]),
                    float(tray_pose["y_m"]),
                    float(tray_pose["z_m"]) - tray_height / 2.0,
                ),
                (
                    float(tray_size["length"]),
                    float(tray_size["width"]),
                    tray_height,
                ),
                float(tray_pose["yaw_deg"]),
            )
            _draw_cuboid(ax, vertices, "saddlebrown", alpha=0.22)
            all_vertices.append(vertices)
            tray_x_axis = np.array(
                [np.cos(np.deg2rad(float(tray_pose["yaw_deg"]))),
                 np.sin(np.deg2rad(float(tray_pose["yaw_deg"]))), 0.0]
            )
            tray_y_axis = np.array([-tray_x_axis[1], tray_x_axis[0], 0.0])
            origin = np.array(
                [
                    float(tray_pose["x_m"]),
                    float(tray_pose["y_m"]),
                    float(tray_pose["z_m"]),
                ]
            )
            # 与实时 RGB 画面和 run_real_pipeline.py 完全一致：
            # +X 沿托盘长边，+Y 沿托盘短边，原点在托盘顶面中心。
            x_length = min(0.35, float(tray_size["length"]) / 2.0)
            y_length = min(0.35, float(tray_size["width"]) / 2.0)
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.006,
                tray_x_axis[0] * x_length,
                tray_x_axis[1] * x_length,
                0.0,
                color="gold",
                linewidth=2.5,
                arrow_length_ratio=0.12,
            )
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.008,
                tray_y_axis[0] * y_length,
                tray_y_axis[1] * y_length,
                0.0,
                color="cyan",
                linewidth=2.5,
                arrow_length_ratio=0.12,
            )
            ax.text(
                float(tray_pose["x_m"]),
                float(tray_pose["y_m"]),
                float(tray_pose["z_m"]) + 0.02,
                "TRAY",
                color="saddlebrown",
                weight="bold",
            )
            ax.text(
                origin[0] + tray_x_axis[0] * x_length,
                origin[1] + tray_x_axis[1] * x_length,
                origin[2] + 0.018,
                "tray +X",
                color="goldenrod",
                weight="bold",
            )
            ax.text(
                origin[0] + tray_y_axis[0] * y_length,
                origin[1] + tray_y_axis[1] * y_length,
                origin[2] + 0.018,
                "tray +Y",
                color="darkcyan",
                weight="bold",
            )

    colors = plt.get_cmap("tab10")
    by_id = {box.id: box for box in manager.map.boxes}
    for box in sorted(manager.map.boxes, key=lambda item: (item.layer, item.id)):
        vertices = _box_vertices_from_state(box)
        _draw_cuboid(ax, vertices, colors((box.layer - 1) % 10), alpha=0.34)
        all_vertices.append(vertices)
        ax.text(
            box.x,
            box.y,
            box.z + box.height / 2.0 + 0.015,
            f"B{box.id} / L{box.layer} "
            f"{box.length * 1000:.0f}x{box.width * 1000:.0f}x{box.height * 1000:.0f}mm",
            color=colors((box.layer - 1) % 10),
            weight="bold",
        )
        for lower_id in box.supported_by:
            lower = by_id.get(lower_id)
            if lower is not None:
                ax.plot(
                    [lower.x, box.x],
                    [lower.y, box.y],
                    [lower.z + lower.height / 2.0, box.z - box.height / 2.0],
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                )

    # 使用固定世界坐标范围，避免每次刷新时坐标轴自动缩放导致画面跳动。
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0.0, z_max)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def main() -> int:
    args = parse_args()
    path = args.stack_map.expanduser().resolve()
    if (
        args.interval <= 0.0
        or args.x_max <= args.x_min
        or args.y_max <= args.y_min
        or args.z_max <= 0.0
    ):
        raise ValueError("坐标范围无效，需满足 interval>0、x/y max>min、z-max>0")

    figure = plt.figure("StackMap 3D", figsize=(11, 8))
    ax = figure.add_subplot(111, projection="3d")
    current_elev, current_azim = float(args.elev), float(args.azim)

    def reset_view(_event: Any = None) -> None:
        nonlocal current_elev, current_azim
        current_elev, current_azim = float(args.elev), float(args.azim)
        ax.view_init(elev=current_elev, azim=current_azim)
        figure.canvas.draw_idle()

    def on_key(event: Any) -> None:
        if event.key in ("r", "R"):
            reset_view(event)
        elif event.key in ("q", "Q", "escape"):
            plt.close(figure)

    def on_scroll(event: Any) -> None:
        # Matplotlib 3D 后端的滚轮缩放行为不完全一致，这里显式统一缩放。
        factor = 0.85 if event.button == "up" else 1.18
        for getter, setter in (
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ):
            low, high = getter()
            center = (low + high) / 2.0
            half = (high - low) * factor / 2.0
            setter(center - half, center + half)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("key_press_event", on_key)
    figure.canvas.mpl_connect("scroll_event", on_scroll)

    last_mtime: float | None = None

    def refresh() -> None:
        nonlocal last_mtime
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime == last_mtime:
            return
        if mtime is None:
            ax.clear()
            ax.set_title(f"等待 StackMap: {path}")
            ax.set_xlim(args.x_min, args.x_max)
            ax.set_ylim(args.y_min, args.y_max)
            ax.set_zlim(0.0, args.z_max)
        else:
            manager = StackMapManager.load(path)
            _draw_scene(
                ax,
                manager,
                elev=current_elev,
                azim=current_azim,
                x_min=args.x_min,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=args.y_max,
                z_max=args.z_max,
            )
            if args.save is not None:
                save_path = args.save.expanduser().resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(save_path, dpi=150, bbox_inches="tight")
        last_mtime = mtime
        figure.canvas.draw_idle()

    refresh()
    if args.no_watch:
        plt.show(block=True)
    else:
        # 使用 GUI timer 轮询文件。文件没有变化时不调用 clear/draw，避免 3D
        # 窗口持续刷屏、闪烁和无意义的重绘。
        timer = figure.canvas.new_timer(interval=max(50, int(args.interval * 1000)))
        timer.add_callback(refresh)
        timer.start()
        plt.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
