"""可视化：OpenCV 2D 画面与 matplotlib 3D 窗口。"""

from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np

from .geometry import project_world_points_to_image
from .types import BoxMap, CandidateBox


# ---------------------------------------------------------------------------
# 2D 画面
# ---------------------------------------------------------------------------

def _draw_world_rectangle(
    image: np.ndarray,
    x: float,
    y: float,
    top_z: float,
    length: float,
    width: float,
    yaw: float,
    manifest: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    """把世界坐标下的箱体矩形投影回图像并绘制。"""
    yaw_rad = math.radians(yaw)
    ux = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    uy = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)])
    center = np.array([x, y])
    corners = np.array(
        [center + sx * length * 0.5 * ux + sy * width * 0.5 * uy
         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
        dtype=np.float64,
    )
    axes = np.array(
        [
            [x + 0.18 * math.cos(yaw_rad), y + 0.18 * math.sin(yaw_rad), top_z],
            [x - 0.14 * math.sin(yaw_rad), y + 0.14 * math.cos(yaw_rad), top_z],
        ],
        dtype=np.float64,
    )
    points = np.vstack(
        [np.array([[x, y, top_z]]), axes, np.column_stack([corners, np.full(4, top_z)])]
    )
    pixels = project_world_points_to_image(
        points,
        np.asarray(manifest["world_T_camera"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["k"], dtype=np.float64),
        np.asarray(manifest["intrinsics"]["distortion"], dtype=np.float64),
    )
    if not np.all(np.isfinite(pixels)):
        return
    p = [tuple(np.round(pt).astype(int)) for pt in pixels]
    cv2.polylines(
        image, [np.asarray(p[3:], dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness
    )
    cv2.arrowedLine(image, p[0], p[1], color, thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, p[0], p[2], (255, 255, 0), thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(
        image, label, (p[0][0] + 6, p[0][1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA,
    )


def _draw_candidate_masks(
    image: np.ndarray, candidates: list[CandidateBox], show_rejected: bool
) -> None:
    """绘制候选 mask 轮廓与标签（绿=通过、红=拒绝）。"""
    overlay = image.copy()
    for c in candidates:
        if c.accepted:
            color = (40, 220, 40)
            overlay[c.mask] = color
        elif show_rejected:
            color = (40, 40, 230)
        else:
            continue
        contours, _ = cv2.findContours(
            c.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        if c.bbox is not None and len(c.bbox.reshape(-1)) >= 4:
            x1, y1, x2, y2 = np.round(c.bbox.reshape(-1)[:4]).astype(int)
            label = (
                f"L{c.layer} slot{c.slot_id} {c.box_type} G={c.geometry_score:.2f}"
                if c.accepted
                else "REJECT " + ",".join(c.reasons[:2])
            )
            cv2.putText(
                image, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, dst=image)


def _draw_status(
    image: np.ndarray,
    active_layer: int,
    box_count: int,
    inference_ms: float,
    display_fps: float,
) -> None:
    """绘制顶部状态栏。"""
    lines = [
        f"active_layer=L{active_layer}  boxes={box_count}  infer={inference_ms:.1f}ms",
        f"display={display_fps:.1f} FPS",
        "Q: save+quit  S: save state",
    ]
    for index, line in enumerate(lines):
        y = 28 + index * 27
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)


# ---------------------------------------------------------------------------
# 3D 窗口
# ---------------------------------------------------------------------------

def _cuboid_vertices(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
) -> np.ndarray:
    """生成长方体 8 个顶点。"""
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


def _draw_cuboid(ax: Any, vertices: np.ndarray, color: Any, alpha: float = 0.34) -> None:
    """在 3D 轴上绘制一个长方体。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def _draw_3d_scene(
    ax: Any,
    boxmap: BoxMap,
    *,
    elev: float,
    azim: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """绘制整个 3D 场景：托盘 + 所有箱子（三色区分来源）。"""
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    std = sum(1 for b in boxmap.boxes if b.source == "standard")
    meas = sum(1 for b in boxmap.boxes if b.source == "measured")
    ax.set_title(
        f"BoxMap 3D | active L{boxmap.active_layer} | standard={std} measured={meas}"
    )

    tray = boxmap.tray_reference
    if isinstance(tray, dict):
        tray_pose = tray.get("pose_4dof", {})
        tray_size = tray.get("measured_size_m", {})
        if {"x_m", "y_m", "z_m", "yaw_deg"} <= tray_pose.keys() and {
            "length", "width"
        } <= tray_size.keys():
            tray_height = float(
                tray_size.get("height", tray_size.get("top_height_above_ground", 0.10))
            )
            vertices = _cuboid_vertices(
                (float(tray_pose["x_m"]), float(tray_pose["y_m"]),
                 float(tray_pose["z_m"]) - tray_height / 2.0),
                (float(tray_size["length"]), float(tray_size["width"]), tray_height),
                float(tray_pose["yaw_deg"]),
            )
            _draw_cuboid(ax, vertices, "saddlebrown", alpha=0.22)
            tray_x_axis = np.array(
                [np.cos(np.deg2rad(float(tray_pose["yaw_deg"]))),
                 np.sin(np.deg2rad(float(tray_pose["yaw_deg"]))), 0.0]
            )
            tray_y_axis = np.array([-tray_x_axis[1], tray_x_axis[0], 0.0])
            origin = np.array(
                [float(tray_pose["x_m"]), float(tray_pose["y_m"]), float(tray_pose["z_m"])]
            )
            x_len = min(0.35, float(tray_size["length"]) / 2.0)
            y_len = min(0.35, float(tray_size["width"]) / 2.0)
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.006,
                tray_x_axis[0] * x_len, tray_x_axis[1] * x_len, 0.0,
                color="gold", linewidth=2.5, arrow_length_ratio=0.12,
            )
            ax.quiver(
                origin[0], origin[1], origin[2] + 0.008,
                tray_y_axis[0] * y_len, tray_y_axis[1] * y_len, 0.0,
                color="cyan", linewidth=2.5, arrow_length_ratio=0.12,
            )
            ax.text(
                float(tray_pose["x_m"]), float(tray_pose["y_m"]),
                float(tray_pose["z_m"]) + 0.02, "TRAY",
                color="saddlebrown", weight="bold",
            )

    for box in sorted(boxmap.boxes, key=lambda item: (item.layer, item.id)):
        color = {"standard": "orange", "frozen": "dodgerblue", "measured": "limegreen"}.get(
            box.source, "gray"
        )
        alpha = 0.28 if box.source == "standard" else 0.55
        vertices = _cuboid_vertices(
            (box.x, box.y, box.z), (box.length, box.width, box.height), box.yaw
        )
        _draw_cuboid(ax, vertices, color, alpha=alpha)
        ax.text(
            box.x, box.y, box.z + box.height / 2.0 + 0.02,
            f"L{box.layer}-{box.id} {box.box_type}",
            color=color, weight="bold", fontsize=8,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _preload_mplot3d() -> None:
    """从当前解释器的 site-packages 加载新版 mpl_toolkits.mplot3d 到 sys.modules。

    部分环境里系统 site-packages 残留旧版 mpl_toolkits（通过 pkg_resources
    namespace 机制劫持 import），导致 ``from mpl_toolkits.mplot3d import Axes3D``
    加载旧版并报 ``cannot import name 'docstring'``。这里在 import matplotlib 之前
    手动加载与当前 matplotlib 配套的 mpl_toolkits，规避该问题。
    """
    import importlib.util
    import os
    import sys
    import types

    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]

    tk_dir = None
    for p in sys.path:
        if not p:
            continue
        axes3d_path = os.path.join(p, "mpl_toolkits", "mplot3d", "axes3d.py")
        if not os.path.isfile(axes3d_path):
            continue
        try:
            with open(axes3d_path, encoding="utf-8") as fh:
                if "_docstring" in fh.read(4000):  # 新版 mpl_toolkits 使用 _docstring
                    tk_dir = os.path.join(p, "mpl_toolkits")
                    break
        except OSError:
            continue
    if tk_dir is None:
        return

    tk = types.ModuleType("mpl_toolkits")
    tk.__path__ = [tk_dir]
    sys.modules["mpl_toolkits"] = tk

    m3_dir = os.path.join(tk_dir, "mplot3d")
    spec = importlib.util.spec_from_file_location(
        "mpl_toolkits.mplot3d",
        os.path.join(m3_dir, "__init__.py"),
        submodule_search_locations=[m3_dir],
    )
    if spec is None or spec.loader is None:
        return
    m3 = importlib.util.module_from_spec(spec)
    sys.modules["mpl_toolkits.mplot3d"] = m3
    spec.loader.exec_module(m3)


class BoxMap3DViewer:
    """在独立线程中运行 matplotlib 3D 窗口，定时重绘最新 boxmap。"""

    def __init__(
        self,
        *,
        elev: float = 24.0,
        azim: float = -58.0,
        x_min: float = 0.5,
        x_max: float = 2.0,
        y_min: float = -1.5,
        y_max: float = 0.0,
        z_max: float = 1.8,
    ) -> None:
        self.elev = elev
        self.azim = azim
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_max = z_max
        self._lock = threading.Lock()
        self._boxmap: BoxMap | None = None
        self._dirty = False
        self._drawn_once = False
        self._thread: threading.Thread | None = None

    def update(self, boxmap: BoxMap) -> None:
        with self._lock:
            self._boxmap = boxmap
            self._dirty = True

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            print(
                f"WARNING: 3D 可视化启动失败（{type(exc).__name__}: {exc}）。"
                f"可能是当前环境 matplotlib/mpl_toolkits 不匹配。"
                f"可加 --no-show-3d 禁用 3D 窗口，仅保留 2D 画面。"
            )

    def _run_impl(self) -> None:
        import matplotlib

        _preload_mplot3d()
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # matplotlib 只在 import 时注册一次 "3d" projection；若当时 Axes3D 因
        # 系统旧版 mpl_toolkits 加载失败而未被注册，这里显式补注册。
        try:
            matplotlib.projections.projection_registry.register(Axes3D)
        except Exception:
            pass

        fig = plt.figure("BoxMap 3D", figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        def on_key(event: Any) -> None:
            if event.key in ("q", "Q", "escape"):
                plt.close(fig)

        def on_scroll(event: Any) -> None:
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
            fig.canvas.draw_idle()

        def refresh() -> None:
            with self._lock:
                boxmap = self._boxmap
                dirty = self._dirty
                self._dirty = False
            if dirty and boxmap is not None:
                # 保留用户拖动的旋转视角和滚轮缩放；仅首次使用默认值。
                if self._drawn_once:
                    elev, azim = ax.elev, ax.azim
                    x_min, x_max = ax.get_xlim3d()
                    y_min, y_max = ax.get_ylim3d()
                    z_min, z_max = ax.get_zlim3d()
                else:
                    elev, azim = self.elev, self.azim
                    x_min, x_max = self.x_min, self.x_max
                    y_min, y_max = self.y_min, self.y_max
                    z_min, z_max = 0.0, self.z_max
                    self._drawn_once = True
                _draw_3d_scene(
                    ax,
                    boxmap,
                    elev=elev,
                    azim=azim,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                )
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        refresh()
        timer = fig.canvas.new_timer(interval=500)
        timer.add_callback(refresh)
        timer.start()
        plt.show(block=True)
