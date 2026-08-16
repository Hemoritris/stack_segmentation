"""垛堆模型：层高/编号/标准位置，以及 boxmap 的增量更新与保存。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .camera import pose_ref_to_world
from .types import BoxMap, BoxState, CandidateBox


# ---------------------------------------------------------------------------
# 层高与标准编号布局
# ---------------------------------------------------------------------------

def layer_top_zs(tray_z: float) -> list[float]:
    """返回 4 层的标准顶面高度（世界 z）。"""
    ha = config.BOX_TYPES["A"]["height"]
    hb = config.BOX_TYPES["B"]["height"]
    return [
        tray_z + ha,
        tray_z + 2 * ha,
        tray_z + 2 * ha + hb,
        tray_z + 2 * ha + 2 * hb,
    ]


def layer_support_z(layer: int, tray_z: float) -> float:
    """第 layer 层的支撑面高度（世界 z）。"""
    tops = layer_top_zs(tray_z)
    if layer == 1:
        return float(tray_z)
    return float(tops[layer - 2])


def layer_from_top_z(top_z: float, tray_z: float) -> tuple[int, float]:
    """观测顶面高度 → 最近的标准层号与偏差。"""
    tops = layer_top_zs(tray_z)
    diffs = [abs(float(top_z) - t) for t in tops]
    layer = int(np.argmin(diffs)) + 1
    return layer, float(top_z) - tops[layer - 1]


def slot_id_from_position(x_ref: float, y_ref: float, length: float, width: float) -> int:
    """托盘局部 XY → 最近的标准槽位编号 1~6。

    布局：3 列沿托盘 +X（箱子短轴 width），2 行沿 +Y（箱子长轴 length）。
    编号：-Y 侧一行 1,2,3（自 -X 到 +X），+Y 侧一行 4,5,6。
    """
    col = int(round(x_ref / max(width, 1e-9))) + 1
    row = int(round(y_ref / max(length, 1e-9) + 0.5))
    col = min(max(col, 0), 2)
    row = min(max(row, 0), 1)
    return row * 3 + col + 1


def standard_slot_world_pose(
    layer: int, slot_id: int, tray_pose: dict[str, Any]
) -> tuple[float, float, float, float, float, float, float]:
    """标准槽位的世界位姿 (x, y, z, yaw, length, width, height)。"""
    box_type = config.LAYER_BOX_TYPES[layer]
    assert box_type is not None
    length = config.BOX_TYPES[box_type]["length"]
    width = config.BOX_TYPES[box_type]["width"]
    height = config.BOX_TYPES[box_type]["height"]
    col = (slot_id - 1) % 3
    row = (slot_id - 1) // 3
    x_ref = (col - 1) * width
    y_ref = (row - 0.5) * length
    x_world, y_world, yaw_world = pose_ref_to_world(x_ref, y_ref, 90.0, tray_pose)
    z_world = layer_support_z(layer, float(tray_pose["z_m"])) + height / 2.0
    return x_world, y_world, z_world, yaw_world, length, width, height


# ---------------------------------------------------------------------------
# boxmap 更新与保存
# ---------------------------------------------------------------------------

def update_boxmap(
    boxmap: BoxMap,
    accepted: list[CandidateBox],
    tray: dict[str, Any],
    map_sha256: str | None,
) -> BoxMap:
    """增量更新 boxmap。

    冻结规则：识别到更高一层的箱子时，冻结前一层。
    - 最高可见层（活动层）每帧用实测实时更新（source="measured"）；
    - 低于活动层的层一旦出现更高层，即冻结（source="frozen"），位置锁定为
      该层作为活动层时最后一次的实测值，之后不再随帧更新；
    - 中途打开时看不到的层（从未出现过）按标准位置补全 6 箱（source="standard"）。
    """
    tray_pose = tray["pose_4dof"]
    measured_by_layer: dict[int, list[CandidateBox]] = {}
    for c in accepted:
        measured_by_layer.setdefault(c.layer, []).append(c)
    active_layer = max((c.layer for c in accepted), default=0)

    def _box(c: CandidateBox, source: str) -> BoxState:
        return BoxState(
            id=c.slot_id,
            layer=c.layer,
            box_type=c.box_type,
            x=c.x,
            y=c.y,
            z=c.z,
            yaw=c.yaw,
            length=c.length,
            width=c.width,
            height=c.height,
            source=source,
            timestamp=c.yolo_confidence,
        )

    new_boxes: list[BoxState] = []
    for layer in range(1, active_layer + 1):
        cur = measured_by_layer.get(layer, [])
        existing = {b.id: b for b in boxmap.boxes if b.layer == layer}

        if layer == active_layer:
            # 活动层：累积更新 + 短暂丢失保持。
            # 当前帧识别到的槽位更新为实测；未识别到的槽位保留上一次实测，
            # 连续未识别超过阈值才移除，避免上层箱子/机械臂短暂遮挡导致误删。
            merged: dict[int, BoxState] = {}
            for c in cur:
                merged[c.slot_id] = _box(c, "measured")
            for b in existing.values():
                if b.id in merged:
                    continue
                b.missed += 1
                if b.missed <= config.MISSED_FRAMES_BEFORE_REMOVE:
                    merged[b.id] = b
            new_boxes.extend(merged.values())
            continue

        # 冻结层（已有更高层）：保留已有位置，忽略本帧该层实测。
        if existing:
            for b in existing.values():
                if b.source != "standard":
                    b.source = "frozen"
            new_boxes.extend(existing.values())
        else:
            # 从未出现过（例如中途打开）：按标准位置补全 6 箱。
            box_type = config.LAYER_BOX_TYPES[layer]
            for slot_id in range(1, config.BOXES_PER_LAYER + 1):
                x, y, z, yaw, length, width, height = standard_slot_world_pose(
                    layer, slot_id, tray_pose
                )
                new_boxes.append(
                    BoxState(
                        id=slot_id,
                        layer=layer,
                        box_type=box_type,
                        x=x,
                        y=y,
                        z=z,
                        yaw=yaw,
                        length=length,
                        width=width,
                        height=height,
                        source="standard",
                    )
                )

    boxmap.boxes = new_boxes
    boxmap.active_layer = active_layer
    boxmap.map_sha256 = map_sha256 or boxmap.map_sha256
    boxmap.tray_reference = tray
    return boxmap


def save_boxmap(path: Path, boxmap: BoxMap) -> None:
    """把 boxmap 序列化保存为 JSON（原子写入）。"""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "purpose": "stack_box_mapper",
        "world_frame": boxmap.world_frame,
        "map_sha256": boxmap.map_sha256,
        "active_layer": boxmap.active_layer,
        "tray_reference": boxmap.tray_reference,
        "boxes": [asdict(b) for b in boxmap.boxes],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
