"""根据箱体底面高度和 XY footprint overlap 构建支撑图。"""

from __future__ import annotations

import math
from typing import Any


def _corners(box: Any) -> list[tuple[float, float]]:
    c, s = math.cos(math.radians(box.yaw)), math.sin(math.radians(box.yaw))
    ux, uy = c, s
    vx, vy = -s, c
    return [
        (
            box.x + sx * box.length * 0.5 * ux + sy * box.width * 0.5 * vx,
            box.y + sx * box.length * 0.5 * uy + sy * box.width * 0.5 * vy,
        )
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def _clip(
    subject: list[tuple[float, float]],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
) -> list[tuple[float, float]]:
    if not subject:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return (
            (edge_b[0] - edge_a[0]) * (point[1] - edge_a[1])
            - (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0])
        ) >= -1e-9

    def intersection(
        a: tuple[float, float], b: tuple[float, float]
    ) -> tuple[float, float]:
        dx1, dy1 = b[0] - a[0], b[1] - a[1]
        dx2, dy2 = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
        denominator = dx1 * dy2 - dy1 * dx2
        if abs(denominator) < 1e-12:
            return b
        t = ((edge_a[0] - a[0]) * dy2 - (edge_a[1] - a[1]) * dx2) / denominator
        return (a[0] + t * dx1, a[1] + t * dy1)

    result: list[tuple[float, float]] = []
    previous = subject[-1]
    previous_inside = inside(previous)
    for current in subject:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                result.append(intersection(previous, current))
            result.append(current)
        elif previous_inside:
            result.append(intersection(previous, current))
        previous, previous_inside = current, current_inside
    return result


def _area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) * 0.5


def _overlap_area(first: Any, second: Any) -> float:
    polygon = _corners(first)
    second_corners = _corners(second)
    for index in range(4):
        polygon = _clip(
            polygon,
            second_corners[index],
            second_corners[(index + 1) % 4],
        )
    return _area(polygon)


def build_support_graph(
    stack_map: Any,
    *,
    vertical_tolerance_m: float = 0.035,
    min_overlap_ratio: float = 0.20,
) -> dict[str, Any]:
    """返回支持关系、层号和每条边的几何信息。

    ``z`` 是箱体中心高度；候选支撑箱体必须位于当前箱体下方，且其顶面
    与当前箱体底面高度相近，XY footprint overlap 至少覆盖当前箱底面积的
    ``min_overlap_ratio``。
    """
    boxes = list(stack_map.boxes)
    supported_by = {str(box.id): [] for box in boxes}
    supports = {str(box.id): [] for box in boxes}
    edges: list[dict[str, float | int]] = []
    footprint = {
        box.id: max(float(box.length * box.width), 1e-9) for box in boxes
    }
    corners = {box.id: _corners(box) for box in boxes}

    def overlap(first: Any, second: Any) -> float:
        polygon = corners[first.id]
        second_corners = corners[second.id]
        for index in range(4):
            polygon = _clip(
                polygon,
                second_corners[index],
                second_corners[(index + 1) % 4],
            )
        return _area(polygon)

    for upper in boxes:
        bottom = upper.z - upper.height * 0.5
        candidates: list[tuple[float, float, Any, float]] = []
        for lower in boxes:
            if lower.id == upper.id:
                continue
            top = lower.z + lower.height * 0.5
            gap = bottom - top
            if gap < -vertical_tolerance_m or gap > vertical_tolerance_m:
                continue
            overlap_m2 = overlap(upper, lower)
            ratio = overlap_m2 / footprint[upper.id]
            if ratio >= min_overlap_ratio:
                candidates.append((abs(gap), -ratio, lower, overlap_m2))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _gap_abs, neg_ratio, lower, overlap_m2 in candidates:
            supported_by[str(upper.id)].append(lower.id)
            supports[str(lower.id)].append(upper.id)
            edges.append(
                {
                    "upper": upper.id,
                    "lower": lower.id,
                    "overlap_m2": overlap_m2,
                    "overlap_ratio": -neg_ratio,
                    "vertical_gap_m": bottom
                    - (lower.z + lower.height * 0.5),
                }
            )

    layers: dict[str, int] = {}

    def layer(box_id: int, visiting: set[int] | None = None) -> int:
        if str(box_id) in layers:
            return layers[str(box_id)]
        visiting = set() if visiting is None else visiting
        if box_id in visiting:
            return 1
        visiting.add(box_id)
        parents = supported_by.get(str(box_id), [])
        value = 1 if not parents else max(layer(parent, visiting) for parent in parents) + 1
        layers[str(box_id)] = value
        return value

    for box in boxes:
        layer(box.id)
    return {
        "supported_by": supported_by,
        "supports": supports,
        "layers": layers,
        "edges": edges,
    }
