"""感知评估：新箱识别率、漏检 / 误检 / 粘连等指标。"""

from __future__ import annotations


def detection_rate(matches: list[bool]) -> float:
    return sum(matches) / len(matches) if matches else 0.0

