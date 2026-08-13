"""位姿精度评估指标。"""

from __future__ import annotations

import numpy as np


def xy_error(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1])


def z_error(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return np.abs(est[:, 2] - gt[:, 2])


def yaw_error_deg(est_yaw: np.ndarray, gt_yaw: np.ndarray) -> np.ndarray:
    est_yaw = np.asarray(est_yaw, dtype=float)
    gt_yaw = np.asarray(gt_yaw, dtype=float)
    d = np.abs(est_yaw - gt_yaw) % 180.0
    return np.minimum(d, 180.0 - d)


def summarize(err: np.ndarray) -> dict[str, float]:
    err = np.asarray(err, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(err**2))),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }

