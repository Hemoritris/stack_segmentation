"""Filesystem helpers for lossless interactive RGB collection."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def next_capture_index(output_dir: str | Path, prefix: str = "rgb") -> int:
    output = Path(output_dir)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})\.png$")
    indices = []
    if output.is_dir():
        for path in output.iterdir():
            match = pattern.match(path.name)
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def save_rgb_png(
    output_dir: str | Path,
    image_bgr: np.ndarray,
    index: int,
    prefix: str = "rgb",
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be an HxWx3 uint8 array")
    if index < 0:
        raise ValueError("capture index must be non-negative")
    path = output / f"{prefix}_{index:06d}.png"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing capture: {path}")
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise RuntimeError(f"failed to save RGB image: {path}")
    return path
