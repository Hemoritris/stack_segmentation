"""世界高度图构建（M2）。

职责：把 world 点云投影到固定 XY 栅格，得到 H(x, y)=z，用于时序差分。
"""

from __future__ import annotations

import numpy as np


class HeightMap:
    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        grid_size_m: float = 0.005,
        aggregation: str = "median",
    ):
        if aggregation not in ("median", "mean"):
            raise ValueError("aggregation 只支持 median 或 mean")
        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.grid_size_m = float(grid_size_m)
        self.aggregation = aggregation
        self.nx = max(1, int(round((self.x_max - self.x_min) / self.grid_size_m)))
        self.ny = max(1, int(round((self.y_max - self.y_min) / self.grid_size_m)))

    def build(self, points_world) -> np.ndarray:
        """把 (N, 3) 世界点云聚合为高度图，空栅格填 NaN。"""
        pts = np.asarray(points_world, dtype=np.float64)
        height = np.full((self.ny, self.nx), np.nan, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            return height

        ix = np.floor((pts[:, 0] - self.x_min) / self.grid_size_m).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.y_min) / self.grid_size_m).astype(np.int64)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix, iy, z = ix[valid], iy[valid], pts[valid, 2]
        if ix.size == 0:
            return height

        cell = iy * self.nx + ix
        counts = np.bincount(cell, minlength=self.nx * self.ny)
        vals = np.full(self.nx * self.ny, np.nan, dtype=np.float64)
        has = counts > 0
        if self.aggregation == "mean":
            sums = np.bincount(cell, weights=z, minlength=self.nx * self.ny)
            vals[has] = sums[has] / counts[has]
        else:  # median
            order = np.lexsort((z, cell))
            cell_sorted = cell[order]
            z_sorted = z[order]
            uniq, first, cnt = np.unique(cell_sorted, return_index=True, return_counts=True)
            lo = first + (cnt - 1) // 2
            hi = first + cnt // 2
            vals[uniq] = (z_sorted[lo] + z_sorted[hi]) / 2.0

        inds = np.flatnonzero(has)
        height[inds // self.nx, inds % self.nx] = vals[inds]
        return height
