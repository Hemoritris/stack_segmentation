import numpy as np

from box_perception.temporal.height_map import HeightMap


def test_median_aggregation():
    hm = HeightMap(x_range=(0, 0.2), y_range=(0, 0.2), grid_size_m=0.1, aggregation="median")
    pts = np.array(
        [
            [0.05, 0.05, 0.30],
            [0.08, 0.08, 0.32],
            [0.02, 0.02, 0.28],
        ]
    )
    h = hm.build(pts)
    assert not np.isnan(h[0, 0])
    assert abs(h[0, 0] - 0.30) < 1e-9
    assert np.isnan(h[1, 1])


def test_mean_aggregation():
    hm = HeightMap(x_range=(0, 0.2), y_range=(0, 0.2), grid_size_m=0.1, aggregation="mean")
    pts = np.array([[0.05, 0.05, 0.30], [0.08, 0.08, 0.32]])
    h = hm.build(pts)
    assert abs(h[0, 0] - 0.31) < 1e-9

