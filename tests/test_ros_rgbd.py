from types import SimpleNamespace

import numpy as np

from box_perception.camera.ros_rgbd import decode_color_image, decode_depth_image


def _image(array, encoding, step=None):
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        step=int(step or array.strides[0]),
        is_bigendian=False,
        data=array.tobytes(),
    )


def test_decode_rgb_to_bgr():
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    bgr = decode_color_image(_image(rgb, "rgb8"))
    np.testing.assert_array_equal(bgr, [[[3, 2, 1], [6, 5, 4]]])


def test_decode_uint16_depth_to_metres_and_nan():
    raw = np.array([[1000, 0], [2500, 500]], dtype=np.uint16)
    depth = decode_depth_image(_image(raw, "16UC1"), 0.001)
    assert depth[0, 0] == 1.0
    assert np.isnan(depth[0, 1])
    assert depth[1, 0] == 2.5
