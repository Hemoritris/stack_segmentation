import cv2
import numpy as np
import pytest

from box_perception.camera.rgb_capture import next_capture_index, save_rgb_png


def test_rgb_capture_continues_numbering_without_overwrite(tmp_path):
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    first = save_rgb_png(tmp_path, image, 0)
    second = save_rgb_png(tmp_path, image + 17, 1)
    assert first.name == "rgb_000000.png"
    assert second.name == "rgb_000001.png"
    assert next_capture_index(tmp_path) == 2
    np.testing.assert_array_equal(cv2.imread(str(second)), image + 17)
    with pytest.raises(FileExistsError):
        save_rgb_png(tmp_path, image, 1)
