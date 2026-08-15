import numpy as np

from box_perception.geometry.tray_detection import detect_tray_from_depth


def test_detects_largest_elevated_tray_and_world_pose():
    height, width = 180, 240
    depth = np.full((height, width), 2.0, dtype=np.float32)
    depth[40:140, 50:190] = 1.7
    k = np.array([[200.0, 0.0, 120.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]])
    world_t_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result, artifacts = detect_tray_from_depth(
        depth,
        k,
        np.zeros(5),
        world_t_camera,
        frame="map",
        min_area_pixels=1000,
        max_plane_points=5000,
    )
    pose = result["pose_4dof"]
    size = result["measured_size_m"]
    assert result["frame"] == "map"
    assert abs(pose["x_m"]) < 0.02
    assert abs(pose["y_m"]) < 0.02
    assert abs(pose["z_m"] - 0.30) < 0.005
    assert abs(pose["yaw_deg"]) < 1.0
    assert abs(size["length"] - 1.19) < 0.03
    assert abs(size["width"] - 0.84) < 0.03
    assert abs(size["top_height_above_ground"] - 0.30) < 0.005
    assert artifacts.image_mask.sum() > 13000
    assert len(artifacts.top_points_world) > 10000
