import json

from box_perception.core.types import BoxState
from box_perception.stack.stack_map import StackMapManager


def _box(z: float) -> BoxState:
    return BoxState(
        id=-1,
        x=0.0,
        y=0.0,
        z=z,
        yaw=0.0,
        length=0.4,
        width=0.3,
        height=0.3,
        confirmed=True,
    )


def test_stack_map_assigns_layers_and_supports() -> None:
    manager = StackMapManager(map_sha256="map-a")
    lower_id = manager.add(_box(0.15))
    upper_id = manager.add(_box(0.45))

    lower = manager.get(lower_id)
    upper = manager.get(upper_id)
    assert lower is not None and upper is not None
    assert lower.layer == 1
    assert upper.layer == 2
    assert lower.supports == [upper_id]
    assert upper.supported_by == [lower_id]


def test_stack_map_round_trip(tmp_path) -> None:
    manager = StackMapManager(map_sha256="map-a")
    manager.add(_box(0.15))
    path = manager.save(tmp_path / "stack_map.json")

    loaded = StackMapManager.load(path, expected_map_sha256="map-a")
    assert len(loaded.map.boxes) == 1
    assert loaded.map.boxes[0].layer == 1
    assert json.loads(path.read_text(encoding="utf-8"))["box_count"] == 1
