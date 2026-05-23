import json
import pytest

from blueball.world import World
from blueball.levels.loader import load_level


def test_load_simple_level_inline(tmp_path):
    level = {
        "name": "Test",
        "background": "#7ec7ff",
        "ground": "#3b8a4a",
        "spawn": [50, 300],
        "chunks": [
            {"type": "flat", "width_tiles": 4},
            {"type": "goal"},
        ],
    }
    path = tmp_path / "level.json"
    path.write_text(json.dumps(level))

    w = World()
    meta = load_level(path, w)
    assert meta.name == "Test"
    assert meta.spawn == (50, 300)
    assert meta.background == (126, 199, 255)
    assert meta.total_width > 0


def test_unknown_chunk_raises(tmp_path):
    level = {
        "name": "Bad",
        "background": "#000000",
        "ground": "#000000",
        "spawn": [0, 0],
        "chunks": [{"type": "not_a_real_chunk"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(level))

    w = World()
    with pytest.raises(ValueError) as ei:
        load_level(path, w)
    assert "not_a_real_chunk" in str(ei.value)
