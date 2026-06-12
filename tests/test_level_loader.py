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


def test_load_vertical_climb_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "vertical_climb.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Vertical Climb"


def test_vertical_climb_declares_double_jump():
    # The vertical-climb specialist (and the generalist) need double-jump to
    # clear the stepped column; the curriculum trainer grants only what the
    # level declares, so it must be in the JSON (like maze.json).
    from pathlib import Path
    from blueball.abilities import Ability
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "vertical_climb.json"
    w = World()
    meta = load_level(path, w)
    assert Ability.DOUBLE_JUMP in meta.starting_abilities
    assert meta.total_width > 0


def test_load_speed_run_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "speed_run.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Speed Run"


def test_load_maze_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "maze.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Maze"


def test_load_level_accepts_dict():
    data = {
        "name": "Test",
        "background": "#000000",
        "ground": "#111111",
        "spawn": [80, 540],
        "chunks": [
            {"type": "flat", "width_tiles": 3},
            {"type": "goal"},
        ],
    }
    w = World()
    meta = load_level(data, w)
    assert meta.name == "Test"
    assert meta.total_width > 0


def test_starting_abilities_defaults_empty(tmp_path):
    level = {
        "name": "NoAbilities", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "lvl.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.starting_abilities == frozenset()


def test_starting_abilities_parsed_from_level(tmp_path):
    from blueball.abilities import Ability
    level = {
        "name": "WithDJ", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "starting_abilities": ["double_jump"],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "dj.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.starting_abilities == frozenset({Ability.DOUBLE_JUMP})


def test_maze_declares_double_jump_starting_ability():
    from pathlib import Path
    import blueball
    from blueball.abilities import Ability
    maze = Path(blueball.__file__).parent / "levels" / "maze.json"
    meta = load_level(maze, World())
    assert Ability.DOUBLE_JUMP in meta.starting_abilities


def test_unknown_starting_ability_raises(tmp_path):
    level = {
        "name": "BadAbility", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "starting_abilities": ["triple_jump"],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(level))
    with pytest.raises(ValueError):
        load_level(path, World())


def test_curriculum_spawns_parsed(tmp_path):
    import json
    from blueball.levels.loader import load_level
    from blueball.world import World
    level = {
        "name": "T", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0],
        "curriculum_spawns": [{"x": 100, "y": 50, "keys": [0], "label": "a"}],
        "chunks": [{"type": "flat", "width_tiles": 2}],
    }
    path = tmp_path / "l.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.curriculum_spawns == ({"x": 100, "y": 50, "keys": [0], "label": "a"},)


def test_curriculum_spawns_absent_is_empty(tmp_path):
    import json
    from blueball.levels.loader import load_level
    from blueball.world import World
    level = {
        "name": "T", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "chunks": [{"type": "flat", "width_tiles": 2}],
    }
    path = tmp_path / "l.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.curriculum_spawns == ()
