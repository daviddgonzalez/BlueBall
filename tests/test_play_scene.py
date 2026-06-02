"""Tests for PlayScene's save-file-driven ability loading."""

import importlib
import json
from pathlib import Path

import pygame
import pytest

import blueball
from blueball.abilities import Ability
from blueball.scenes.play import PlayScene


@pytest.fixture
def headless_pygame():
    # Use a dummy SDL driver so pygame can run in CI without a display.
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.display.quit()


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    """Redirect BLUEBALL_SAVE_PATH at a tmp file. The save module captures
    SAVE_PATH at import time, so we both reload it and overwrite the attribute
    as belt-and-suspenders against future refactors. scenes/play.py does
    `from .. import save` and looks up `save.load()` / `save.SAVE_PATH` through
    the module reference, so reloading save_mod alone is sufficient — no need
    to reload player or play_mod."""
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    import blueball.save as save_mod
    importlib.reload(save_mod)
    monkeypatch.setattr(save_mod, "SAVE_PATH", save_path)
    return save_path, save_mod


def _level_path() -> Path:
    return Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"


def test_play_scene_starts_with_no_abilities_when_save_missing(headless_pygame, tmp_save):
    _path, _save_mod = tmp_save
    scene = PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == set()


def test_play_scene_loads_unlocked_abilities_from_save(headless_pygame, tmp_save):
    path, _save_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["double_jump"]}))
    scene = PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == {Ability.DOUBLE_JUMP}


def test_play_scene_ignores_unknown_abilities_in_save(headless_pygame, tmp_save):
    path, _save_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["frobnicate", "double_jump"]}))
    scene = PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == {Ability.DOUBLE_JUMP}


def test_play_scene_persists_unlocks_on_level_complete(headless_pygame, tmp_save):
    """Unlocks gained during a successful run are written to disk when the
    player reaches the goal."""
    path, save_mod = tmp_save
    scene = PlayScene(headless_pygame, _level_path())
    # Player picks up an ability mid-run (in-memory only).
    scene.player.unlock(Ability.DOUBLE_JUMP)
    assert not path.exists()  # not persisted yet
    # World marks the level complete (as it would on goal contact).
    scene.world.complete_level()
    # Drain pygame's event queue so the QUIT post inside update() doesn't
    # leak across tests.
    pygame.event.clear()
    scene.update(frame_dt=1 / 60)
    assert path.exists()
    assert save_mod.load() == {"double_jump"}


def test_play_scene_does_not_persist_unlocks_on_death(headless_pygame, tmp_save):
    """If the player dies mid-run, in-memory unlocks are dropped on respawn
    and the save file is unchanged."""
    path, save_mod = tmp_save
    scene = PlayScene(headless_pygame, _level_path())
    scene.player.unlock(Ability.DOUBLE_JUMP)
    assert Ability.DOUBLE_JUMP in scene.player.abilities
    # Trigger death + reset
    scene.player.die()
    scene.update(frame_dt=1 / 60)
    # After respawn, the new player should have NO abilities (save is empty).
    assert scene.player.abilities == set()
    assert not path.exists()


def test_play_scene_accepts_level_data(monkeypatch, tmp_path):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    data = {
        "name": "Test", "background": "#000000", "ground": "#111111",
        "spawn": [80, 540],
        "chunks": [{"type": "flat", "width_tiles": 3}, {"type": "goal"}],
    }
    scene = PlayScene(screen, level_data=data, sampler_seed=12345)
    assert scene.sampler_seed == 12345
    assert scene.level_data is data
    pygame.display.quit()


def test_play_scene_level_complete_returns_menu_scene(headless_pygame, tmp_save):
    """After level_complete triggers, handle_events([]) must return a MenuScene."""
    from blueball.scenes.menu import MenuScene
    _path, save_mod = tmp_save
    scene = PlayScene(headless_pygame, _level_path())
    scene.world.complete_level()
    pygame.event.clear()
    scene.update(frame_dt=1 / 60)
    result = scene.handle_events([])
    assert isinstance(result, MenuScene)


def test_play_scene_esc_returns_menu_scene(monkeypatch, tmp_path):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    from blueball.scenes.menu import MenuScene
    scene = PlayScene(screen, level_path=_level_path())
    result = scene.handle_events([pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})])
    assert isinstance(result, MenuScene)
    pygame.display.quit()


def test_play_scene_streaming_first_chunk_has_spawn_ground(headless_pygame, tmp_save):
    """The first emitted built-chunks entry must include a static ground
    segment at y=GROUND_Y. The sampler can pick floating chunks (e.g.
    'platform') as its first emission; PlayScene compensates by planting
    a guaranteed Flat at x=0 ahead of the sampler.
    """
    import pymunk
    from blueball.levels.chunks.flat import GROUND_Y

    data = {
        "name": "Infinite",
        "background": "#202028",
        "ground": "#666c70",
        "spawn": [80, 540],
        "chunks": [],
    }
    # Seed 0 reliably picks a floating 'platform' as the sampler's first
    # emission — without the spawn floor it would leave the player with
    # no ground beneath them.
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=0)
    assert scene._built_chunks, "expected at least one materialized chunk"
    first = scene._built_chunks[0]
    assert first["x_start"] == 0.0
    ground_segments = [
        s for s in first["shapes"]
        if isinstance(s, pymunk.Segment)
        and s.a.y == GROUND_Y and s.b.y == GROUND_Y
    ]
    assert ground_segments, "first chunk must include a ground segment at GROUND_Y"


def test_play_scene_streaming_rerandomizes_seed_on_death(headless_pygame, tmp_save):
    """In Infinite Run, dying picks a NEW sampler seed (fresh layout) rather
    than replaying the same deterministic run."""
    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=12345)
    assert scene.sampler_seed == 12345
    scene.player.dead = True
    pygame.event.clear()
    scene.update(frame_dt=1 / 60)
    assert scene.sampler_seed != 12345          # re-randomized
    assert scene._last_respawn_xy is None       # no checkpoint respawn
    assert not scene.player.dead                # fresh player after reset


def test_play_scene_streaming_has_no_checkpoints(headless_pygame, tmp_save):
    """The streaming sampler is built with checkpoints disabled."""
    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=7)
    assert scene._sampler.emit_checkpoints is False


def test_play_scene_streaming_carries_ground_height(headless_pygame, tmp_save):
    """After a stairs_up the running ground seam rises, and a following flat is
    built at that raised height — surfaces connect across the seam."""
    import pymunk
    from blueball.levels.chunks.flat import GROUND_Y, Flat
    from blueball.levels.chunks.stairs import StairsUp

    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=1)
    base0 = scene._base_y
    scene._materialize_chunk(StairsUp(steps=3, step_height=40, rounded=True))
    assert scene._base_y == base0 - 120  # seam climbed 120px (up = -y)
    scene._materialize_chunk(Flat(width_tiles=2))
    flat_chunk = scene._built_chunks[-1]
    ground = [s for s in flat_chunk["shapes"] if isinstance(s, pymunk.Segment)][0]
    assert ground.a.y == base0 - 120  # flat sits at the raised seam


def test_play_scene_streaming_ground_stays_in_band(headless_pygame, tmp_save):
    """Over a long stream the running ground height never goes underground or
    above the elevation cap (stairs are biased to stay in band)."""
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball.scenes.play import _MAX_GROUND_ELEV

    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=3)
    for _ in range(400):
        scene._maintain_streaming(scene._build_x + 50)
        assert GROUND_Y - _MAX_GROUND_ELEV - 1e-6 <= scene._base_y <= GROUND_Y + 1e-6


def test_play_scene_streaming_builds_ahead(headless_pygame, tmp_save):
    """Infinite Run uses sampler_seed and streams chunks. After construction,
    only the initial buffer is built (NOT all 500). Advancing the player
    triggers more builds; sliding past chunks culls them."""
    data = {
        "name": "Infinite",
        "background": "#202028",
        "ground": "#666c70",
        "spawn": [80, 540],
        "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=42)
    assert scene._streaming is True
    initial_count = len(scene._built_chunks)
    assert 0 < initial_count <= 20  # only a few chunks, not 500

    # Jump the player far ahead — streaming maintainer should build more
    initial_build_x = scene._build_x
    scene.player.body.position = (scene._build_x + 100, 540)
    scene._maintain_streaming(scene.player.body.position.x)
    assert scene._build_x > initial_build_x  # more chunks built ahead
    assert len(scene._built_chunks) > initial_count

    # Jump the player way ahead — chunks behind should cull
    far_x = scene._build_x + 5000
    scene.player.body.position = (far_x, 540)
    scene._maintain_streaming(far_x)
    # All chunks fully behind player by more than LOAD_BEHIND should be gone
    for info in scene._built_chunks:
        # No chunk should end more than LOAD_BEHIND px behind the player
        assert info["x_end"] >= far_x - 800 - 1


def test_play_scene_culls_box_lava_gap_entities(headless_pygame, tmp_save):
    """A chunk-spawned static Lava + PushableBox must be diff-tracked and culled
    once the player slides far past the chunk."""
    from blueball.entities.lava import Lava
    from blueball.entities.pushable_box import PushableBox
    from blueball.levels.chunks.box_lava_gap import BoxLavaGap

    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=42)
    scene._materialize_chunk(BoxLavaGap(pit_tiles=6))
    rec = scene._built_chunks[-1]
    my_lava = next(e for e in rec["entities"] if isinstance(e, Lava))
    my_box = next(e for e in rec["entities"] if isinstance(e, PushableBox))
    assert my_lava in scene.world.entities
    assert my_box in scene.world.entities

    far = rec["x_end"] + 5000  # well past _LOAD_BEHIND (800 px)
    scene.player.body.position = (far, 540)
    scene._maintain_streaming(far)

    assert my_lava not in scene.world.entities
    assert my_box not in scene.world.entities
    assert my_lava.body not in scene.world.space.bodies
    assert my_box.body not in scene.world.space.bodies
    assert my_lava.shape not in scene.world.space.shapes
