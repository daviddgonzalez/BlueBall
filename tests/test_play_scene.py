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
