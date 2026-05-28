"""Tests for Infinite Run scoring and best-score persistence."""

import importlib
import json

import pygame
import pytest


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    import blueball.save as save_mod
    importlib.reload(save_mod)
    monkeypatch.setattr(save_mod, "SAVE_PATH", save_path)
    return save_path, save_mod


@pytest.fixture
def headless_screen():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.display.quit()


def test_best_score_round_trip(tmp_save):
    _path, save_mod = tmp_save
    assert save_mod.get_best_score() == 0
    save_mod.set_best_score(1500)
    assert save_mod.get_best_score() == 1500


def test_best_score_only_increases(tmp_save):
    _path, save_mod = tmp_save
    save_mod.set_best_score(2000)
    save_mod.set_best_score(900)  # lower — ignored
    assert save_mod.get_best_score() == 2000


def test_best_score_and_abilities_coexist(tmp_save):
    """Persisting one field must not clobber the other."""
    _path, save_mod = tmp_save
    save_mod.set_best_score(1234)
    save_mod.add_ability("double_jump")
    assert save_mod.get_best_score() == 1234
    assert save_mod.load() == {"double_jump"}
    save_mod.set_best_score(5678)
    assert save_mod.load() == {"double_jump"}  # abilities preserved
    assert save_mod.get_best_score() == 5678


def test_score_is_ten_times_x(tmp_save, headless_screen):
    from blueball.scenes.play import PlayScene
    data = {"name": "Infinite", "background": "#202028", "ground": "#666c70",
            "spawn": [80, 540], "chunks": []}
    scene = PlayScene(headless_screen, level_data=data, sampler_seed=1)
    scene.player.body.position = (250, 540)
    pygame.event.clear()
    scene.update(1 / 60)
    assert scene._score == int(10 * 250)


def test_score_tracks_furthest_x_not_current(tmp_save, headless_screen):
    from blueball.scenes.play import PlayScene
    data = {"name": "Infinite", "background": "#202028", "ground": "#666c70",
            "spawn": [80, 540], "chunks": []}
    scene = PlayScene(headless_screen, level_data=data, sampler_seed=1)
    pygame.event.clear()
    scene.player.body.position = (400, 540)
    scene.update(1 / 60)
    scene.player.body.position = (300, 540)  # moved back
    scene.update(1 / 60)
    assert scene._score == 4000  # holds the max, doesn't drop


def test_best_score_persisted_on_death(tmp_save, headless_screen):
    _path, save_mod = tmp_save
    from blueball.scenes.play import PlayScene
    data = {"name": "Infinite", "background": "#202028", "ground": "#666c70",
            "spawn": [80, 540], "chunks": []}
    scene = PlayScene(headless_screen, level_data=data, sampler_seed=1)
    pygame.event.clear()
    scene.player.body.position = (600, 540)
    scene.update(1 / 60)
    assert scene._score == 6000
    scene.player.dead = True
    scene.update(1 / 60)  # death banks the score + re-randomizes
    assert save_mod.get_best_score() == 6000
    assert scene._score == 0  # new run reset


def test_non_streaming_level_does_not_score(tmp_save, headless_screen):
    from pathlib import Path
    import blueball
    from blueball.scenes.play import PlayScene
    p = Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"
    scene = PlayScene(headless_screen, level_path=p)
    pygame.event.clear()
    scene.player.body.position = (500, 540)
    scene.update(1 / 60)
    assert scene._score == 0  # hand levels aren't scored
