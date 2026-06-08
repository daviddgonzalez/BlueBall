"""Full-scene smoke render + theme switch.

These run against a real (dummy-driver) SDL display so the whole render
pipeline -- RenderCore, Renderer, parallax, particles, scoreboard, present --
executes end to end. We assert only that one full frame (update + draw) does
not crash and that the window surface stays 1280x720; the per-primitive
correctness lives in the targeted renderer tests.
"""

import dataclasses
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

# Minimal streaming Infinite-Run level_data. sampler_seed enables streaming;
# PlayScene requires exactly one of level_path/level_data, so we pass this dict.
_INFINITE_LEVEL_DATA = {
    "name": "Infinite Run (smoke)",
    "background": "#202028",
    "ground": "#666c70",
    "spawn": [80, 540],
    "chunks": [],
}


@pytest.fixture(autouse=True)
def _pg():
    pygame.init()
    yield
    pygame.quit()


def _make_streaming_scene(screen, sampler_seed):
    from blueball.scenes.play import PlayScene

    # dict() so each scene gets its own copy; PlayScene reads level_data in place.
    return PlayScene(
        screen,
        level_data=dict(_INFINITE_LEVEL_DATA),
        sampler_seed=sampler_seed,
    )


def test_playscene_renders_one_frame():
    screen = pygame.display.set_mode((1280, 720))
    scene = _make_streaming_scene(screen, sampler_seed=1)
    scene.update(1 / 60)
    scene.draw()
    assert screen.get_size() == (1280, 720)


def test_playscene_renders_under_switched_theme():
    """Render a frame under the pixel theme, then under a dummy theme that is a
    dataclasses.replace of the pixel theme, switching config.ACTIVE_THEME via
    monkeypatch. Both must render without crashing. The dummy is removed from
    the registry afterward so it can't leak into other tests."""
    from blueball import config
    from blueball.render import theme as theme_mod

    screen = pygame.display.set_mode((1280, 720))

    # 1) Baseline render under the real pixel theme.
    scene = _make_streaming_scene(screen, sampler_seed=2)
    scene.update(1 / 60)
    scene.draw()
    assert screen.get_size() == (1280, 720)

    # 2) Register a dummy theme cloned from pixel, switch to it, render again.
    dummy_name = "__smoke_dummy_theme__"
    pixel_theme = theme_mod.get_theme("pixel")
    dummy_theme = dataclasses.replace(pixel_theme)
    theme_mod.register_theme(dummy_name, dummy_theme)
    prev_active = config.ACTIVE_THEME
    try:
        config.ACTIVE_THEME = dummy_name
        assert theme_mod.get_active_theme() is dummy_theme
        scene2 = _make_streaming_scene(screen, sampler_seed=3)
        scene2.update(1 / 60)
        scene2.draw()
        assert screen.get_size() == (1280, 720)
    finally:
        # Clean up the registry + active-theme override regardless of outcome.
        config.ACTIVE_THEME = prev_active
        theme_mod._REGISTRY.pop(dummy_name, None)
