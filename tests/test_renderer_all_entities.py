"""Smoke test: every draw_* method renders without crashing and produces
visible (non-background) pixels. Guards Task 9's per-entity conversion against
a missing sprite key or a method left half-converted."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pymunk
import pytest

from blueball.camera import FollowCamera
from blueball.render.core import RenderCore
from blueball.render.renderer import Renderer


@pytest.fixture(autouse=True)
def _pg():
    pygame.init()
    yield
    pygame.quit()


def _setup():
    core = RenderCore(pygame.Surface((1280, 720)))
    cam = FollowCamera(core.vw, core.vh)
    cam.scale = 1.0 / core.scale
    cam.position = (320, 180)          # center the view on world (320, 180)
    return core, Renderer(core, cam)


def _body(xy=(320, 180)):
    b = pymunk.Body(1, 1)
    b.position = xy
    return b


def _segment_space():
    space = pymunk.Space()
    space.add(pymunk.Segment(space.static_body, (300, 180), (340, 180), 3))
    return space


def _has_visible_pixels(surf) -> bool:
    return any(
        surf.get_at((x, y))[:3] != (0, 0, 0)
        for x in range(0, surf.get_width(), 3)
        for y in range(0, surf.get_height(), 3)
    )


def test_every_draw_method_renders():
    core, r = _setup()
    space = _segment_space()
    calls = {
        "background": lambda: r.draw_background((20, 30, 40)),
        "parallax": lambda: r.draw_parallax(r.camera),
        "ball": lambda: r.draw_ball(_body(), 1.0),
        "spike": lambda: r.draw_spike((320, 180), 32, 24, "up"),
        "collectible": lambda: r.draw_collectible((320, 180)),
        "ability_pickup": lambda: r.draw_ability_pickup((320, 180), 12, "double_jump"),
        "boost_pad_right": lambda: r.draw_boost_pad((320, 180), 48, 1.0),
        "boost_pad_left": lambda: r.draw_boost_pad((320, 180), 48, -1.0),
        "goal": lambda: r.draw_goal((320, 180), 32, 48),
        "patroller": lambda: r.draw_patroller(_body(), (24, 24), 1.0),
        "falling_hazard": lambda: r.draw_falling_hazard(_body(), 12, 1.0),
        "static_segments": lambda: r.draw_static_segments(space),
        "moving_platform": lambda: r.draw_moving_platform(_body(), 1.0, 80),
        "spring": lambda: r.draw_spring((320, 180), 32, 0.0),
        "checkpoint": lambda: r.draw_checkpoint((320, 180), 12, 0.0, False),
        "checkpoint_active": lambda: r.draw_checkpoint((320, 180), 12, 0.0, True),
        "crumbling": lambda: r.draw_crumbling_platform((320, 180), 1.0, 64, 0.3),
        "key": lambda: r.draw_key((320, 180), 10, 0),
        "door_closed": lambda: r.draw_door((320, 200), 64, False),
        "door_open": lambda: r.draw_door((320, 200), 64, True),
        "pushable_box": lambda: r.draw_pushable_box(_body(), 1.0, 32),
        "swinging_hazard": lambda: r.draw_swinging_hazard(
            _body((320, 150)), _body((320, 210)), 12, 1.0),
        "one_way": lambda: r.draw_one_way_platform((320, 180), 64),
        "charger_patrol": lambda: r.draw_charger(_body(), 1.0, "patrol"),
        "charger_charge": lambda: r.draw_charger(_body(), 1.0, "charge"),
        "lava": lambda: r.draw_lava(_body(), 1.0, 80, 40),
        "projectile": lambda: r.draw_projectile(_body(), 1.0, 10),
        "cannon_right": lambda: r.draw_cannon((320, 180), "right"),
        "cannon_left": lambda: r.draw_cannon((320, 180), "left"),
        "score": lambda: r.draw_score(123, 456),
    }
    for name, call in calls.items():
        core.surface.fill((0, 0, 0))
        call()
        assert _has_visible_pixels(core.surface), f"{name} produced no visible pixels"
