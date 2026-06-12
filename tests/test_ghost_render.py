import os

import numpy as np
import pygame
import pytest


@pytest.fixture
def renderer():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    screen = pygame.display.set_mode((1280, 720))
    from blueball.camera import FollowCamera
    from blueball.render.core import RenderCore
    from blueball.render.renderer import Renderer
    core = RenderCore(screen)
    cam = FollowCamera(core.vw, core.vh)
    cam.scale = 1.0 / core.scale
    yield Renderer(core, cam)
    pygame.display.quit()


def _avg_alpha(surf):
    a = pygame.surfarray.array_alpha(surf)
    return float(np.mean(a))


def test_draw_ghost_ball_smoke_and_translucent(renderer):
    renderer.draw_ghost_ball((100.0, 100.0), deg=0.0)
    ghost = renderer._ghost_ball
    assert ghost is not None
    base = renderer._theme().sprites["ball"].frame(0, renderer._theme().palette)
    assert _avg_alpha(ghost) < _avg_alpha(base)
    renderer.draw_ghost_ball((150.0, 120.0), deg=45.0)
