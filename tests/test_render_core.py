import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame
import pytest
from blueball.render.core import RenderCore


@pytest.fixture(autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


def test_virtual_surface_is_half_resolution():
    window = pygame.Surface((1280, 720))
    core = RenderCore(window, pixel_scale=2)
    assert core.surface.get_size() == (640, 360)
    assert (core.vw, core.vh) == (640, 360)


def test_present_upscales_onto_window():
    window = pygame.Surface((1280, 720))
    core = RenderCore(window, pixel_scale=2)
    core.surface.fill((10, 20, 30))
    core.present(flip=False)  # flip=False so the test needs no real display
    assert window.get_at((0, 0))[:3] == (10, 20, 30)
    assert window.get_at((1279, 719))[:3] == (10, 20, 30)


def test_rejects_indivisible_window():
    window = pygame.Surface((1281, 720))
    with pytest.raises(ValueError):
        RenderCore(window, pixel_scale=2)


def test_rejects_scale_below_one():
    window = pygame.Surface((1280, 720))
    with pytest.raises(ValueError):
        RenderCore(window, pixel_scale=0)


def test_present_identity_scale():
    window = pygame.Surface((640, 360))
    core = RenderCore(window, pixel_scale=1)
    assert core.surface.get_size() == (640, 360)
    core.surface.fill((99, 88, 77))
    core.present(flip=False)
    assert window.get_at((0, 0))[:3] == (99, 88, 77)


def test_shake_decays():
    window = pygame.Surface((1280, 720))
    core = RenderCore(window, pixel_scale=2)
    core.add_shake(10.0)
    for _ in range(200):
        core.update(0.1)
    assert abs(core.shake_offset[0]) < 1.0 and abs(core.shake_offset[1]) < 1.0
