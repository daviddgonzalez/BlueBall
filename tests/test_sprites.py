import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame, pytest
from blueball.render.sprites import SpriteDef


@pytest.fixture(autouse=True)
def _pg():
    pygame.init(); yield; pygame.quit()


PALETTE = {"ball": (10, 20, 30), "ball_hi": (200, 210, 220)}


def test_bake_dimensions_and_colors():
    sd = SpriteDef(grid=["bB", ".b"], palette_key="ball")
    surf = sd.bake(PALETTE)
    assert surf.get_size() == (2, 2)
    assert surf.get_at((0, 0))[:3] == (10, 20, 30)     # 'b'
    assert surf.get_at((1, 0))[:3] == (200, 210, 220)  # 'B' -> ball_hi
    assert surf.get_at((0, 1))[3] == 0                 # '.' transparent


def test_bake_is_cached():
    sd = SpriteDef(grid=["b"], palette_key="ball")
    assert sd.bake(PALETTE) is sd.bake(PALETTE)


def test_multiframe():
    sd = SpriteDef(grid=[["b"], ["B"]], palette_key="ball", frames=2)
    assert sd.frame(0, PALETTE).get_at((0, 0))[:3] == (10, 20, 30)
    assert sd.frame(1, PALETTE).get_at((0, 0))[:3] == (200, 210, 220)
