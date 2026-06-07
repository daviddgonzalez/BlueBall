import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame, pytest
from blueball.render.themes import pixel


@pytest.fixture(autouse=True)
def _pg():
    pygame.init(); yield; pygame.quit()


def test_core_sprites_present_and_bakeable():
    theme = pixel.build()
    for key in ("ball", "spike", "collectible", "goal", "coin"):
        assert key in theme.sprites, key
        surf = theme.sprites[key].bake(theme.palette)
        assert surf.get_width() > 0 and surf.get_height() > 0
