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


def test_uppercase_grid_chars_have_hi_palette_keys():
    """Any sprite cell using an uppercase (highlight) char must have a matching
    `<key>_hi` color, or the highlight silently renders as the base color.
    Guards every current and future pixel sprite against that mistake."""
    theme = pixel.build()
    for name, sd in theme.sprites.items():
        for grid in sd._grids:
            for row in grid:
                if any(ch.isupper() for ch in row):
                    hi = f"{sd.palette_key}_hi"
                    assert hi in theme.palette, (
                        f"sprite {name!r} uses a highlight char but {hi!r} "
                        f"is missing from the palette"
                    )
                    break
