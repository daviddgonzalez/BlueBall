import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame, pymunk, pytest
from blueball.camera import FollowCamera
from blueball.render.core import RenderCore
from blueball.render.renderer import Renderer
from blueball.render.theme import get_active_theme


@pytest.fixture(autouse=True)
def _pg():
    pygame.init(); yield; pygame.quit()


def _renderer():
    core = RenderCore(pygame.Surface((1280, 720)))
    cam = FollowCamera(core.vw, core.vh); cam.scale = 1.0 / core.scale
    return core, Renderer(core, cam)


def test_ball_draws_theme_color_pixels():
    core, r = _renderer()
    r.camera.position = (320, 180)
    body = pymunk.Body(1, 1); body.position = (320, 180)
    core.surface.fill((0, 0, 0))
    r.draw_ball(body, alpha=1.0)
    ball_rgb = get_active_theme().palette["ball"]
    # The camera is positioned at the ball's world coords, so world_to_screen
    # projects it to the viewport center (vw/2, vh/2) in surface space.
    cx, cy = core.vw // 2, core.vh // 2
    found = any(core.surface.get_at((cx + dx, cy + dy))[:3] == ball_rgb
                for dx in range(-8, 9) for dy in range(-8, 9))
    assert found


def test_theme_cache_honors_runtime_switch(monkeypatch):
    """The per-renderer theme memo must re-resolve when config.ACTIVE_THEME
    changes, so runtime theme switching still works despite the cache."""
    from blueball import config
    from blueball.render import theme as theme_mod

    _, r = _renderer()
    pixel = r._theme()                       # resolves + caches "pixel"
    assert pixel is get_active_theme()

    sentinel = object()
    theme_mod.register_theme("test_alt", sentinel)
    monkeypatch.setattr(config, "ACTIVE_THEME", "test_alt")
    try:
        assert r._theme() is sentinel        # cache re-resolved on name change
    finally:
        theme_mod._REGISTRY.pop("test_alt", None)


def test_no_color_constants_remain():
    # Resolve the renderer source relative to THIS test file, not the cwd, so
    # the scan can't FileNotFoundError when pytest is invoked from elsewhere.
    import pathlib, re
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src = (repo_root / "src" / "blueball" / "render" / "renderer.py").read_text()
    # Guard against re-introducing baked-in color constants under the
    # color-ish suffixes the overhaul stripped out. Kept scoped to those
    # suffixes so it doesn't trip on legit non-color constants.
    leaked = re.findall(r"\b[A-Z][A-Z0-9_]*(?:_COLOR|_DARK|_FLAG|_EDGE|_DEFAULT)\b", src)
    assert not leaked, f"color-ish constants leaked back into renderer.py: {leaked}"
