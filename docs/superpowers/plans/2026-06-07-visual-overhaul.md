# Visual Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BlueBall's flat-primitive rendering with a pixel-art look built on a switchable theme system, plus animation, particles, and a parallax background — without touching physics, gameplay, or the AI/GA training path.

**Architecture:** One themed render path organized into modules under `src/blueball/render/`. A theme-agnostic `RenderCore` owns a 640×360 virtual surface that is nearest-neighbor-upscaled ×2 to the 1280×720 window (the camera is scaled by `1/PIXEL_SCALE` so the visible-world span is unchanged). A `Theme` dataclass (selected by `config.ACTIVE_THEME`) supplies palette, data-defined sprites, parallax layers, and tuning params. The ~23 `draw_*` methods become theme-driven sprite blits. Pixel is the first theme; `themes/neon.py` is a stubbed slot for later.

**Tech Stack:** Python 3.12, pygame-ce ≥ 2.5, pymunk ≥ 6.6, numpy, pytest. Headless tests run with `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.

**Spec:** `docs/superpowers/specs/2026-06-07-visual-overhaul-design.md`

**Baseline:** 409 tests green on this worktree (branch `worktree-feature+visual-overhaul`, off `origin/master`). Every task must keep them green.

**Conventions:**
- Run tests with: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./.venv/bin/python -m pytest -q`
- Commit at the end of each task (the user controls higher-level commits; per-task commits inside this plan are expected).
- Tune art/feel by eye via the run+screenshot loop (`./.venv/bin/python main.py`), then `pygame.image.save(window, "/tmp/shot.png")` to capture.

---

### Task 1: RenderCore — virtual surface + nearest-neighbor upscale

**Goal:** Introduce a theme-agnostic render engine that draws the world to a 640×360 virtual surface and upscales ×2 to the window, preserving the current visible-world span.

**Files:**
- Create: `src/blueball/render/core.py`
- Modify: `src/blueball/config.py` (add `PIXEL_SCALE`)
- Modify: `src/blueball/render/renderer.py:32-39` (`Renderer.__init__` accepts a `RenderCore` or a raw `Surface`)
- Modify: `src/blueball/scenes/play.py:49-50,177-185` (wire core, upscale on present)
- Modify: any other `Renderer(` construction sites — run `grep -rn "Renderer(" src/blueball/scenes` and update each (at least `scenes/train.py`)
- Test: `tests/test_render_core.py`

**Acceptance Criteria:**
- [ ] `RenderCore(window).surface` is 640×360 for a 1280×720 window; `present()` upscales onto the window.
- [ ] `RenderCore` rejects a window size not divisible by `PIXEL_SCALE`.
- [ ] `Renderer` accepts either a `RenderCore` or a raw `Surface` (back-compat for existing tests).
- [ ] PlayScene renders through the core and the visible-world span is unchanged (camera scale = `1/PIXEL_SCALE`).
- [ ] Full suite still green.

**Verify:** `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./.venv/bin/python -m pytest -q tests/test_render_core.py` → PASS

**Steps:**

- [ ] **Step 1: Write the failing test** — `tests/test_render_core.py`

```python
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
    core.present(flip=False)  # flip=False so the test needs no display
    assert window.get_at((0, 0))[:3] == (10, 20, 30)
    assert window.get_at((1279, 719))[:3] == (10, 20, 30)


def test_rejects_indivisible_window():
    window = pygame.Surface((1281, 720))
    with pytest.raises(ValueError):
        RenderCore(window, pixel_scale=2)
```

- [ ] **Step 2: Run to verify it fails** — `... pytest tests/test_render_core.py -q` → FAIL (`ModuleNotFoundError: blueball.render.core`).

- [ ] **Step 3: Add `PIXEL_SCALE` to `config.py`** (near the Display block, after `BACKGROUND_COLOR`):

```python
# Pixel-art render pipeline: world is drawn to a (WINDOW/ PIXEL_SCALE) virtual
# surface and nearest-neighbor-upscaled to the window. Must divide WINDOW evenly.
PIXEL_SCALE = 2
```

- [ ] **Step 4: Create `src/blueball/render/core.py`**

```python
"""RenderCore — theme-agnostic engine: a low-res virtual surface that is
nearest-neighbor-upscaled to the window. Owns the screen-shake offset that the
renderer adds into its world->surface transform (populated by the particles
task; (0, 0) until then)."""

from __future__ import annotations

import pygame

from .. import config


class RenderCore:
    def __init__(self, window: pygame.Surface, pixel_scale: int | None = None) -> None:
        self.window = window
        self.scale = pixel_scale if pixel_scale is not None else config.PIXEL_SCALE
        ww, wh = window.get_size()
        if ww % self.scale or wh % self.scale:
            raise ValueError(
                f"window {ww}x{wh} not divisible by pixel_scale {self.scale}"
            )
        self.vw, self.vh = ww // self.scale, wh // self.scale
        # .convert() requires a display; guard so headless construction works.
        surf = pygame.Surface((self.vw, self.vh))
        self.surface = surf.convert() if pygame.display.get_surface() else surf
        self.shake_offset: tuple[float, float] = (0.0, 0.0)

    def present(self, flip: bool = True) -> None:
        """Upscale the virtual surface onto the window (nearest-neighbor)."""
        pygame.transform.scale(self.surface, self.window.get_size(), self.window)
        if flip:
            pygame.display.flip()
```

- [ ] **Step 5: Make `Renderer` accept a core or a surface** — `renderer.py:32-39`

```python
    def __init__(self, target, camera) -> None:
        # `target` is a RenderCore (production) or a raw Surface (legacy tests).
        from .core import RenderCore
        if isinstance(target, RenderCore):
            self.core = target
            self.screen = target.surface
        else:
            self.core = None
            self.screen = target
        self.camera = camera
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._prev_angle: dict[int, float] = {}
        self._hud_font = None
```

- [ ] **Step 6: Wire PlayScene** — `play.py:49-50`

```python
        from ..render.core import RenderCore
        self.core = RenderCore(screen)
        self.camera = FollowCamera(self.core.vw, self.core.vh)
        # Preserve the pre-overhaul visible-world span on the smaller surface.
        self.camera.scale = 1.0 / self.core.scale
        self.renderer = Renderer(self.core, self.camera)
```

and replace `pygame.display.flip()` at `play.py:185` with:

```python
        self.core.present()
```

- [ ] **Step 7: Update other Renderer sites** — `grep -rn "Renderer(" src/blueball/scenes`. For `train.py`, build a `RenderCore` the same way and set `camera.scale = 1.0 / core.scale` (FreeCamera keeps its own zoom on top — multiply: `core_scale * zoom`; if that complicates the dev camera, leave train.py on a full-res `RenderCore(screen, pixel_scale=1)` for now and note it — train visuals are out of scope).

- [ ] **Step 8: Run tests** — new file passes; then full suite:
`SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./.venv/bin/python -m pytest -q` → 409+ pass. Fix any test that constructed `Renderer(screen, camera)` and asserted on window-sized draws (they now draw on the 640×360 surface — update expected coordinates or pass a 640×360 surface).

- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat(render): RenderCore virtual surface + upscale pipeline"`

---

### Task 2: Theme system — `Theme` dataclass + registry + config flag

**Goal:** A switchable theme abstraction selected by one config flag, with the pixel theme registered by default and a stubbed neon slot.

**Files:**
- Create: `src/blueball/render/theme.py`
- Create: `src/blueball/render/themes/__init__.py`
- Create: `src/blueball/render/themes/pixel.py` (palette-only stub here; sprites land in Task 4)
- Modify: `src/blueball/config.py` (add `ACTIVE_THEME`)
- Test: `tests/test_theme.py`

**Acceptance Criteria:**
- [ ] `get_active_theme()` returns the pixel theme when `config.ACTIVE_THEME == "pixel"`.
- [ ] `register_theme(name, theme)` + `get_theme(name)` round-trip; unknown name raises `KeyError`.
- [ ] Switching `config.ACTIVE_THEME` changes what `get_active_theme()` returns (proven with a dummy registered theme).
- [ ] `Theme` exposes `palette`, `sprites`, `parallax`, `pixel_scale`, `params`.

**Verify:** `... pytest -q tests/test_theme.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_theme.py`

```python
from dataclasses import replace
import blueball.config as config
from blueball.render.theme import Theme, register_theme, get_theme, get_active_theme


def test_pixel_theme_is_default():
    t = get_active_theme()
    assert isinstance(t, Theme)
    assert t.pixel_scale == config.PIXEL_SCALE
    assert "ball" in t.palette


def test_register_and_switch(monkeypatch):
    dummy = replace(get_theme("pixel"), palette={"ball": (1, 2, 3)})
    register_theme("dummy", dummy)
    monkeypatch.setattr(config, "ACTIVE_THEME", "dummy")
    assert get_active_theme().palette["ball"] == (1, 2, 3)


def test_unknown_theme_raises():
    import pytest
    with pytest.raises(KeyError):
        get_theme("does-not-exist")
```

- [ ] **Step 2: Run → FAIL** (no `blueball.render.theme`).

- [ ] **Step 3: Add flag to `config.py`**

```python
# Active render theme. "pixel" ships now; "neon" is a future slot.
ACTIVE_THEME = "pixel"
```

- [ ] **Step 4: Create `render/theme.py`**

```python
"""Theme — data the renderer reads to draw a given art style. Switchable via
config.ACTIVE_THEME. Themes register themselves at import time."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ParallaxLayer:
    """A baked background strip + a scroll factor (0 = static, 1 = world-locked)."""
    sprite_key: str
    factor: float
    y: int = 0


@dataclass(frozen=True)
class Theme:
    palette: dict[str, Color]
    sprites: dict[str, object] = field(default_factory=dict)   # name -> SpriteDef (Task 3)
    parallax: list[ParallaxLayer] = field(default_factory=list)
    pixel_scale: int = config.PIXEL_SCALE
    params: dict = field(default_factory=dict)


_REGISTRY: dict[str, Theme] = {}


def register_theme(name: str, theme: Theme) -> None:
    _REGISTRY[name] = theme


def get_theme(name: str) -> Theme:
    return _REGISTRY[name]


def get_active_theme() -> Theme:
    return _REGISTRY[config.ACTIVE_THEME]


# Import side effect: register built-in themes.
from .themes import pixel as _pixel  # noqa: E402
register_theme("pixel", _pixel.build())
```

- [ ] **Step 5: Create `render/themes/__init__.py`** (empty) and `render/themes/pixel.py` palette stub:

```python
"""Pixel theme — the first concrete art style. Sprites are added in Task 4."""

from __future__ import annotations

from ..theme import Theme

PALETTE = {
    "ball": (58, 138, 255),
    "ball_hi": (191, 224, 255),
    "ground": (63, 154, 85),
    "ground_top": (111, 217, 138),
    "spike": (226, 80, 63),
    "coin": (255, 210, 58),
    "sky_top": (207, 234, 255),
    "sky_bottom": (126, 199, 255),
}


def build() -> Theme:
    return Theme(palette=dict(PALETTE), params={
        "squash_max": 0.35, "shake_decay": 8.0, "particle_cap": 300,
    })
```

- [ ] **Step 6: Run** `... pytest -q tests/test_theme.py` → PASS, then full suite green.

- [ ] **Step 7: Commit** — `git commit -am "feat(render): switchable Theme registry + pixel theme stub"`

---

### Task 3: Sprite format + baker (pixel-grid → cached `Surface`)

**Goal:** A data sprite format that bakes a grid of palette-key chars into a cached `pygame.Surface`, supporting multi-frame sprites.

**Files:**
- Create: `src/blueball/render/sprites.py`
- Test: `tests/test_sprites.py`

**Acceptance Criteria:**
- [ ] `SpriteDef(grid, palette_key)` bakes to a Surface sized `len(grid[0]) × len(grid)` with `SRCALPHA`.
- [ ] `.` cells are transparent; other chars resolve to palette shades (`<key>` lowercase = base, `<key>_hi` uppercase = highlight).
- [ ] Baking is cached (second `bake()` returns the same Surface object).
- [ ] Multi-frame `SpriteDef` exposes `frame(i)`.

**Verify:** `... pytest -q tests/test_sprites.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_sprites.py`

```python
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
    assert sd.frame(0).get_at((0, 0))[:3] == (10, 20, 30)
    assert sd.frame(1).get_at((0, 0))[:3] == (200, 210, 220)
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Create `render/sprites.py`**

```python
"""Data-defined pixel sprites. A grid of palette-key chars baked once into a
cached pygame.Surface. Lowercase char = palette[key]; uppercase = palette[key+'_hi'];
'.' = transparent."""

from __future__ import annotations

import pygame

Color = tuple[int, int, int]


class SpriteDef:
    def __init__(self, grid, palette_key: str, frames: int = 1) -> None:
        # Single-frame: grid is list[str]. Multi-frame: grid is list[list[str]].
        self.frames_n = frames
        self._grids = grid if frames > 1 else [grid]
        self.palette_key = palette_key
        self._cache: list[pygame.Surface] | None = None

    def _resolve(self, ch: str, palette: dict[str, Color]):
        if ch == ".":
            return None
        if ch.isupper():
            return palette.get(f"{self.palette_key}_hi", palette[self.palette_key])
        if ch.islower():
            return palette[self.palette_key]
        # Explicit palette key char map could be added; default to base.
        return palette[self.palette_key]

    def _bake_grid(self, rows, palette) -> pygame.Surface:
        h = len(rows)
        w = max(len(r) for r in rows)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                col = self._resolve(ch, palette)
                if col is not None:
                    surf.set_at((x, y), col)
        return surf

    def bake(self, palette: dict[str, Color]) -> pygame.Surface:
        return self.frame(0, palette)

    def frame(self, i: int, palette: dict[str, Color] | None = None) -> pygame.Surface:
        if self._cache is None:
            if palette is None:
                raise ValueError("first bake needs a palette")
            self._cache = [self._bake_grid(g, palette) for g in self._grids]
        return self._cache[i % self.frames_n]
```

- [ ] **Step 4: Run → PASS**, full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat(render): data sprite format + cached baker"`

---

### Task 4: Pixel theme data — palette + core sprites

**Goal:** Populate the pixel theme with the palette and the initial sprite set (designed at 2× grain), enough for the renderer refactor in Task 5; remaining entity sprites are finished in Task 9.

**Files:**
- Modify: `src/blueball/render/themes/pixel.py`
- Test: `tests/test_pixel_theme.py`

**Acceptance Criteria:**
- [ ] `build().sprites` contains at least: `ball`, `spike`, `collectible`, `goal`, `coin`. (Sized in virtual px: e.g. ball ≈ 16×16.)
- [ ] Every sprite key bakes without error against the theme palette.
- [ ] Palette has a color for every `*_key` referenced by a sprite.

**Verify:** `... pytest -q tests/test_pixel_theme.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_pixel_theme.py`

```python
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
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Add sprites to `pixel.py`** (starter art — tune by eye in Task 9). Example ball (16×16) reusing the stepped-circle from the design mockups:

```python
from ..sprites import SpriteDef

_BALL = SpriteDef(grid=[
    ".....bbbbbb.....",
    "...bbbbbbbbbb...",
    "..bbBBBbbbbbbb..",
    ".bbBBBBbbbbbbbb.",
    ".bbBBbbbbbbbbbb.",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    ".bbbbbbbbbbbbbb.",
    ".bbbbbbbbbbbbbb.",
    "..bbbbbbbbbbbb..",
    "...bbbbbbbbbb...",
    ".....bbbbbb.....",
], palette_key="ball")

_SPIKE = SpriteDef(grid=[
    ".......ss.......",
    "......ssss......",
    ".....ssssss.....",
    "....ssssssss....",
    "...ssssssssss...",
    "..ssssssssssss..",
    ".ssssssssssssss.",
    "ssssssssssssssss",
], palette_key="spike")

_COIN = SpriteDef(grid=[
    "..cccc..",
    ".cCCccc.",
    "cCCccccc",
    "cCcccccc",
    "cccccccc",
    "cccccccc",
    ".cccccc.",
    "..cccc..",
], palette_key="coin")
```

Update `build()`:

```python
def build() -> Theme:
    sprites = {
        "ball": _BALL,
        "spike": _SPIKE,
        "collectible": _COIN,
        "coin": _COIN,
        "goal": _GOAL,   # define a small flag/portal grid similarly
    }
    return Theme(palette=dict(PALETTE), sprites=sprites, params={
        "squash_max": 0.35, "shake_decay": 8.0, "particle_cap": 300,
    })
```

(Define `_GOAL` as a ~24×24 flag grid with palette keys `goal`/`goal_hi`; add those colors to `PALETTE`.)

- [ ] **Step 4: Run → PASS**, full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat(render): pixel theme core sprite set"`

---

### Task 5: Theme-driven renderer — blit sprites instead of primitives

**Goal:** Convert the `draw_*` methods to blit baked theme sprites onto the virtual surface; remove module-level color constants. Establish the blit-from-theme pattern for all entities (the long tail is finished in Task 9).

**Files:**
- Modify: `src/blueball/render/renderer.py` (remove `_*_COLOR` constants; add a sprite-blit helper; convert `draw_ball`, `draw_spike`, `draw_collectible`, `draw_goal`, `draw_static_segments` now)
- Test: `tests/test_renderer_theme.py`

**Acceptance Criteria:**
- [ ] A `_blit_sprite(world_xy, sprite_key, *, anchor="center", angle=0.0)` helper places a baked sprite at the world position (via `camera.world_to_screen`, centered).
- [ ] `draw_ball`/`draw_spike`/`draw_collectible`/`draw_goal` draw their theme sprite (no hardcoded RGB).
- [ ] No `_*_COLOR` module constants remain in `renderer.py` (`grep -n "_COLOR" src/blueball/render/renderer.py` is empty).
- [ ] Existing render tests updated and green.

**Verify:** `... pytest -q tests/test_renderer_theme.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_renderer_theme.py`

```python
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
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
    body = pymunk.Body(1, 1); body.position = (320, 180)  # center of view
    r.camera.position = (320, 180)
    core.surface.fill((0, 0, 0))
    r.draw_ball(body, alpha=1.0)
    ball_rgb = get_active_theme().palette["ball"]
    # Some pixel near the surface center now carries the ball base color.
    found = any(core.surface.get_at((320 // 2 + dx, 180 // 2 + dy))[:3] == ball_rgb
                for dx in range(-4, 5) for dy in range(-4, 5))
    assert found
```

- [ ] **Step 2: Run → FAIL** (ball still drawn as a primitive circle in the old color).

- [ ] **Step 3: Add the sprite-blit helper + active-theme access** in `Renderer`:

```python
    from .theme import get_active_theme

    def _theme(self):
        from .theme import get_active_theme
        return get_active_theme()

    def _blit_sprite(self, world_xy, key, *, angle=0.0, frame=0):
        theme = self._theme()
        sd = theme.sprites[key]
        surf = sd.frame(frame, theme.palette)
        if angle:
            surf = pygame.transform.rotate(surf, -math.degrees(angle))
        sx, sy = self.camera.world_to_screen(world_xy)
        ox, oy = self.core.shake_offset if self.core else (0.0, 0.0)
        rect = surf.get_rect(center=(int(sx + ox), int(sy + oy)))
        self.screen.blit(surf, rect)
```

- [ ] **Step 4: Convert `draw_ball`** (replace lines 106-115):

```python
    def draw_ball(self, body, alpha: float) -> None:
        wx, wy = self._interp_body_pos(body, alpha)
        angle = self._interp_body_angle(body, alpha)
        self._blit_sprite((wx, wy), "ball", angle=angle)
```

- [ ] **Step 5: Convert `draw_spike`, `draw_collectible`, `draw_goal`** to `_blit_sprite((x, y), "spike"/"collectible"/"goal")` (drop the polygon/circle code and the color constants). For `draw_static_segments`, blit a tiled `ground` sprite along each segment or keep a themed line using `theme.palette["ground"]`/`["ground_top"]` (palette, not the deleted constants).

- [ ] **Step 6: Delete the `_*_COLOR` / `_GROUND_*` / `_BOOST_*` module constants** (lines 13-25). Any `draw_*` not yet converted (Task 9) should read its color from `self._theme().palette[...]` in the interim so nothing references a deleted name. Run `grep -n "_COLOR\|_BALL_DARK\|_GROUND_EDGE\|_BOOST_PAD" src/blueball/render/renderer.py` → empty.

- [ ] **Step 7: Run** the new test + full suite; fix render tests that asserted old primitive colors/shapes.

- [ ] **Step 8: Commit** — `git commit -am "refactor(render): theme-driven sprite blits for core entities"`

---

### Task 6: Animation system (transform squash/stretch/spin + frame `Anim`)

**Goal:** Procedural transform animation for the ball (stretch on rise, squash on land impact, spin already via `angle`) plus a small frame-based `Anim` helper and a palette-cycle helper.

**Files:**
- Create: `src/blueball/render/animation.py`
- Modify: `src/blueball/render/renderer.py` (`draw_ball` applies squash/stretch from velocity)
- Test: `tests/test_animation.py`

**Acceptance Criteria:**
- [ ] `squash_stretch(vy, max_amount)` returns an `(sx, sy)` scale tuple: vertical stretch for large |vy| upward, squash on downward impact, ~(1,1) at rest, area-preserving.
- [ ] `Anim(n_frames, fps).index(t)` cycles 0..n-1 over time.
- [ ] `draw_ball` applies the squash/stretch scale to the ball sprite (visible change in rendered bounding box when `vy` is large).

**Verify:** `... pytest -q tests/test_animation.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_animation.py`

```python
from blueball.render.animation import squash_stretch, Anim


def test_rest_is_identity():
    sx, sy = squash_stretch(0.0, max_amount=0.3)
    assert abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6


def test_fast_rise_stretches_vertically():
    sx, sy = squash_stretch(-400.0, max_amount=0.3)
    assert sy > 1.0 and sx < 1.0           # taller, thinner
    assert abs(sx * sy - 1.0) < 0.05       # ~area-preserving


def test_anim_cycles():
    a = Anim(n_frames=3, fps=10)
    assert a.index(0.0) == 0
    assert a.index(0.15) == 1
    assert a.index(0.25) == 2
    assert a.index(0.35) == 0
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Create `render/animation.py`**

```python
"""Procedural animation helpers. Transform-based (resolution-independent) plus a
tiny frame cycler and a palette-cycle utility."""

from __future__ import annotations


def squash_stretch(vy: float, max_amount: float = 0.3, ref_speed: float = 400.0):
    """Return an (sx, sy) scale for vertical velocity vy (pymunk y-down: vy<0 is
    up). Stretches tall+thin while moving fast vertically; ~area-preserving."""
    amt = max(-1.0, min(1.0, -vy / ref_speed)) * max_amount  # up (vy<0) -> +stretch
    sy = 1.0 + amt
    sx = 1.0 / sy if sy != 0 else 1.0
    return (sx, sy)


class Anim:
    def __init__(self, n_frames: int, fps: float) -> None:
        self.n_frames = n_frames
        self.fps = fps

    def index(self, t_seconds: float) -> int:
        return int(t_seconds * self.fps) % self.n_frames


def palette_cycle(base: list, t_seconds: float, hz: float) -> int:
    """Index into a cyclic palette list (for lava shimmer / coin twinkle)."""
    return int(t_seconds * hz) % len(base)
```

- [ ] **Step 4: Apply squash/stretch in `draw_ball`** — scale the baked ball sprite before the blit:

```python
    def draw_ball(self, body, alpha: float) -> None:
        from .animation import squash_stretch
        wx, wy = self._interp_body_pos(body, alpha)
        angle = self._interp_body_angle(body, alpha)
        sx, sy = squash_stretch(body.velocity.y,
                                self._theme().params.get("squash_max", 0.3))
        self._blit_sprite((wx, wy), "ball", angle=angle, scale=(sx, sy))
```

Extend `_blit_sprite` to accept `scale=(sx, sy)` and apply `pygame.transform.scale` (round to ints, min 1px) before optional rotate.

- [ ] **Step 5: Run → PASS**, full suite green.
- [ ] **Step 6: Commit** — `git commit -am "feat(render): procedural squash/stretch + frame anim helper"`

---

### Task 7: Particle system, impact FX & screen shake

**Goal:** A capped particle pool with emit presets, fired from scene-known state transitions, plus a decaying screen shake folded into the core's offset.

**Files:**
- Create: `src/blueball/render/particles.py`
- Modify: `src/blueball/render/core.py` (decay shake in an `update(dt)`; expose `add_shake`)
- Modify: `src/blueball/scenes/play.py` (own a `ParticleSystem`; emit on land/collect/die/boost; update+draw each frame)
- Test: `tests/test_particles.py`

**Acceptance Criteria:**
- [ ] `ParticleSystem(cap)` never holds more than `cap` particles (oldest dropped).
- [ ] `emit("dust"/"sparkle"/"burst"/"trail", at, n)` adds particles; `update(dt)` ages+removes dead ones; `draw(renderer)` blits them.
- [ ] `RenderCore.add_shake(mag)` then `update(dt)` decays `shake_offset` toward 0.
- [ ] Particle emission is driven from PlayScene transitions; **no physics/entity logic changed**.

**Verify:** `... pytest -q tests/test_particles.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_particles.py`

```python
from blueball.render.particles import ParticleSystem


def test_cap_enforced():
    ps = ParticleSystem(cap=10)
    ps.emit("dust", (0, 0), n=50)
    assert len(ps) <= 10


def test_particles_age_out():
    ps = ParticleSystem(cap=100)
    ps.emit("burst", (0, 0), n=5)
    for _ in range(1000):
        ps.update(0.1)
    assert len(ps) == 0
```

(Add a shake test in `tests/test_render_core.py`: `core.add_shake(10); core.update(1.0); assert abs(core.shake_offset[0]) < 10`.)

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Create `render/particles.py`**

```python
"""Lightweight particle pool. Pos/vel/life/color/size dataclass-free for speed.
Capped so it never blows the frame budget."""

from __future__ import annotations

import collections
import math

_PRESETS = {
    # kind: (speed, life, size, gravity, color_key)
    "dust":    (60.0, 0.4, 2, 40.0, "ground_top"),
    "sparkle": (90.0, 0.5, 2, -20.0, "coin"),
    "burst":   (160.0, 0.6, 2, 120.0, "spike"),
    "trail":   (40.0, 0.3, 2, 0.0, "ball_hi"),
}


class ParticleSystem:
    def __init__(self, cap: int = 300) -> None:
        self.cap = cap
        self._p = collections.deque(maxlen=cap)  # auto-drops oldest past cap

    def __len__(self) -> int:
        return len(self._p)

    def emit(self, kind: str, at, n: int = 8, seed_angle: float = 0.0) -> None:
        speed, life, size, grav, key = _PRESETS[kind]
        for i in range(n):
            a = seed_angle + (i / max(1, n)) * 2 * math.pi
            self._p.append([at[0], at[1],
                            math.cos(a) * speed, math.sin(a) * speed,
                            life, size, grav, key])

    def update(self, dt: float) -> None:
        for p in self._p:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += p[6] * dt          # gravity
            p[4] -= dt                 # life
        # Drop dead (rebuild deque preserving order)
        self._p = collections.deque((p for p in self._p if p[4] > 0), maxlen=self.cap)

    def draw(self, renderer) -> None:
        theme = renderer._theme()
        for p in self._p:
            col = theme.palette[p[7]]
            renderer._blit_point((p[0], p[1]), col, int(p[5]))
```

Add `Renderer._blit_point(world_xy, color, size)` (a small filled rect on the virtual surface via `camera.world_to_screen` + shake offset).

- [ ] **Step 4: Add shake to `core.py`**

```python
    def add_shake(self, magnitude: float) -> None:
        self._shake_mag = getattr(self, "_shake_mag", 0.0) + magnitude

    def update(self, dt: float) -> None:
        mag = getattr(self, "_shake_mag", 0.0)
        decay = config.__dict__.get("SHAKE_DECAY", 8.0)
        mag = max(0.0, mag - decay * mag * dt)
        self._shake_mag = mag
        # Cheap deterministic-ish jitter from a frame counter (no Math.random).
        self._t = getattr(self, "_t", 0) + 1
        s = (self._t % 7 - 3)
        self.shake_offset = (s * mag * 0.3, ((self._t // 3) % 5 - 2) * mag * 0.3)
```

- [ ] **Step 5: Wire PlayScene** — construct `self.particles = ParticleSystem(get_active_theme().params["particle_cap"])`; detect transitions using existing state (e.g. a `was_grounded`/`was_dead`/score-increase compare across frames) and call `self.particles.emit(...)` + `self.core.add_shake(...)`; in `update()` call `self.core.update(frame_dt)` and `self.particles.update(frame_dt)`; in `draw()` call `self.particles.draw(self.renderer)` after entities. **Touch no entity/physics code.**

- [ ] **Step 6: Run → PASS**, full suite green.
- [ ] **Step 7: Commit** — `git commit -am "feat(render): particles, impact FX, screen shake"`

---

### Task 8: Parallax background (layered scrolling)

**Goal:** Replace the flat background fill with N scrolling parallax layers defined in theme data.

**Files:**
- Create: `src/blueball/render/parallax.py`
- Modify: `src/blueball/render/renderer.py` (`draw_parallax(camera)` replaces flat `draw_background` in the play path)
- Modify: `src/blueball/render/themes/pixel.py` (sky/hills/near layer sprites + `parallax` list)
- Modify: `src/blueball/scenes/play.py:178` (call `draw_parallax` instead of `draw_background`)
- Test: `tests/test_parallax.py`

**Acceptance Criteria:**
- [ ] `layer_offset(camera_x, factor, tile_w)` returns the horizontal blit offset wrapped into `[-tile_w, 0]`.
- [ ] `draw_parallax` draws each theme layer tiled across the surface; factor 0 layer doesn't move with camera, factor 1 layer moves 1:1.
- [ ] Play scene shows depth as the camera pans (manual check).

**Verify:** `... pytest -q tests/test_parallax.py` → PASS

**Steps:**

- [ ] **Step 1: Failing test** — `tests/test_parallax.py`

```python
from blueball.render.parallax import layer_offset


def test_offset_wraps():
    assert layer_offset(0.0, 0.5, 100) == 0.0
    off = layer_offset(250.0, 0.5, 100)   # 250*0.5=125 -> wrap into (-100,0]
    assert -100 < off <= 0


def test_factor_zero_is_static():
    assert layer_offset(9999.0, 0.0, 64) == 0.0
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Create `render/parallax.py`**

```python
"""Parallax background: tiled strips scrolled by a per-layer factor."""

from __future__ import annotations


def layer_offset(camera_x: float, factor: float, tile_w: int) -> float:
    return -((camera_x * factor) % tile_w)
```

- [ ] **Step 4: Add `draw_parallax` to `Renderer`** — for each `ParallaxLayer` in `self._theme().parallax`, bake its sprite, compute `layer_offset(camera.position[0], layer.factor, tile_w)`, and tile-blit horizontally across `core.vw` at `layer.y`. Fill the sky gradient first (a vertical gradient from `palette["sky_top"]` to `["sky_bottom"]`, baked once).

- [ ] **Step 5: Define layers in `pixel.py`** — add `sky` (full-height gradient strip, factor 0), `hills_far` (factor 0.3), `hills_near` (factor 0.6) sprite grids + `parallax=[ParallaxLayer("sky",0,0), ParallaxLayer("hills_far",0.3, vh-80), ParallaxLayer("hills_near",0.6, vh-50)]`.

- [ ] **Step 6: Swap in PlayScene** — replace `self.renderer.draw_background(self.level_meta.background)` with `self.renderer.draw_parallax(self.camera)`.

- [ ] **Step 7: Run → PASS**, full suite green; visually confirm via `./.venv/bin/python main.py`.
- [ ] **Step 8: Commit** — `git commit -am "feat(render): parallax scrolling background"`

---

### Task 9: Per-entity visual treatment pass

**Goal:** Finish every remaining entity from the spec's treatment table (§5): sprite + animation + FX. This is the by-eye art pass; mechanism is already in place from Tasks 5–8.

**Files:**
- Modify: `src/blueball/render/themes/pixel.py` (sprites/frames for all remaining entities + HUD font)
- Modify: `src/blueball/render/renderer.py` (convert remaining `draw_*` to sprites/anim; wire FX emit points that belong in the renderer)
- Test: `tests/test_renderer_all_entities.py` (smoke per entity)

**Acceptance Criteria (one box per entity in the table):**
- [ ] ball ✓ (Task 5/6) · static_segments · one_way_platform · moving_platform · crumbling_platform · pushable_box
- [ ] spike ✓ · lava (palette-cycle shimmer) · falling_hazard · swinging_hazard · patroller (2-frame) · charger (idle/charge) · cannon (+muzzle flash) · projectile (palette-cycle)
- [ ] collectible ✓ · ability_pickup · key · door (locked/open) · goal ✓ · checkpoint (inactive/active) · boost_pad (animated chevrons) · spring (compress/extend)
- [ ] HUD score uses a pixel bitmap font
- [ ] `grep` confirms no raw RGB literals left in `draw_*` except via `palette`
- [ ] Each entity has a smoke test; full suite green

**Verify:** `... pytest -q tests/test_renderer_all_entities.py` → PASS, then full suite.

**Steps:**

- [ ] **Step 1: Smoke test scaffold** — `tests/test_renderer_all_entities.py` constructs a `RenderCore`+`Renderer`, then calls each `draw_*` with a minimal valid argument set and asserts it doesn't raise and writes at least one non-background pixel. (Reuse the `_renderer()` helper pattern from Task 5's test.)

- [ ] **Step 2: For each entity**, working down the table: (a) add its sprite grid(s) to `pixel.py` (add palette keys as needed), (b) convert its `draw_*` to `_blit_sprite`/frame/`palette_cycle`, (c) for FX-bearing entities wire the emit at the correct transition (in PlayScene where state is known). Run its smoke test, eyeball with `main.py`, iterate the grid until it reads well.

- [ ] **Step 3: HUD font** — bake a small bitmap-font sprite sheet (or use a pixel TTF rendered once per glyph and cached) for `draw_score`.

- [ ] **Step 4: Full visual pass** — run the game, screenshot a scene with several entity types, tune palette cohesion and grain. Confirm 60fps via a temporary FPS print in the main loop.

- [ ] **Step 5: Commit** — `git commit -am "feat(render): pixel treatment for all entities + HUD font"`

---

### Task 10: Rendering tests, training-isolation guard & smoke render

**Goal:** Consolidate the render test suite, add the determinism/isolation guard (training must never import the renderer), and a full-scene smoke render.

**Files:**
- Create: `tests/test_render_isolation.py`
- Create: `tests/test_render_smoke.py`
- Test: the above

**Acceptance Criteria:**
- [ ] A test imports the training entrypoints (`blueball.scenes.train` headless path and `train_infinite`/`train_levels` modules) and asserts `blueball.render.renderer` / `blueball.render.core` are **not** in `sys.modules` after a headless GA step. (If training legitimately imports a scene that pulls the renderer, assert the renderer is never *instantiated* on the training path instead — document which.)
- [ ] Theme-switch test: set `config.ACTIVE_THEME="pixel"`, render a frame; register a dummy theme, switch, render — both succeed.
- [ ] Smoke render: a level/scene containing every entity type renders one frame onto a 640×360 surface and `present()`s to a 1280×720 window without error.
- [ ] `pixel_scale` upscale produces a 1280×720 window surface.
- [ ] Full suite green (≥ 409 + new tests).

**Verify:** `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./.venv/bin/python -m pytest -q` → all pass.

**Steps:**

- [ ] **Step 1: Isolation test** — `tests/test_render_isolation.py`

```python
import importlib, sys


def test_training_does_not_import_renderer():
    for mod in list(sys.modules):
        if mod.startswith("blueball.render"):
            del sys.modules[mod]
    # Import the headless training surface and run a tiny evaluation.
    from blueball.train import run as train_run  # adjust to real entrypoint
    train_run(generations=1, pop_size=2, max_steps=10, seed=1)
    assert "blueball.render.renderer" not in sys.modules
    assert "blueball.render.core" not in sys.modules
```

(Adjust the import to the real headless trainer entrypoint discovered via `grep -rn "def run\|def main" train_*.py src/blueball/scenes/train.py`. If the trainer imports a scene that imports the renderer, change the assertion to "renderer never instantiated" using a monkeypatched `Renderer.__init__` that records calls.)

- [ ] **Step 2: Smoke render** — `tests/test_render_smoke.py` builds a small in-memory level dict containing one of each entity, runs one `PlayScene` frame (`update` + `draw`) on a dummy display, asserts no exception and the window is 1280×720.

- [ ] **Step 3: Run full suite → green.**
- [ ] **Step 4: Commit** — `git commit -am "test(render): isolation guard + full-scene smoke render"`

---

## Notes for the implementer

- **Keep entities untouched.** All visual logic lives in `render/`. Entities already pass their physics state to `draw_*`; read from there. The only PlayScene additions are FX emit calls on transitions you can already observe — never change physics, collision, input, level gen, or the agent/GA code.
- **Theme switch is the deliverable's spine.** When in doubt, put style data in the theme, not in `renderer.py`. The neon theme later should need *no* renderer or entity changes — only `themes/neon.py` + one registry line.
- **Art is iterative.** Sprite grids and palette values in this plan are starting points; the by-eye loop (`main.py` + screenshot) is where they get good. That's expected, not a deviation.
- **Performance:** sprites are baked once and cached; particle pool is capped; only the play/menu path renders. If `transform.rotate` on the ball shows hot, cache rotations in angle buckets.
