# Visual Overhaul — Design Spec

**Date:** 2026-06-07
**Status:** Approved design, pending implementation plan
**Scope:** Replace BlueBall's flat-primitive rendering with a pixel-art look, built on a switchable theme system so a second style (neon) can drop in later. Physics, gameplay, level design, and the AI/GA training harness are **not** touched.

---

## 1. Goals & non-goals

**Goals**
- A cohesive **pixel-art** visual style for the whole game (the current look is untextured polygons).
- A **switchable theme/skin system** — pixel is the first theme; **neon synthwave** is the planned second (built later, not now).
- Game-feel "juice": **animated sprites**, **particles & impact FX**, and a **parallax background**.

**Non-goals (this slice)**
- The neon theme itself (only the seam + a `themes/neon.py` placeholder location).
- Retro post-FX (full-screen scanlines / CRT overlay, global glow) — explicitly **cut** for now; neon can bring its own glow later.
- Any change to physics, collision, input feel, level generation, entity behavior, or AI/GA training.

---

## 2. Decisions locked during brainstorming

| Decision | Choice | Why |
|---|---|---|
| Art direction | **Pixel art** first; **neon** planned second | User liked both B (pixel) and D (neon); wants pixel now, switchable later |
| Sprite source | **Procedural / data sprites** (pixel grids defined in Python, baked + cached) | In-repo, deterministic, diff-able; palette-swap makes the neon theme cheap; fits this code-first codebase |
| Pixel grain | **2×** → 640×360 internal, upscaled to 1280×720 | Room for a face + shading; clean integer scale (no pillarbox); animation cost barely higher than 4× because most motion is transform/procedural |
| Juice scope | Animated sprites + particles/impact FX + parallax background | User selection; retro post-FX **excluded** |
| Architecture | **One themed render path, organized into modules** (Approach 3) | Pixel & neon are both raster styles → no need for per-theme backend classes; modular split keeps `renderer.py` readable |

---

## 3. Architecture

Entities are **unchanged**: each still implements `draw(self, renderer, alpha)` and delegates to `renderer.draw_*`. The overhaul lives in the `render/` package. The `Renderer` is the existing seam.

### 3.1 Module layout

```
src/blueball/render/
  core.py        # RenderCore: 640x360 virtual surface + nearest-neighbor upscale,
                 #   camera transform (visible-world span preserved), screen-shake
                 #   offset, and the tick-interpolation moved out of renderer.py
  theme.py       # Theme dataclass + registry + get_active_theme() (reads config.ACTIVE_THEME)
  sprites.py     # SpriteDef (pixel grid + palette) + bake-to-Surface (cached, multi-frame)
  animation.py   # transform squash/stretch/spin + frame-based Anim helper + palette-cycle
  particles.py   # ParticleSystem (capped pool) + emit presets + CameraShake
  parallax.py    # layered scrolling background
  themes/
    __init__.py
    pixel.py     # the pixel theme: palette + sprite grids + parallax layers + params
    # neon.py    # FUTURE — placeholder location only, not built this slice
  renderer.py    # the ~23 draw_* methods — theme-driven glue, no hardcoded colors
```

### 3.2 Theme abstraction

A theme is **data**, selected by one config flag:

```python
# config.py
ACTIVE_THEME = "pixel"          # the switch

# render/theme.py
@dataclass(frozen=True)
class Theme:
    palette: dict[str, Color]            # named colors: "ball", "ground_top", "spike", ...
    sprites: dict[str, SpriteDef]        # entity-name -> pixel-grid sprite(s)
    parallax: list[ParallaxLayer]        # background layers
    pixel_scale: int                     # 2 (640x360 internal)
    params: dict                         # squash/stretch, particle, shake tuning
```

`register_theme("pixel", build_pixel_theme())`; `get_active_theme()` reads the flag. Adding neon later = one new file + one registry line. Switching = change the flag (or a `--theme` CLI arg).

### 3.3 Sprite format

Each sprite is a tiny grid of palette-key chars, baked **once** into a `pygame.Surface` and cached. Multi-frame sprites are a list of grids.

```python
BALL = SpriteDef(grid=[
    "..bbbb..",
    ".bBBBbb.",     # '.' transparent; lowercase/uppercase = palette shades
    "bBBbbbbb",
    ...
], palette_key="ball")
```

Recoloring for neon = resolve the same grids against a different palette. This is why the data-sprite format makes a second theme cheap.

### 3.4 Pixel pipeline

The world is drawn to a **640×360 virtual surface**, then **nearest-neighbor-upscaled ×2** to the 1280×720 window. The camera is scaled so the **visible-world span is exactly what it is today** — the overhaul changes how things look, not how much you see or how the game plays. 2× divides 1280×720 evenly, so there is no pillarbox.

---

## 4. Dynamic systems

### 4.1 Animation
Entities stay untouched; the `draw_*` methods read the physics state entities already expose.
- **Transform-based (procedural, no frame art):** ball stretches when rising fast, squashes on landing impact, spins continuously from `body.angle`. Scale/rotate on the cached sprite. This is the bulk of the ball's life.
- **Frame-based (`Anim` helper):** list of baked frames + per-entity timer/index for cyclic motion (patroller, charger, lava, twinkle, crumble dust).
- **Palette-cycle/offset:** "animations" like lava shimmer and collectible twinkle are done by cycling palette indices — free, no drawn frames.
- Squash/stretch + timing live in theme `params` (neon can tune differently).

### 4.2 Particles, impact FX & screen shake
- `ParticleSystem` with a **capped pool (~300)** of lightweight particles (pos/vel/life/color/size). `emit(kind, at, …)` presets: **landing dust, collect sparkle, death burst, boost trail**. The cap guarantees the frame budget.
- **Hooks** come from scene-known state transitions ("just landed", "collected", "died", "boosted"); at most a tiny per-frame flag is added where one doesn't exist. **No physics changes.**
- **Screen shake:** a decaying-offset `CameraShake` folded into `core.py`'s camera→screen step; fires on death / spring / hard landing. Params in theme.

### 4.3 Parallax background
- N layers, each = a baked strip + a scroll factor (0 = static sky … →1 = moves with world), drawn before the world; offset = `camera.x * factor`, tiled horizontally.
- Pixel theme ships ~3: **sky gradient (0) · distant hills (0.3) · near hills/clouds (0.6)**. Replaces today's flat `draw_background(BACKGROUND_COLOR)`.
- Layer defs live in theme data (neon swaps to a dark grid/horizon).

---

## 5. Per-entity visual treatment

Authoring checklist for the pixel theme + animation/FX work. "Anim" = motion; "FX" = particles triggered.

| Entity | Base sprite | Anim | FX |
|---|---|---|---|
| Ball (player) | round body + highlight + eyes | squash/stretch + spin (transform) | dust on land · trail on boost · burst on death |
| Static segments | tiled blocks, grass-top edge | — | — |
| One-way platform | thin pixel ledge | — | — |
| Moving platform | pixel block | translates (physics) | — |
| Crumbling platform | pixel block | crack frames | dust on break |
| Pushable box | wooden crate | — | scuff dust on push |
| Spike | pixel spikes | — | — |
| Lava | pixel lava | palette-cycle shimmer + bubbles | — |
| Falling hazard | rock | wobble before drop | impact dust |
| Swinging hazard | spiked ball + chain | swings (physics) | — |
| Patroller | walking critter | 2-frame walk | — |
| Charger | angry critter | idle / charge frames | dust on charge |
| Cannon | barrel | — | muzzle flash |
| Projectile | small orb | palette-cycle pulse | — |
| Collectible | coin | twinkle (palette-cycle) | sparkle on collect |
| Ability pickup | glowing icon | gentle pulse | sparkle on collect |
| Key | key sprite | gentle bob | sparkle on collect |
| Door | door (locked/open) | open transition | — |
| Goal | flag / portal | gentle wave | celebratory burst on reach |
| Checkpoint | flag (inactive/active) | raise on activate | small puff on activate |
| Boost pad | arrow strip | animated chevrons | trail while boosting |
| Spring | coil | compress/extend on trigger | puff on launch |
| HUD score | pixel bitmap font | — | — |

---

## 6. Testing strategy

- **Headless rendering** — render to an offscreen `Surface` with `SDL_VIDEODRIVER=dummy` (matches the existing suite). Cover: sprite baking color-correctness; theme registry + **theme-switch**; particle pool **cap** never exceeded; parallax offset math; virtual-surface/upscale dimensions.
- **Determinism / isolation** — assert the **training path never imports the renderer**, keeping headless GA training untouched and fast. Real regression guard.
- **Smoke render** — one test builds a scene with *every* entity type, renders a frame, asserts no crash + correct dimensions.
- **Manual visual pass** — run + screenshot loop for by-eye tuning (grain, palette, juice intensity); a small FPS readout to confirm the 60fps budget empirically.
- Keep the existing **~420 tests green** throughout.

---

## 7. Performance notes

- Sprite **art resolution barely affects performance**: a blit is one Python→C SDL call; per-call overhead dominates, and sprites are **baked once and cached** (no per-frame pixel work).
- 2× vs 4× differs only in virtual-surface **fill rate** (230k vs 58k px) — trivial for pygame-ce/SDL at 60fps.
- Avoided pygame slow paths: no runtime per-pixel Python loops (bake+cache); cache ball rotation in angle buckets only if it shows hot; **capped** particle pool; scene has only tens of entities.
- Rendering runs **only in play/menu** scenes; **headless training never instantiates the renderer**.

---

## 8. Out of scope / future

- **Neon theme** — second `Theme` (new palette + glow + dark parallax). Enabled via `config.ACTIVE_THEME = "neon"`. Reuses sprite shapes where possible.
- **Retro post-FX** — scanline/CRT overlay + global glow. Deferred; revisit if desired.

---

## 9. Implementation notes

- Work happens in a **parallel git worktree** (per user request) branched from `master`.
- The theme switch is a single config flag — the deliverable includes the pixel theme fully wired and the neon slot stubbed.
- Tuning (grain feel, palette, juice intensity) is done live with the run + screenshot loop, not pre-specified.

See task list (#9–#16 + testing/wiring tasks) for the implementation breakdown; the writing-plans skill will expand these into a sequenced plan.
