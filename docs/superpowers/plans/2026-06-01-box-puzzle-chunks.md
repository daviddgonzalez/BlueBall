# Box-Puzzle Chunks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three pushable-box puzzle chunks — a long lava-gap crossing and two spring puzzles — reusing existing physics, and wire them into the hand-authored levels (lava gap also into the Infinite Run sampler).

**Architecture:** Each chunk is a `Chunk` subclass in `src/blueball/levels/chunks/`, registered via `@register_chunk`, materializing static segments (`world.space.add`) and reusing the `Lava`, `Spring`, and `PushableBox` entities (`world.add_entity`). Lava is made static by passing `rise_speed=0`. The streaming builder (`scenes/play.py::_materialize_chunk`) diff-captures every shape/body/entity a chunk adds, so chunk-spawned lava/boxes cull automatically.

**Tech Stack:** Python, pymunk-ce, pytest. Run tests with **`.venv/bin/python -m pytest`** (the system `python3` resolves `blueball` from a different worktree and fails).

---

## Background facts (verified in the codebase)

- `TILE = 32`. Ground line `GROUND_Y = 600` (in `chunks/flat.py`). pymunk is **y-down**: larger y = lower, "up" = smaller y.
- Chunk `build` signature used by both the loader (`build(world, x_offset=x)`) and the streaming builder (`build(world, x_offset=..., base_y=chunk_base)`). **New chunks must accept `base_y: float = GROUND_Y`.**
- `Lava(world, position, width, rise_speed, height=600)` — kinematic sensor (`CT_LAVA`), poly extends **down** from `position.y` (surface) by `height`. `on_lava` kills only the player; a `PushableBox` passes through unharmed. `rise_speed=0` → velocity `(0,0)` → static.
- `Spring(world, position, width, impulse)` — static sensor (`CT_SPRING`). `on_spring` sets any dynamic body's `vy = min(vy, -impulse)` on each begin-contact (so a box bounces at a consistent height).
- `PushableBox(world, position, size=32, mass=0.5)` — dynamic box.
- Sampler pool = chunks with `sampler_include == True`, weighted by `difficulty` vs. a ramping target.
- Patterns to follow: `chunks/door.py`, `chunks/pushable_box.py`, `chunks/spring.py`, `chunks/spike_wall.py`.

---

### Task 1: `box_lava_gap` chunk

**Goal:** A long, shallow lava pit with a solid floor; a box shoved in rests on the floor as a mid-pit stepping stone. In the Infinite Run sampler.

**Files:**
- Create: `src/blueball/levels/chunks/box_lava_gap.py`
- Modify: `src/blueball/levels/chunks/__init__.py` (register import)
- Test: `tests/test_chunks.py` (append)

**Acceptance Criteria:**
- [ ] `box_lava_gap` is in `CHUNK_REGISTRY`; `sampler_include is True`; `difficulty == 3`.
- [ ] `build` returns `(approach_tiles + pit_tiles + exit_tiles) * TILE` and adds exactly 5 static segments (approach ledge, exit ledge, near wall, far wall, pit floor), one `Lava` (sensor), one `PushableBox`.
- [ ] Safety invariant by construction: the resting box's top is strictly above (smaller y than) the lava surface.
- [ ] Physics smoke: a box nudged into the pit comes to rest on the pit floor and is **not** removed/destroyed by the lava sensor.
- [ ] `build` accepts `base_y` and places everything relative to it.

**Verify:** `.venv/bin/python -m pytest tests/test_chunks.py -k box_lava_gap -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append to `tests/test_chunks.py`)

```python
# ---------------------------------------------------------------------------
# box_lava_gap chunk
# ---------------------------------------------------------------------------

def test_box_lava_gap_in_registry_and_sampler():
    from blueball.levels.chunks.box_lava_gap import BoxLavaGap
    assert "box_lava_gap" in CHUNK_REGISTRY
    assert BoxLavaGap.sampler_include is True
    assert BoxLavaGap.difficulty == 3


def test_box_lava_gap_builds_segments_lava_and_box():
    from blueball.entities.lava import Lava
    from blueball.entities.pushable_box import PushableBox
    w = World()
    chunk = CHUNK_REGISTRY["box_lava_gap"](approach_tiles=2, pit_tiles=6, exit_tiles=2)
    width = chunk.build(w, x_offset=0.0)
    assert width == 10 * TILE
    segs = [s for s in w.space.shapes
            if isinstance(s, pymunk.Segment) and s.body is w.space.static_body]
    assert len(segs) == 5  # approach, exit, near wall, far wall, pit floor
    lavas = [e for e in w.entities if isinstance(e, Lava)]
    boxes = [e for e in w.entities if isinstance(e, PushableBox)]
    assert len(lavas) == 1 and len(boxes) == 1
    assert lavas[0].shape.sensor is True


def test_box_lava_gap_box_top_above_lava_surface():
    from blueball.entities.lava import Lava
    from blueball.entities.pushable_box import PushableBox
    w = World()
    CHUNK_REGISTRY["box_lava_gap"]().build(w, x_offset=0.0)
    lava = next(e for e in w.entities if isinstance(e, Lava))
    box = next(e for e in w.entities if isinstance(e, PushableBox))
    box_top = box.body.position.y - box.size / 2
    assert box_top < lava.position[1]  # smaller y = higher = above the lava


def test_box_lava_gap_box_rests_on_pit_floor():
    import blueball.collision as collision
    from blueball.entities.pushable_box import PushableBox
    w = World()
    collision.register(w.space, w)
    CHUNK_REGISTRY["box_lava_gap"](pit_tiles=6).build(w, x_offset=0.0)
    box = next(e for e in w.entities if isinstance(e, PushableBox))
    box.body.velocity = (200, 0)  # shove it into the pit
    for _ in range(360):
        w.step(1 / 120)
    assert box in w.entities                      # not destroyed by lava
    floor_y = 600 + 72                            # base_y + default depth
    assert box.body.position.y <= floor_y - box.size / 2 + 3
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_chunks.py -k box_lava_gap -v`
Expected: FAIL (`ModuleNotFoundError: blueball.levels.chunks.box_lava_gap` / not in registry)

- [ ] **Step 3: Create `src/blueball/levels/chunks/box_lava_gap.py`**

```python
"""box_lava_gap chunk — a long, shallow lava pit crossed by shoving a box in
as a mid-pit stepping stone.

The pit is too long to clear in one jump. The player pushes the PushableBox off
the near ledge; it drops to the solid pit floor (lava is a player-only sensor,
so the box is unharmed) and, with momentum across the low-friction floor,
settles near the middle, turning one long jump into two short ones:
    near ledge -> box top -> far ledge.
"""

from __future__ import annotations

import pymunk

from ...entities.lava import Lava
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

_PIT_DEPTH = 72       # pit floor sits this far below the ledges (px)
_BOX_SIZE = 40        # box edge length (px)
_BOX_MASS = 0.6
_LAVA_BELOW_BOX = 8   # lava surface sits this far below the resting box's top
_PIT_FLOOR_FRICTION = 0.1  # low friction so a firm push carries the box to mid


@register_chunk("box_lava_gap")
class BoxLavaGap(Chunk):
    sampler_include: bool = True
    difficulty: int = 3

    def __init__(
        self,
        approach_tiles: int = 2,
        pit_tiles: int = 6,
        exit_tiles: int = 2,
        depth: int = _PIT_DEPTH,
        box_size: int = _BOX_SIZE,
        box_mass: float = _BOX_MASS,
    ) -> None:
        self.approach_tiles = approach_tiles
        self.pit_tiles = pit_tiles
        self.exit_tiles = exit_tiles
        self.depth = depth
        self.box_size = box_size
        self.box_mass = box_mass

    @classmethod
    def random_params(cls, rng) -> dict:
        # Vary only the pit length (the difficulty knob); keep depth/box fixed
        # so the box-as-step geometry stays solvable.
        return {"pit_tiles": rng.randint(5, 7)}

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        ax = self.approach_tiles * TILE
        px = self.pit_tiles * TILE
        ex = self.exit_tiles * TILE
        total = ax + px + ex
        pit_left = x_offset + ax
        pit_right = pit_left + px
        floor_y = base_y + self.depth

        def seg(a, b, friction=1.0):
            s = pymunk.Segment(world.space.static_body, a, b, 5)
            s.friction = friction
            world.space.add(s)

        seg((x_offset, base_y), (pit_left, base_y))           # approach ledge
        seg((pit_right, base_y), (x_offset + total, base_y))  # exit ledge
        seg((pit_left, base_y), (pit_left, floor_y))          # near wall
        seg((pit_right, base_y), (pit_right, floor_y))        # far wall
        seg((pit_left, floor_y), (pit_right, floor_y), _PIT_FLOOR_FRICTION)  # floor

        box_rest_top = floor_y - self.box_size
        lava_surface = box_rest_top + _LAVA_BELOW_BOX
        world.add_entity(Lava(
            world,
            position=(pit_left + px / 2, lava_surface),
            width=px,
            rise_speed=0.0,
            height=self.depth,
        ))

        # Box starts on the approach ledge at the pit edge, ready to shove in.
        world.add_entity(PushableBox(
            world,
            position=(pit_left - self.box_size / 2 - 2, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))
        return total
```

- [ ] **Step 4: Register the chunk** — add `box_lava_gap` to the import tuple in `src/blueball/levels/chunks/__init__.py` (alongside `door`, `cannon_lane`, etc.):

```python
from . import (  # noqa: F401
    flat,
    # ... existing entries ...
    cannon_lane,
    box_lava_gap,
)
```

- [ ] **Step 5: Run the tests, confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_chunks.py -k box_lava_gap -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full suite** — `.venv/bin/python -m pytest -q` → expect all pass (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/levels/chunks/box_lava_gap.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat(chunks): box_lava_gap — push a box across a long lava pit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wire `box_lava_gap` into maze.json + streaming-cull test

**Goal:** Place a `box_lava_gap` in the hand-authored `maze.json`, and prove its chunk-spawned `Lava` + `PushableBox` cull correctly under Infinite Run streaming.

**Files:**
- Modify: `src/blueball/levels/maze.json`
- Test: `tests/test_play_scene.py` (append)

**Acceptance Criteria:**
- [ ] `maze.json` contains one `box_lava_gap` entry and still loads (level-load test passes).
- [ ] A `box_lava_gap` materialized in a streaming `PlayScene` has its `Lava` and `PushableBox` tracked in the chunk's `_built_chunks` record.
- [ ] After the player advances past it (cull), those exact `Lava`/`PushableBox` entities are removed from `world.entities` and their bodies from `world.space.bodies`.

**Verify:** `.venv/bin/python -m pytest tests/test_play_scene.py -k box_lava_gap -v` and `.venv/bin/python -m pytest tests/test_loader.py -q` → pass

**Steps:**

- [ ] **Step 1: Write the failing streaming-cull test** (append to `tests/test_play_scene.py`)

```python
def test_play_scene_culls_box_lava_gap_entities(headless_pygame, tmp_save):
    """A chunk-spawned static Lava + PushableBox must be diff-tracked and culled
    once the player slides far past the chunk."""
    from blueball.entities.lava import Lava
    from blueball.entities.pushable_box import PushableBox
    from blueball.levels.chunks.box_lava_gap import BoxLavaGap

    data = {
        "name": "Infinite", "background": "#202028", "ground": "#666c70",
        "spawn": [80, 540], "chunks": [],
    }
    scene = PlayScene(headless_pygame, level_data=data, sampler_seed=42)
    scene._materialize_chunk(BoxLavaGap(pit_tiles=6))
    rec = scene._built_chunks[-1]
    my_lava = next(e for e in rec["entities"] if isinstance(e, Lava))
    my_box = next(e for e in rec["entities"] if isinstance(e, PushableBox))
    assert my_lava in scene.world.entities
    assert my_box in scene.world.entities

    far = rec["x_end"] + 5000
    scene.player.body.position = (far, 540)
    scene._maintain_streaming(far)

    assert my_lava not in scene.world.entities
    assert my_box not in scene.world.entities
    assert my_lava.body not in scene.world.space.bodies
    assert my_box.body not in scene.world.space.bodies
```

- [ ] **Step 2: Run it, confirm it passes already** (the streaming builder is generic, so the new chunk should cull without code changes — this test guards that).

Run: `.venv/bin/python -m pytest tests/test_play_scene.py -k box_lava_gap -v`
Expected: PASS. If it FAILS, the chunk added something the diff-tracker missed — investigate before proceeding.

- [ ] **Step 3: Add `box_lava_gap` to `maze.json`** — replace the lone `pushable_box` line (currently `{"type": "pushable_box", "width_tiles": 3, "size_px": 32, "mass": 0.4}`) with the lava gap, so the box gets a real purpose:

```json
    {"type": "box_lava_gap", "approach_tiles": 2, "pit_tiles": 6, "exit_tiles": 2},
```

(Keep the surrounding `flat`/`spring` chunks; exact pit length is tuned in Task 6.)

- [ ] **Step 4: Confirm the level still loads** — run the loader test suite:

Run: `.venv/bin/python -m pytest tests/test_loader.py -q`
Expected: PASS. (If there is no maze-specific load test, add one mirroring the existing loader tests: `load_level(<maze.json path>, World())` returns a `LevelMeta` without raising.)

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/levels/maze.json tests/test_play_scene.py
git commit -m "feat(levels): use box_lava_gap in maze; verify streaming cull

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `box_spring_trampoline` chunk

**Goal:** A spring + a box on the ground + a high exit ledge unreachable without bouncing off the box. Hand-authored only (`sampler_include = False`).

**Files:**
- Create: `src/blueball/levels/chunks/box_spring_trampoline.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Test: `tests/test_chunks.py` (append)

**Acceptance Criteria:**
- [ ] In `CHUNK_REGISTRY`; `sampler_include is False`; `difficulty == 4`.
- [ ] `build` adds a ground segment + a high exit-ledge segment, one `Spring`, one `PushableBox`; returns `width_tiles * TILE`; accepts `base_y`.
- [ ] Physics smoke: a box placed on the spring gains upward velocity (`vy <= -impulse`) within a few steps (confirms the bounce drives the box).

**Verify:** `.venv/bin/python -m pytest tests/test_chunks.py -k box_spring_trampoline -v` → pass

**Steps:**

- [ ] **Step 1: Write failing tests** (append to `tests/test_chunks.py`)

```python
# ---------------------------------------------------------------------------
# box_spring_trampoline chunk
# ---------------------------------------------------------------------------

def test_box_spring_trampoline_registry_and_flags():
    from blueball.levels.chunks.box_spring_trampoline import BoxSpringTrampoline
    assert "box_spring_trampoline" in CHUNK_REGISTRY
    assert BoxSpringTrampoline.sampler_include is False
    assert BoxSpringTrampoline.difficulty == 4


def test_box_spring_trampoline_builds_spring_box_and_exit():
    from blueball.entities.spring import Spring
    from blueball.entities.pushable_box import PushableBox
    w = World()
    width = CHUNK_REGISTRY["box_spring_trampoline"](width_tiles=6).build(w, x_offset=0.0)
    assert width == 6 * TILE
    assert len([e for e in w.entities if isinstance(e, Spring)]) == 1
    assert len([e for e in w.entities if isinstance(e, PushableBox)]) == 1
    segs = [s for s in w.space.shapes
            if isinstance(s, pymunk.Segment) and s.body is w.space.static_body]
    assert len(segs) == 2  # ground + exit ledge


def test_box_spring_trampoline_box_launched_by_spring():
    import blueball.collision as collision
    from blueball.entities.spring import Spring
    from blueball.entities.pushable_box import PushableBox
    w = World()
    collision.register(w.space, w)
    CHUNK_REGISTRY["box_spring_trampoline"](width_tiles=6, impulse=720.0).build(w, x_offset=0.0)
    spring = next(e for e in w.entities if isinstance(e, Spring))
    box = next(e for e in w.entities if isinstance(e, PushableBox))
    # Drop the box onto the spring.
    box.body.position = (spring.position[0], spring.position[1] - box.size)
    launched = False
    for _ in range(240):
        w.step(1 / 120)
        if box.body.velocity.y <= -700:  # ~ -impulse upward (y-down)
            launched = True
            break
    assert launched
```

- [ ] **Step 2: Run, confirm fail.** `.venv/bin/python -m pytest tests/test_chunks.py -k box_spring_trampoline -v` → FAIL (module missing).

- [ ] **Step 3: Create `src/blueball/levels/chunks/box_spring_trampoline.py`**

```python
"""box_spring_trampoline chunk — push a box onto a spring, then bounce off the
rising box to reach a high exit ledge that's out of reach otherwise.

No catch ledge: the box never lands as a permanent step. The bounce itself is
the only way up. Exact heights are tuned by playtest.
"""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.spring import Spring
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("box_spring_trampoline")
class BoxSpringTrampoline(Chunk):
    sampler_include: bool = False
    difficulty: int = 4

    def __init__(
        self,
        width_tiles: int = 6,
        impulse: float = 720.0,
        box_size: int = 36,
        box_mass: float = 0.5,
        exit_height: int = 220,   # exit ledge px above base_y (playtest-tuned)
        exit_tiles: int = 2,
    ) -> None:
        self.width_tiles = width_tiles
        self.impulse = impulse
        self.box_size = box_size
        self.box_mass = box_mass
        self.exit_height = exit_height
        self.exit_tiles = exit_tiles

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        ground = pymunk.Segment(
            world.space.static_body, (x_offset, base_y), (x_offset + w, base_y), 5
        )
        ground.friction = 1.0
        world.space.add(ground)

        spring_cx = x_offset + w * 0.62
        world.add_entity(Spring(
            world, position=(spring_cx, base_y - 8), width=2 * TILE, impulse=self.impulse
        ))

        # Box starts left of the spring, ready to push onto it.
        world.add_entity(PushableBox(
            world,
            position=(x_offset + w * 0.25, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))

        # High exit ledge above the spring, out of normal-jump reach.
        ledge_y = base_y - self.exit_height
        ledge = pymunk.Segment(
            world.space.static_body,
            (x_offset + w - self.exit_tiles * TILE, ledge_y),
            (x_offset + w, ledge_y),
            5,
        )
        ledge.friction = 1.0
        world.space.add(ledge)
        return w
```

- [ ] **Step 4: Register** — add `box_spring_trampoline` to `chunks/__init__.py` import tuple.

- [ ] **Step 5: Run tests, confirm pass.** `.venv/bin/python -m pytest tests/test_chunks.py -k box_spring_trampoline -v` → PASS (3 tests).

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q` → all pass.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/levels/chunks/box_spring_trampoline.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat(chunks): box_spring_trampoline — bounce off a box on a spring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `box_spring_relay` chunk

**Goal:** Two springs (one on the ground, one on a raised platform) + guide walls; the box relays from spring 1 up to spring 2 to reach a higher exit. Hand-authored only.

**Files:**
- Create: `src/blueball/levels/chunks/box_spring_relay.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Test: `tests/test_chunks.py` (append)

**Acceptance Criteria:**
- [ ] In `CHUNK_REGISTRY`; `sampler_include is False`; `difficulty == 5`.
- [ ] `build` adds ground + raised platform + exit-ledge segments + two guide-wall segments, **two** `Spring`s, one `PushableBox`; returns `width_tiles * TILE`; accepts `base_y`.
- [ ] Spring 2 sits higher (smaller y) than spring 1.

**Verify:** `.venv/bin/python -m pytest tests/test_chunks.py -k box_spring_relay -v` → pass

**Steps:**

- [ ] **Step 1: Write failing tests** (append to `tests/test_chunks.py`)

```python
# ---------------------------------------------------------------------------
# box_spring_relay chunk
# ---------------------------------------------------------------------------

def test_box_spring_relay_registry_and_flags():
    from blueball.levels.chunks.box_spring_relay import BoxSpringRelay
    assert "box_spring_relay" in CHUNK_REGISTRY
    assert BoxSpringRelay.sampler_include is False
    assert BoxSpringRelay.difficulty == 5


def test_box_spring_relay_builds_two_springs_and_box():
    from blueball.entities.spring import Spring
    from blueball.entities.pushable_box import PushableBox
    w = World()
    width = CHUNK_REGISTRY["box_spring_relay"](width_tiles=8).build(w, x_offset=0.0)
    assert width == 8 * TILE
    springs = [e for e in w.entities if isinstance(e, Spring)]
    boxes = [e for e in w.entities if isinstance(e, PushableBox)]
    assert len(springs) == 2 and len(boxes) == 1
    # Spring 2 is higher (smaller y) than spring 1.
    ys = sorted(s.position[1] for s in springs)
    assert ys[0] < ys[1]


def test_box_spring_relay_has_guide_walls_and_platform():
    w = World()
    CHUNK_REGISTRY["box_spring_relay"](width_tiles=8).build(w, x_offset=0.0)
    segs = [s for s in w.space.shapes
            if isinstance(s, pymunk.Segment) and s.body is w.space.static_body]
    # ground + raised platform + exit ledge + 2 guide walls = 5
    assert len(segs) == 5
    verticals = [s for s in segs if abs(s.a.x - s.b.x) < 1e-6]
    assert len(verticals) == 2  # two guide walls
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Create `src/blueball/levels/chunks/box_spring_relay.py`**

```python
"""box_spring_relay chunk — push a box onto a ground spring; it arcs onto a
second spring on a raised platform, which relaunches it higher, to reach an
exit above the second spring. Guide walls constrain the box's arc.

The horizontal arc depends on the box's launch velocity; exact geometry is
tuned by playtest (see plan Task 6). If it proves unsolvable after tuning,
fall back to a trampoline variant.
"""

from __future__ import annotations

import pymunk

from ...entities.spring import Spring
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("box_spring_relay")
class BoxSpringRelay(Chunk):
    sampler_include: bool = False
    difficulty: int = 5

    def __init__(
        self,
        width_tiles: int = 8,
        impulse1: float = 720.0,
        impulse2: float = 760.0,
        platform_height: int = 200,   # spring-2 platform px above base_y
        relay_dx_tiles: int = 4,      # horizontal offset of spring 2
        platform_tiles: int = 2,
        box_size: int = 36,
        box_mass: float = 0.5,
        exit_height: int = 360,       # exit ledge px above base_y (tuned)
        exit_tiles: int = 2,
        wall_height: int = 240,       # guide-wall height (tuned)
    ) -> None:
        self.width_tiles = width_tiles
        self.impulse1 = impulse1
        self.impulse2 = impulse2
        self.platform_height = platform_height
        self.relay_dx_tiles = relay_dx_tiles
        self.platform_tiles = platform_tiles
        self.box_size = box_size
        self.box_mass = box_mass
        self.exit_height = exit_height
        self.exit_tiles = exit_tiles
        self.wall_height = wall_height

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE

        def seg(a, b):
            s = pymunk.Segment(world.space.static_body, a, b, 5)
            s.friction = 1.0
            world.space.add(s)

        # Ground.
        seg((x_offset, base_y), (x_offset + w, base_y))

        # Spring 1 near the left, on the ground.
        s1_cx = x_offset + 1.5 * TILE
        world.add_entity(Spring(
            world, position=(s1_cx, base_y - 8), width=2 * TILE, impulse=self.impulse1
        ))

        # Raised platform carrying spring 2.
        plat_y = base_y - self.platform_height
        plat_left = s1_cx + self.relay_dx_tiles * TILE - (self.platform_tiles * TILE) / 2
        plat_right = plat_left + self.platform_tiles * TILE
        seg((plat_left, plat_y), (plat_right, plat_y))
        s2_cx = (plat_left + plat_right) / 2
        world.add_entity(Spring(
            world, position=(s2_cx, plat_y - 8), width=self.platform_tiles * TILE,
            impulse=self.impulse2,
        ))

        # Two guide walls to keep the box's arc on course (left of spring 1 and
        # right of the platform), so a too-hard push doesn't fling it offscreen.
        seg((x_offset, base_y), (x_offset, base_y - self.wall_height))
        seg((x_offset + w, base_y), (x_offset + w, base_y - self.wall_height))

        # Box starts just right of spring 1, ready to shove onto it.
        world.add_entity(PushableBox(
            world,
            position=(s1_cx + TILE, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))

        # Exit ledge above spring 2.
        ledge_y = base_y - self.exit_height
        seg((s2_cx - self.exit_tiles * TILE / 2, ledge_y),
            (s2_cx + self.exit_tiles * TILE / 2, ledge_y))
        return w
```

Note: the exit-ledge `seg` is the 3rd horizontal segment; ground + platform + exit = 3 horizontals, plus 2 vertical guide walls = 5 total (matches the test).

- [ ] **Step 4: Register** — add `box_spring_relay` to `chunks/__init__.py`.

- [ ] **Step 5: Run tests, confirm pass.** `.venv/bin/python -m pytest tests/test_chunks.py -k box_spring_relay -v` → PASS (3 tests).

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q` → all pass.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/levels/chunks/box_spring_relay.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat(chunks): box_spring_relay — relay a box across two springs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire spring puzzles into vertical_climb.json

**Goal:** Add `box_spring_trampoline` and `box_spring_relay` chunk entries to the vertical level so the spring puzzles appear in real play.

**Files:**
- Modify: `src/blueball/levels/vertical_climb.json`
- Test: `tests/test_loader.py` (add a load assertion if none covers vertical_climb)

**Acceptance Criteria:**
- [ ] `vertical_climb.json` `chunks` array includes one `box_spring_trampoline` and one `box_spring_relay` entry.
- [ ] The level loads via `load_level` without raising (loader test passes).

**Verify:** `.venv/bin/python -m pytest tests/test_loader.py -q` → pass

**Steps:**

- [ ] **Step 1: Edit `vertical_climb.json`** — insert the two puzzles into the `chunks` array between the intro `flat` and the big `vertical_column` (so the player meets a box puzzle before the long climb). New `chunks` array:

```json
  "chunks": [
    {"type": "flat", "width_tiles": 3},
    {"type": "box_spring_trampoline", "width_tiles": 6, "impulse": 720, "exit_height": 220},
    {"type": "flat", "width_tiles": 2},
    {"type": "box_spring_relay", "width_tiles": 8, "platform_height": 200, "exit_height": 360},
    {"type": "flat", "width_tiles": 2},
    {"type": "vertical_column", "width_tiles": 6, "steps": 32, "step_height": 86, "bottom_offset": 64, "platform_tiles": 2, "pattern": [0, 4, [1, 4], 2, 4, [0, 3], 1, 4]},
    {"type": "goal", "width_tiles": 2, "y_offset": 2730}
  ]
```

(Exact `exit_height` / `platform_height` are tuned in Task 6.)

- [ ] **Step 2: Confirm it loads** — if `tests/test_loader.py` has no vertical_climb case, add:

```python
def test_vertical_climb_level_loads():
    from pathlib import Path
    import blueball
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(blueball.__file__).parent / "levels" / "vertical_climb.json"
    meta = load_level(path, World())
    assert meta.name == "Vertical Climb"
```

Run: `.venv/bin/python -m pytest tests/test_loader.py -q` → PASS

- [ ] **Step 3: Full suite** — `.venv/bin/python -m pytest -q` → all pass.

- [ ] **Step 4: Commit**

```bash
git add src/blueball/levels/vertical_climb.json tests/test_loader.py
git commit -m "feat(levels): add box spring puzzles to vertical_climb

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Playtest tuning pass (solvability + feel)

**Goal:** Run the levels via the `/run` skill and tune each chunk's constants so all three puzzles are reliably solvable by a human and feel fair.

**Files:**
- Modify (tuning only): `src/blueball/levels/chunks/box_lava_gap.py`, `box_spring_trampoline.py`, `box_spring_relay.py`, `src/blueball/levels/maze.json`, `src/blueball/levels/vertical_climb.json`

**Acceptance Criteria:**
- [ ] `box_lava_gap`: the pit cannot be cleared in a single jump, but a box shoved off the near ledge reliably settles where it serves as a stepping stone to cross; falling in is fatal.
- [ ] `box_spring_trampoline`: the exit ledge is unreachable by spring-alone/normal jump, but bouncing off the box reliably reaches it.
- [ ] `box_spring_relay`: the box relays spring 1 → spring 2 and the exit is reachable; **OR**, if unsolvable after a reasonable tuning effort, it is simplified to a trampoline-style variant and that decision is noted in the commit.
- [ ] Full suite still green after any constant changes.

**Verify:** Manual via `/run` (launch maze.json and vertical_climb.json, attempt each puzzle); then `.venv/bin/python -m pytest -q` → all pass.

**Steps:**

- [ ] **Step 1: Launch and observe** — use the `/run` skill to start the app on `maze.json`, reach the `box_lava_gap`, and confirm: (a) the gap is not single-jumpable, (b) shoving the box in creates a usable mid-pit step, (c) touching lava kills.

- [ ] **Step 2: Tune the lava gap** — adjust `pit_tiles` (length vs. single jump), `_PIT_DEPTH` / `_BOX_SIZE` (step reachability), `_PIT_FLOOR_FRICTION` (how far a push carries the box). Keep the box-top-above-lava invariant (Task 1 test must stay green).

- [ ] **Step 3: Launch vertical_climb.json** and attempt both spring puzzles. Tune `impulse`/`exit_height`/`box_mass` (trampoline) and `impulse1/impulse2`/`platform_height`/`relay_dx_tiles`/`wall_height`/`exit_height` (relay) until each is reliably solvable.

- [ ] **Step 4: Re-run the full suite** — `.venv/bin/python -m pytest -q` → all pass. Fix any structural test whose hardcoded expectation (e.g. default depth in `test_box_lava_gap_box_rests_on_pit_floor`) drifted from a tuned default.

- [ ] **Step 5: Commit the tuned values**

```bash
git add -A
git commit -m "tune(chunks): playtest-tune box puzzle geometry for solvability

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** box_lava_gap (Tasks 1–2, sampler + maze + cull), box_spring_trampoline (Task 3), box_spring_relay (Task 4), vertical_climb integration (Task 5), playtest validation of #2/#3 (Task 6), streaming/cull (Task 2), structural + physics tests (Tasks 1–4). The level-rebalance aside is explicitly out of scope (its own future spec). ✓
- **No new collision handlers / no ai/ touched.** ✓
- **Types/names consistent:** `Lava(... rise_speed, height)`, `Spring(... impulse)`, `PushableBox(... size, mass)`, `build(world, x_offset, base_y=GROUND_Y)`, `sampler_include`, `difficulty` — all match the verified code. ✓
- **Risk recorded:** Infinite-Run idle-stall for box_lava_gap (spec); relay fragility with a documented fallback (Task 4 + Task 6). ✓
