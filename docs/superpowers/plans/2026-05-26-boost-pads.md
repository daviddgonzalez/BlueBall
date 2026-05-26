# Boost Pads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship boost pads — floor-strip sensors that raise the player's max-linear-speed cap (and matching max-angular-velocity cap) until the next time the player transitions from airborne to grounded. Multiple pads stack take-the-max.

**Architecture:** A new `BoostPad` sensor entity sits on the ground; on contact the collision dispatcher calls `player.receive_boost(multiplier)`. Player gains two new fields (`_boost_multiplier`, `_aerial_since_pickup`), a `receive_boost(m)` method (take-the-max), and a per-tick `_update_boost(grounded)` that clears the boost once the player has been airborne and lands again. The existing speed-cap and angular-cap reads in `Player.update()` multiply by `_boost_multiplier`. A new chunk type + renderer method + tutorial-level wiring complete the slice.

**Tech Stack:** Python 3.11+, PyGame-ce, Pymunk, pytest. No new third-party dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-23-boost-pads-design.md` (approved during v1 feel-tuning; sat in the bank during the abilities-framework slice; now activated).

---

## File structure

Final layout after this plan lands. New files marked `+`, modified files marked `~`.

```
src/blueball/
├── config.py                              (~ BOOST_PAD_THICKNESS, BOOST_PAD_DEFAULT_MULTIPLIER)
├── collision.py                           (~ CT_BOOST_PAD=6, on_boost_pad)
├── entities/
│   ├── player.py                          (~ _boost_multiplier, receive_boost, _update_boost, effective caps)
│   └── boost_pad.py                       (+ sensor strip entity)
├── levels/
│   ├── tutorial_hill.json                 (~ insert boost_pad between patrol_platform and stairs_up)
│   └── chunks/
│       ├── __init__.py                    (~ import boost_pad)
│       └── boost_pad.py                   (+ BoostPadChunk)
└── render/renderer.py                     (~ draw_boost_pad, _BOOST_PAD_* colors)

tests/
├── test_player.py                         (~ 4 receive_boost / cap tests)
├── test_entities.py                       (~ 2 BoostPad tests)
├── test_chunks.py                         (~ 2 boost_pad chunk tests)
└── test_collision.py                      (~ 1 contact-gives-boost test)
```

---

## Task 0: Player boost state + receive_boost + effective caps

**Goal:** `Player` accepts boost via `receive_boost(multiplier)`, take-the-max'd against any active boost. Boost ends on the first airborne→grounded transition after pickup. While active, the linear-speed cap and angular-velocity cap are both multiplied by the boost multiplier so the ball rolls faster without visual slip.

**Files:**
- Modify: `src/blueball/entities/player.py`
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] `Player.__init__` initializes `self._boost_multiplier = 1.0` and `self._aerial_since_pickup = False`.
- [ ] `player.receive_boost(2.0)` sets `_boost_multiplier = 2.0`. Calling `receive_boost(1.5)` after that is a no-op (take-the-max). Calling `receive_boost(3.0)` raises it to 3.0.
- [ ] At `receive_boost(m)` time, `_aerial_since_pickup` is set to `not self.grounded` — so a pickup while grounded starts in the "needs to leave the ground first" state.
- [ ] A per-tick helper `_update_boost(grounded)` is called from `update()` after `_refresh_contact_normals()`. While `_boost_multiplier > 1.0`: airborne ticks set `_aerial_since_pickup = True`; the first grounded tick with `_aerial_since_pickup == True` clears the boost (`_boost_multiplier = 1.0`, `_aerial_since_pickup = False`).
- [ ] In `Player.update()`, the linear-velocity-magnitude cap uses `config.MAX_LINEAR_SPEED * self._boost_multiplier`, and the angular-velocity cap uses `config.MAX_ANGULAR_VEL * self._boost_multiplier`. When `_boost_multiplier == 1.0` the behavior is identical to today.

**Verify:** `pytest -q tests/test_player.py -v` → all pass (existing tests + 4 new).

**Steps:**

- [ ] **Step 1: Add 4 failing tests to `tests/test_player.py`**

Append:

```python
def test_player_receive_boost_raises_multiplier():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p._boost_multiplier == 1.0
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0


def test_player_receive_boost_takes_max():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.receive_boost(1.5)
    assert p._boost_multiplier == 1.5
    p.receive_boost(1.2)
    assert p._boost_multiplier == 1.5  # take-the-max: weaker is no-op
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0  # stronger replaces


def test_player_boost_clears_on_air_to_ground_transition():
    """receive_boost while grounded; then airborne (sets _aerial_since_pickup);
    then grounded again -> boost clears."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.receive_boost(2.0)
    # Manually drive _update_boost: airborne tick, then grounded tick
    p._update_boost(grounded=False)
    assert p._boost_multiplier == 2.0  # still active in air
    assert p._aerial_since_pickup is True
    p._update_boost(grounded=True)
    assert p._boost_multiplier == 1.0  # cleared on landing
    assert p._aerial_since_pickup is False


def test_player_boost_persists_while_grounded_until_jump_land_cycle():
    """Pickup while grounded: boost stays active indefinitely on grounded
    ticks, only clears after a full air-then-ground cycle."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.receive_boost(2.0)
    # Several grounded ticks - boost should persist (never went airborne)
    for _ in range(5):
        p._update_boost(grounded=True)
    assert p._boost_multiplier == 2.0
    # Now jump (airborne)
    p._update_boost(grounded=False)
    assert p._boost_multiplier == 2.0
    assert p._aerial_since_pickup is True
    # Land - boost clears
    p._update_boost(grounded=True)
    assert p._boost_multiplier == 1.0
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_player.py -v`
Expected: 4 new failures (`AttributeError: 'Player' object has no attribute '_boost_multiplier'`).

- [ ] **Step 3: Add the new state, method, and helper to `src/blueball/entities/player.py`**

In `__init__`, just before `self._contact_normals = []`, add:

```python
        # Boost state. _boost_multiplier scales the linear-speed and angular-
        # velocity caps. _aerial_since_pickup tracks whether the player has
        # left the ground since the most recent receive_boost; the boost ends
        # on the first grounded tick after that flag becomes True.
        self._boost_multiplier: float = 1.0
        self._aerial_since_pickup: bool = False
```

Add two methods anywhere below `unlock()`:

```python
    def receive_boost(self, multiplier: float) -> None:
        """Apply a boost multiplier; take-the-max against any active boost.
        Weaker boosts arriving while a stronger one is active are no-ops.
        """
        if multiplier > self._boost_multiplier:
            self._boost_multiplier = multiplier
            self._aerial_since_pickup = not self.grounded

    def _update_boost(self, grounded: bool) -> None:
        """Per-tick boost decay. Boost ends on the first grounded tick after
        the player has been airborne since pickup."""
        if self._boost_multiplier <= 1.0:
            return
        if not grounded:
            self._aerial_since_pickup = True
        elif self._aerial_since_pickup:
            self._boost_multiplier = 1.0
            self._aerial_since_pickup = False
```

In `Player.update()`, add a call to `_update_boost` right after `self._refresh_contact_normals()` and BEFORE the `observation = self._observe()` line:

```python
        self._refresh_contact_normals()
        self._update_boost(self.grounded)
```

Then change the angular-velocity cap block from:
```python
        if av > config.MAX_ANGULAR_VEL:
            self.body.angular_velocity = config.MAX_ANGULAR_VEL
        elif av < -config.MAX_ANGULAR_VEL:
            self.body.angular_velocity = -config.MAX_ANGULAR_VEL
```
to:
```python
        max_ang = config.MAX_ANGULAR_VEL * self._boost_multiplier
        if av > max_ang:
            self.body.angular_velocity = max_ang
        elif av < -max_ang:
            self.body.angular_velocity = -max_ang
```

And change the linear-speed cap block from:
```python
        if speed > config.MAX_LINEAR_SPEED:
            scale = config.MAX_LINEAR_SPEED / speed
            self.body.velocity = (v.x * scale, v.y * scale)
```
to:
```python
        max_speed = config.MAX_LINEAR_SPEED * self._boost_multiplier
        if speed > max_speed:
            scale = max_speed / speed
            self.body.velocity = (v.x * scale, v.y * scale)
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_player.py -v`
Expected: all pass (was 12, now 16).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all green (65 + 4 = 69).

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: boost-pad state and effective caps on Player"
```

---

## Task 1: BoostPad entity + collision handler

**Goal:** A new sensor entity `BoostPad` plus a registered `CT_PLAYER ↔ CT_BOOST_PAD` begin handler that calls `player.receive_boost(pad.multiplier)`. Mirrors the `AbilityPickup` / `Collectible` collision pattern.

**Files:**
- Create: `src/blueball/entities/boost_pad.py`
- Modify: `src/blueball/collision.py`
- Modify: `tests/test_entities.py`
- Modify: `tests/test_collision.py`

**Acceptance Criteria:**
- [ ] `BoostPad(world, position, width=128, multiplier=2.0)` creates a STATIC body with a sensor `Poly` rectangle (`width` × `BOOST_PAD_THICKNESS=16` px), `collision_type == CT_BOOST_PAD`.
- [ ] `collision.CT_BOOST_PAD == 6`.
- [ ] On a `CT_PLAYER ↔ CT_BOOST_PAD` begin contact, `player.receive_boost(pad.multiplier)` is called. Handler returns `False` (sensor — no physical response).
- [ ] The pad does NOT consume itself on contact (unlike `AbilityPickup`). It remains in the space and can fire again after the player leaves and re-enters.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_entities.py`**

Append:

```python
from blueball.entities.boost_pad import BoostPad


def test_boost_pad_is_sensor_with_correct_collision_type():
    w = World()
    pad = BoostPad(w, position=(100, 200), width=128, multiplier=2.0)
    w.add_entity(pad)
    assert pad.shapes[0].sensor is True
    assert pad.shapes[0].collision_type == collision.CT_BOOST_PAD


def test_boost_pad_stores_multiplier_and_width():
    pad = BoostPad(World(), position=(50, 50), width=192, multiplier=1.7)
    assert pad.multiplier == 1.7
    assert pad.width == 192
```

- [ ] **Step 2: Add failing test to `tests/test_collision.py`**

Append:

```python
def test_player_receives_boost_on_pad_contact():
    from blueball.entities.boost_pad import BoostPad
    w, p = _player_world()
    # Place a boost pad overlapping the player position so contact is immediate
    pad = BoostPad(w, position=(100, 100), width=64, multiplier=2.0)
    w.add_entity(pad)

    for _ in range(5):
        w.step(1 / 60)
        if p._boost_multiplier > 1.0:
            break
    assert p._boost_multiplier == 2.0
    # Pad must still be present in the space (not consumed)
    assert pad.shapes[0] in w.space.shapes
```

- [ ] **Step 3: Run, confirm failure**

Run: `pytest -q tests/test_entities.py tests/test_collision.py -v`
Expected: import errors then test failures.

- [ ] **Step 4: Implement `src/blueball/entities/boost_pad.py`**

```python
"""BoostPad — a floor-strip sensor that raises the player's speed cap.

The pad is a static-body sensor rectangle. On contact the collision dispatcher
reads `pad.multiplier` and calls `player.receive_boost(m)`. The pad is NOT
consumed and can fire again the next time the player enters its volume.
"""

from __future__ import annotations

import pymunk

from ..collision import CT_BOOST_PAD
from .base import Entity


BOOST_PAD_THICKNESS = 16  # px — how thick the floor strip is in world units


class BoostPad(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        width: int = 128,
        multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self._world = world
        self.position = position
        self.width = width
        self.multiplier = multiplier

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        hw = width / 2
        hh = BOOST_PAD_THICKNESS / 2
        shape = pymunk.Poly(self.body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        shape.sensor = True
        shape.collision_type = CT_BOOST_PAD
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_boost_pad(self.position, self.width)
```

- [ ] **Step 5: Modify `src/blueball/collision.py`** — add the constant and handler.

Add the constant alongside the others (numerically between `CT_GOAL=5` and `CT_ABILITY_PICKUP=7`):

```python
CT_BOOST_PAD = 6
```

Inside `register(space, world_ref)`, add the handler near the other sensor handlers:

```python
    def on_boost_pad(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if not hasattr(entity, "multiplier"):
                continue
            if player is not None:
                player.receive_boost(entity.multiplier)
        return False  # sensor — no physical response
```

Register it alongside the other `space.on_collision(...)` calls:

```python
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_BOOST_PAD, begin=on_boost_pad)
```

- [ ] **Step 6: Run targeted tests, confirm pass**

Run: `pytest -q tests/test_entities.py tests/test_collision.py -v`
Expected: all pass.

- [ ] **Step 7: Run full suite**

Run: `pytest -q`
Expected: all green (was 69, now 72).

- [ ] **Step 8: Commit**

```bash
git add src/blueball/entities/boost_pad.py src/blueball/collision.py tests/test_entities.py tests/test_collision.py
git commit -m "feat: BoostPad entity and collision handler"
```

---

## Task 2: `boost_pad` chunk

**Goal:** A new chunk type `boost_pad` lays a flat ground segment with a `BoostPad` sensor sitting flush at ground level. Registered in the chunk registry so level JSON can reference it.

**Files:**
- Create: `src/blueball/levels/chunks/boost_pad.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `"boost_pad"` is in `CHUNK_REGISTRY` after importing the chunks package.
- [ ] Building a `boost_pad` chunk adds exactly one `BoostPad` entity to the world; pad's `multiplier` matches the JSON-supplied value (defaults to `config.BOOST_PAD_DEFAULT_MULTIPLIER`).
- [ ] The chunk's reported width equals `width_tiles * TILE`.
- [ ] The pad's width matches the segment width so the visual and sensor span the same ground.

**Verify:** `pytest -q tests/test_chunks.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_chunks.py`**

```python
from blueball.entities.boost_pad import BoostPad


def test_boost_pad_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "boost_pad" in CHUNK_REGISTRY


def test_boost_pad_chunk_adds_one_boost_pad_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    w = World()
    chunk = CHUNK_REGISTRY["boost_pad"](width_tiles=3, multiplier=2.5)
    width = chunk.build(w, x_offset=100.0)
    assert width == 3 * TILE
    pads = [e for e in w.entities if isinstance(e, BoostPad)]
    assert len(pads) == 1
    assert pads[0].multiplier == 2.5
    # Pad width should match the segment span
    assert pads[0].width == 3 * TILE
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_chunks.py -v`
Expected: `KeyError: 'boost_pad'`.

- [ ] **Step 3: Implement `src/blueball/levels/chunks/boost_pad.py`**

```python
"""boost_pad chunk — a flat ground segment with a BoostPad sensor on top."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.boost_pad import BoostPad
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("boost_pad")
class BoostPadChunk(Chunk):
    def __init__(
        self,
        width_tiles: int = 4,
        multiplier: float = config.BOOST_PAD_DEFAULT_MULTIPLIER,
    ) -> None:
        self.width_tiles = width_tiles
        self.multiplier = multiplier

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        # Ground segment under the pad
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, GROUND_Y),
            (x_offset + w, GROUND_Y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        # Pad sits flush at the top of the ground so the ball rolls over it.
        # Center the sensor on the segment; the half-thickness lift puts the
        # pad's top edge at GROUND_Y.
        world.add_entity(BoostPad(
            world,
            position=(x_offset + w / 2, GROUND_Y - config.BOOST_PAD_THICKNESS / 2),
            width=w,
            multiplier=self.multiplier,
        ))
        return w
```

- [ ] **Step 4: Register the chunk**

Edit `src/blueball/levels/chunks/__init__.py`:

```python
"""Importing this package registers every chunk type."""

from . import (  # noqa: F401
    flat,
    gap,
    spike_pit,
    patrol_platform,
    stairs,
    bump,
    falling_hazard,
    goal,
    ability_pickup,
    boost_pad,
)
```

(Reformatted to multi-line for readability now that there are 10 entries; same imports as before plus `boost_pad`.)

- [ ] **Step 5: Run chunk tests**

Run: `pytest -q tests/test_chunks.py -v`
Expected: all pass.

- [ ] **Step 6: Run full suite**

Run: `pytest -q`
Expected: all green (was 72, now 74).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/levels/chunks/boost_pad.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat: boost_pad chunk type"
```

---

## Task 3: Renderer + config + tutorial level + manual playtest

**Goal:** Wire the boost-pad visual into the renderer, add the config tunable, place a boost-pad chunk into the tutorial level, and verify end-to-end by playing the game.

**Files:**
- Modify: `src/blueball/render/renderer.py`
- Modify: `src/blueball/config.py`
- Modify: `src/blueball/levels/tutorial_hill.json`

**Acceptance Criteria:**
- [ ] `Renderer.draw_boost_pad(pos, width)` draws a flat cyan strip with a forward-pointing chevron at the center.
- [ ] `config.BOOST_PAD_DEFAULT_MULTIPLIER == 2.0`.
- [ ] `tutorial_hill.json` has a `boost_pad` chunk between the `patrol_platform` and the `stairs_up` (per the spec placement).
- [ ] `pytest -q` is fully green.
- [ ] Manual playtest with `BLUEBALL_SAVE_PATH=/tmp/...` shows: rolling over the cyan strip the player visibly accelerates (top speed roughly doubles), the speed boost lasts only until landing after the next jump, and a second pass over the pad re-applies the boost.

**Verify:** `pytest -q` → all green; then `python main.py` and observe.

**Steps:**

- [ ] **Step 1: Add the config constant to `src/blueball/config.py`**

Append (under the existing `# Abilities` section or in a new `# Boost pads` section):

```python
# Boost pads
BOOST_PAD_DEFAULT_MULTIPLIER = 2.0
```

- [ ] **Step 2: Add renderer constants and method to `src/blueball/render/renderer.py`**

Add module-level color constants alongside the existing ones (near `_ABILITY_PICKUP_COLORS`):

```python
_BOOST_PAD_COLOR = (80, 220, 240)   # cyan
_BOOST_PAD_EDGE = (30, 150, 180)    # deeper cyan
```

Add the method to the `Renderer` class:

```python
    def draw_boost_pad(self, pos, width) -> None:
        # Flat cyan strip with a forward-pointing chevron at the center.
        x, y = pos
        hw = width / 2
        pad_h = 8
        p1 = self._w2s((x - hw, y - pad_h))
        p2 = self._w2s((x + hw, y - pad_h))
        p3 = self._w2s((x + hw, y + pad_h))
        p4 = self._w2s((x - hw, y + pad_h))
        pygame.draw.polygon(self.screen, _BOOST_PAD_COLOR, [p1, p2, p3, p4])
        cx, cy = self._w2s((x, y))
        pygame.draw.polygon(
            self.screen,
            _BOOST_PAD_EDGE,
            [(cx - 8, cy - 6), (cx + 6, cy), (cx - 8, cy + 6)],
        )
```

- [ ] **Step 3: Insert the boost-pad chunk into `src/blueball/levels/tutorial_hill.json`**

Edit the chunks list — insert a `boost_pad` between `patrol_platform` and `stairs_up`:

```diff
     {"type": "patrol_platform", "length_tiles": 6, "patroller_speed": 60},
+    {"type": "boost_pad", "width_tiles": 3, "multiplier": 2.0},
     {"type": "stairs_up", "steps": 3, "step_height": 32},
```

The exact `width_tiles` and `multiplier` are starting points for feel-tuning, not contract.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`
Expected: all green (still 74 — no new tests in this task).

- [ ] **Step 5: Manual playtest**

Run:
```bash
BLUEBALL_SAVE_PATH=/tmp/blueball_boost_test.json .venv/bin/python main.py
```

Expected sequence:
- Roll past the patrol platform. The cyan strip appears between it and the stairs.
- Rolling over the strip, top rolling speed visibly increases (roughly doubles).
- The boost stays active through the stair climb if you jump while still on the pad.
- After your next mid-air → ground transition (i.e., the first time you land after leaving the strip), top speed returns to normal.
- Walking back onto the pad (without ever having left ground) keeps the boost indefinitely until you jump and land — confirming the "until you land" semantics.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/render/renderer.py src/blueball/config.py src/blueball/levels/tutorial_hill.json
git commit -m "feat: render boost pads and wire them into the tutorial level"
```

- [ ] **Step 7: (Optional) Iterate feel**

If the multiplier or pad width feels wrong, adjust the values in `tutorial_hill.json` and commit separately. Per standing memory: don't auto-commit during iterative tuning — wait for explicit "commit" from the user.

---

## Self-review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| `BoostPad` entity (static sensor poly, `multiplier` field) | Task 1 |
| `CT_BOOST_PAD = 6` and begin handler | Task 1 |
| Pad NOT consumed on contact | Task 1 (no `consume()` call; explicit test asserts pad remains in space) |
| `Player._boost_multiplier`, `_aerial_since_pickup` | Task 0 |
| `Player.receive_boost(m)` with take-the-max + aerial-tracking arm | Task 0 |
| `Player._update_boost(grounded)` end-on-landing | Task 0 |
| Effective speed cap (`MAX_LINEAR_SPEED * _boost_multiplier`) | Task 0 |
| Effective angular cap (`MAX_ANGULAR_VEL * _boost_multiplier`) | Task 0 |
| `boost_pad` chunk + registration | Task 2 |
| Renderer `draw_boost_pad` (cyan strip + chevron) | Task 3 |
| `BOOST_PAD_DEFAULT_MULTIPLIER = 2.0` in config | Task 3 |
| Tutorial level placement (between patrol_platform and stairs_up) | Task 3 |
| Manual end-to-end verification | Task 3 |
| Edge case: pickup while grounded persists until jump-and-land | Task 0 (test_player_boost_persists_while_grounded_until_jump_land_cycle) |
| Edge case: stronger boost replaces weaker mid-air | Task 0 (test_player_receive_boost_takes_max) |
| Death/respawn clears boost | Implicit — `_reset()` rebuilds Player (per spec's "Player dies / respawns → Boost cleared (`_reset()` rebuilds the Player)"). Not separately tested here; covered by the existing reset-on-death flow. |

No spec section is unaddressed. Spec's "What's deliberately out of scope" items (directional pads, instant velocity bump, time-based decay, audio/particles, persistent boost across deaths, multiplicative stacking) are explicitly NOT implemented.

**Type / name consistency:** `BoostPad.multiplier`, `BoostPad.width`, `Player._boost_multiplier`, `Player._aerial_since_pickup`, `receive_boost`, `_update_boost`, `CT_BOOST_PAD`, `BOOST_PAD_DEFAULT_MULTIPLIER`, `BOOST_PAD_THICKNESS`, `draw_boost_pad` — consistent across all tasks.

**Placeholder scan:** none.
