# Abilities Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the unlockable-ability framework on Blue Ball's `Player`, persist unlocks to a save file, place a level-side pickup entity that grants abilities on contact, and ship **double jump** as the first concrete ability.

**Architecture:** A new `Ability` `StrEnum` and a tiny JSON-backed `save` module hold unlock state. `Player` gains an `abilities: set[Ability]` field (passed in at construction, mutated by `unlock()`), shared by reference with `JumpController` so the controller sees subsequent unlocks without a push. Double jump is implemented inside `JumpController.tick(...)` as a single new `fire = True` branch keyed on an `_air_jumps_remaining` counter. Pickups are sensor entities (`AbilityPickup`) routed through the existing collision dispatcher pattern and placed via a new `ability_pickup` chunk.

**Tech Stack:** Python 3.11+, PyGame-ce, Pymunk, pytest. No new third-party dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-25-abilities-framework-design.md`.

---

## File structure

Final layout after this plan lands. New files marked `+`, modified files marked `~`.

```
src/blueball/
├── abilities.py                            (+ Ability StrEnum)
├── save.py                                 (+ JSON-backed unlock store)
├── config.py                               (~ ABILITY_PICKUP_DEFAULT_HEIGHT)
├── collision.py                            (~ CT_ABILITY_PICKUP, on_ability_pickup)
├── input_feel.py                           (~ JumpController.__init__(abilities=...), air-jump branch)
├── entities/
│   ├── player.py                           (~ abilities slot, unlock())
│   └── ability_pickup.py                   (+ sensor entity)
├── levels/
│   ├── tutorial_hill.json                  (~ insert ability_pickup + gap)
│   └── chunks/
│       ├── __init__.py                     (~ import ability_pickup)
│       └── ability_pickup.py               (+ AbilityPickupChunk)
├── render/renderer.py                      (~ draw_ability_pickup, color constants)
└── scenes/play.py                          (~ load save in _reset())

tests/
├── test_save.py                            (+ )
├── test_play_scene.py                      (+ )
├── test_input_feel.py                      (~ double jump cases)
├── test_player.py                          (~ abilities slot, unlock())
├── test_entities.py                        (~ AbilityPickup cases)
├── test_chunks.py                          (~ ability_pickup case)
└── test_collision.py                       (~ pickup-contact case)
```

---

## Task 0: `Ability` enum + save file module

**Goal:** A pure-Python `Ability` `StrEnum` and a tiny `save` module that loads / persists the set of unlocked ability names to a JSON file. Both modules have no dependencies on the rest of the codebase, so future code can import them without import cycles.

**Files:**
- Create: `src/blueball/abilities.py`
- Create: `src/blueball/save.py`
- Create: `tests/test_save.py`

**Acceptance Criteria:**
- [ ] `Ability` is a `StrEnum` with member `DOUBLE_JUMP = "double_jump"`. Future abilities are added as new members.
- [ ] `save.load()` returns an empty set when the save file does not exist.
- [ ] `save.add_ability(name)` creates the parent directory if needed, writes a JSON object `{"unlocked_abilities": [...]}` with the names sorted, and is idempotent.
- [ ] The save path is overridable via the `BLUEBALL_SAVE_PATH` env var so tests can redirect to a tmp file.

**Verify:** `pytest -q tests/test_save.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_save.py`:

```python
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    """Redirect BLUEBALL_SAVE_PATH at a tmp file and force the save module to re-read it."""
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    # Force a fresh import so the SAVE_PATH module-level constant picks up the env var
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    return save_path, save_mod


def test_load_returns_empty_set_when_file_missing(tmp_save):
    _path, save_mod = tmp_save
    assert save_mod.load() == set()


def test_add_ability_creates_file_and_persists(tmp_save):
    path, save_mod = tmp_save
    save_mod.add_ability("double_jump")
    assert path.exists()
    assert save_mod.load() == {"double_jump"}
    data = json.loads(path.read_text())
    assert data == {"unlocked_abilities": ["double_jump"]}


def test_add_ability_is_idempotent(tmp_save):
    _path, save_mod = tmp_save
    save_mod.add_ability("double_jump")
    save_mod.add_ability("double_jump")
    assert save_mod.load() == {"double_jump"}


def test_add_ability_preserves_existing_unlocks(tmp_save):
    path, save_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["wall_jump"]}))
    save_mod.add_ability("double_jump")
    assert save_mod.load() == {"double_jump", "wall_jump"}
    # Stored sorted
    assert json.loads(path.read_text())["unlocked_abilities"] == ["double_jump", "wall_jump"]


def test_add_ability_creates_parent_directory(tmp_save, tmp_path, monkeypatch):
    nested = tmp_path / "nested" / "dir" / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(nested))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    save_mod.add_ability("double_jump")
    assert nested.exists()
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest -q tests/test_save.py -v`
Expected: `ModuleNotFoundError: No module named 'blueball.save'`

- [ ] **Step 3: Implement `src/blueball/abilities.py`**

```python
"""Ability enum — names persisted in the save file and referenced in level JSON.

Add a new member here to introduce a new ability. The string value is the
canonical name used on disk, in chunk parameters, and in code paths.
"""

from __future__ import annotations

from enum import StrEnum


class Ability(StrEnum):
    DOUBLE_JUMP = "double_jump"
    # Future: WALL_JUMP = "wall_jump", GROUND_POUND = "ground_pound"
```

- [ ] **Step 4: Implement `src/blueball/save.py`**

```python
"""Simple JSON-backed save file storing the set of unlocked ability names."""

from __future__ import annotations

import json
import os
from pathlib import Path

SAVE_PATH = Path(os.environ.get(
    "BLUEBALL_SAVE_PATH",
    str(Path.home() / ".blueball" / "save.json"),
))


def load() -> set[str]:
    if not SAVE_PATH.exists():
        return set()
    data = json.loads(SAVE_PATH.read_text())
    return set(data.get("unlocked_abilities", []))


def add_ability(name: str) -> None:
    abilities = load()
    if name in abilities:
        return
    abilities.add(name)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps(
        {"unlocked_abilities": sorted(abilities)}, indent=2,
    ))
```

- [ ] **Step 5: Run tests, confirm pass**

Run: `pytest -q tests/test_save.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/blueball/abilities.py src/blueball/save.py tests/test_save.py
git commit -m "feat: Ability enum and JSON-backed save file"
```

---

## Task 1: Double jump in `JumpController`

**Goal:** Extend `JumpController` so that it accepts the player's `abilities` set and, when `Ability.DOUBLE_JUMP` is present, allows exactly one extra mid-air jump per ground→air cycle. Existing buffer / coyote / cut behavior is unchanged when no abilities are passed.

**Files:**
- Modify: `src/blueball/input_feel.py`
- Modify: `tests/test_input_feel.py`

**Acceptance Criteria:**
- [ ] `JumpController(abilities=None)` defaults to an empty set; all existing `tick(...)` behavior is identical to v1 (the existing 7 tests in `test_input_feel.py` still pass).
- [ ] With `abilities={Ability.DOUBLE_JUMP}`: after a ground jump and a fresh airborne press, `fire=True` once; a second airborne press in the same air phase yields `fire=False`.
- [ ] After landing (grounded=True tick), the air-jump counter resets so the next air phase has one extra jump available again.
- [ ] Walking off a ledge without jumping leaves the air jump available — pressing jump while airborne fires it.
- [ ] An air jump still goes through buffer/coyote/cut logic — releasing mid-rise still sets `cut=True` on the next tick.

**Verify:** `pytest -q tests/test_input_feel.py -v` → all pass (existing 7 + new cases)

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_input_feel.py`**

Append:

```python
from blueball.abilities import Ability


def test_double_jump_disabled_when_ability_missing():
    jc = JumpController()
    # Ground jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Release in air
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh airborne press — no ability, should NOT fire
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_double_jump_fires_one_extra_air_jump_when_unlocked():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Ground jump (consumes the primary)
    d = jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True
    # Release in air
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # First airborne fresh press → air jump fires
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Release
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Second airborne fresh press → no more air jumps
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_double_jump_resets_on_landing():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # First cycle: ground jump, air jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Land
    jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Second cycle: ground jump fires, air jump fires again
    d = jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_double_jump_available_after_walk_off_ledge():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Several grounded ticks (no jump used)
    for _ in range(5):
        jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Walk off — grounded becomes False
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Past the coyote window
    for _ in range(int(config.COYOTE_TIME / config.PHYS_DT) + 2):
        jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh press → air jump should fire (we never used the primary)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_double_jump_air_jump_can_be_cut():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Ground jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Hold through one airborne tick (so we don't get a 'released' immediately)
    jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    # Release
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh air press → fires
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Release → cut next tick
    d = jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    assert d.cut is True
```

- [ ] **Step 2: Run, confirm new tests fail**

Run: `pytest -q tests/test_input_feel.py -v`
Expected: 7 existing pass, 5 new fail (and import error for `Ability` until step 3 lands).

- [ ] **Step 3: Modify `src/blueball/input_feel.py`**

Replace the file with:

```python
"""Input feel — jump buffering, coyote time, jump cut, optional double jump.

Pure state machine. No PyGame, no pymunk. Takes per-tick (action, grounded)
inputs and emits a JumpDecision telling the Player what to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .abilities import Ability
from .agent import Action


_JUMP_ACTIONS = {Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP}


@dataclass(frozen=True)
class JumpDecision:
    fire: bool   # apply jump impulse this tick
    cut: bool    # apply jump-cut (multiply upward velocity by JUMP_CUT_FACTOR) this tick


class JumpController:
    """Tracks jump-buffer, coyote-time, jump-cut, and air-jump state across ticks.

    `abilities` is shared by reference with the Player so unlocks land without
    a push. We hold the reference; reads happen each tick.
    """

    def __init__(self, abilities: set | None = None) -> None:
        self.abilities: set = abilities if abilities is not None else set()
        self._buffer_remaining = 0.0      # seconds until buffered jump expires
        self._coyote_remaining = 0.0      # seconds we still allow a jump after walking off
        self._was_grounded = False
        self._was_jump_held = False
        self._air_jumps_remaining = 0     # set when leaving the ground / firing a ground jump

    def _max_air_jumps(self) -> int:
        return 1 if Ability.DOUBLE_JUMP in self.abilities else 0

    def tick(self, action: Action, grounded: bool, dt: float) -> JumpDecision:
        jump_held = action in _JUMP_ACTIONS

        # Coyote timer: starts when we lose grounding while previously grounded
        if grounded:
            self._coyote_remaining = config.COYOTE_TIME
        else:
            self._coyote_remaining = max(0.0, self._coyote_remaining - dt)

        # Air-jump counter: reset on the grounded→airborne transition. Walking
        # off a ledge restocks the air jump; landing does too (handled below
        # by the next grounded→airborne transition).
        if self._was_grounded and not grounded:
            self._air_jumps_remaining = self._max_air_jumps()

        # Jump buffer: a fresh press while airborne starts (or refreshes) the buffer
        fresh_press = jump_held and not self._was_jump_held
        if fresh_press and not grounded:
            self._buffer_remaining = config.JUMP_BUFFER_TIME
        else:
            self._buffer_remaining = max(0.0, self._buffer_remaining - dt)

        # Decide fire:
        # 1. Fresh press while grounded → fire (primary)
        # 2. Fresh press during coyote window → fire (primary)
        # 3. Landing with a live buffer → fire (primary)
        # 4. Fresh airborne press, no coyote left, air-jump available → fire (air jump)
        fire = False
        primary_consumed = False
        if fresh_press and grounded:
            fire = True
            primary_consumed = True
        elif fresh_press and self._coyote_remaining > 0.0:
            fire = True
            primary_consumed = True
        elif grounded and not self._was_grounded and self._buffer_remaining > 0.0:
            fire = True
            primary_consumed = True
        elif fresh_press and not grounded and self._air_jumps_remaining > 0:
            fire = True
            self._air_jumps_remaining -= 1

        if primary_consumed:
            # The primary jump just fired — refill the air-jump counter so the
            # player can use it during this airborne phase.
            self._air_jumps_remaining = self._max_air_jumps()

        if fire:
            self._buffer_remaining = 0.0
            self._coyote_remaining = 0.0

        # Decide cut: released jump this tick (held last tick, not held now)
        released = (not jump_held) and self._was_jump_held
        cut = released

        self._was_grounded = grounded
        self._was_jump_held = jump_held
        return JumpDecision(fire=fire, cut=cut)
```

- [ ] **Step 4: Run tests, confirm all pass**

Run: `pytest -q tests/test_input_feel.py -v`
Expected: 12 passed (7 existing + 5 new)

- [ ] **Step 5: Run the full suite — no other test should regress**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/input_feel.py tests/test_input_feel.py
git commit -m "feat: double-jump support in JumpController"
```

---

## Task 2: `abilities` slot and `unlock()` on Player

**Goal:** `Player` accepts an `abilities` set at construction, exposes `unlock(ability)` that adds to the set, persists via `save.add_ability`, and propagates to its `JumpController`. The set is shared by reference, so future unlocks land in `JumpController` without re-pushing.

**Files:**
- Modify: `src/blueball/entities/player.py`
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] `Player(agent, spawn_xy)` constructs with `abilities == set()`.
- [ ] `Player(agent, spawn_xy, abilities={Ability.DOUBLE_JUMP})` constructs with that ability and the `JumpController` sees it (`Ability.DOUBLE_JUMP in player.jump_ctrl.abilities`).
- [ ] `player.unlock(Ability.DOUBLE_JUMP)` adds the ability, persists via `save.add_ability("double_jump")`, and the `JumpController` reflects the new ability on the next tick.
- [ ] Calling `unlock` for an already-unlocked ability is a no-op (no extra write).
- [ ] Existing v1 player behavior (movement, jump, die) is unchanged for default construction.

**Verify:** `pytest -q tests/test_player.py -v` → all pass

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_player.py`**

```python
from blueball.abilities import Ability


def test_player_default_abilities_is_empty():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.abilities == set()
    assert p.jump_ctrl.abilities is p.abilities  # shared by reference


def test_player_constructed_with_abilities_propagates_to_jump_controller():
    p = Player(
        agent=_ScriptedAgent([Action.IDLE]),
        spawn_xy=(100, 100),
        abilities={Ability.DOUBLE_JUMP},
    )
    assert Ability.DOUBLE_JUMP in p.abilities
    assert Ability.DOUBLE_JUMP in p.jump_ctrl.abilities


def test_player_unlock_adds_and_persists(monkeypatch, tmp_path):
    save_file = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_file))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    # Player must import the freshly-reloaded module
    import blueball.entities.player as player_mod
    importlib.reload(player_mod)

    p = player_mod.Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.unlock(Ability.DOUBLE_JUMP)
    assert Ability.DOUBLE_JUMP in p.abilities
    assert Ability.DOUBLE_JUMP in p.jump_ctrl.abilities
    assert save_mod.load() == {"double_jump"}


def test_player_unlock_is_idempotent(monkeypatch, tmp_path):
    save_file = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_file))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    import blueball.entities.player as player_mod
    importlib.reload(player_mod)

    p = player_mod.Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.unlock(Ability.DOUBLE_JUMP)
    mtime_first = save_file.stat().st_mtime_ns
    p.unlock(Ability.DOUBLE_JUMP)
    mtime_second = save_file.stat().st_mtime_ns
    assert mtime_first == mtime_second  # no rewrite on no-op unlock
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_player.py -v`
Expected: 4 new failures (TypeError on the abilities kwarg / missing attribute).

- [ ] **Step 3: Modify `src/blueball/entities/player.py`**

Patch the top-of-file imports and `__init__`. At the top, add:

```python
from .. import save
from ..abilities import Ability
```

Update the constructor signature and body:

```python
    def __init__(
        self,
        agent: Agent,
        spawn_xy: tuple[float, float],
        abilities: set[Ability] | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        moment = pymunk.moment_for_circle(config.BALL_MASS, 0, config.BALL_RADIUS)
        self.body = pymunk.Body(mass=config.BALL_MASS, moment=moment)
        self.body.position = spawn_xy
        self.shape = pymunk.Circle(self.body, config.BALL_RADIUS)
        self.shape.friction = config.BALL_FRICTION
        self.shape.elasticity = config.BALL_ELASTICITY
        self.shape.collision_type = 1
        self.bodies.append(self.body)
        self.shapes.append(self.shape)

        self.abilities: set[Ability] = set(abilities) if abilities else set()
        self.jump_ctrl = JumpController(abilities=self.abilities)
        self.dead = False
        self.collectibles_collected = 0
        self._contact_normals: list = []
```

Add the new method anywhere below `die()`:

```python
    def unlock(self, ability: Ability) -> None:
        if ability in self.abilities:
            return
        self.abilities.add(ability)
        save.add_ability(ability.value)
```

(The set is shared by reference with `JumpController.abilities`; adding here is visible there on the next tick.)

- [ ] **Step 4: Run player tests**

Run: `pytest -q tests/test_player.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: ability slots and unlock() on Player"
```

---

## Task 3: `AbilityPickup` entity + collision handler

**Goal:** A new sensor entity `AbilityPickup` plus a registered begin handler on `CT_PLAYER ↔ CT_ABILITY_PICKUP` that calls `player.unlock(pickup.ability)` and tears down the pickup (mirrors how `Collectible` works).

**Files:**
- Create: `src/blueball/entities/ability_pickup.py`
- Modify: `src/blueball/collision.py`
- Modify: `tests/test_entities.py`
- Modify: `tests/test_collision.py`

**Acceptance Criteria:**
- [ ] `AbilityPickup(world, position, ability, radius=18)` creates a STATIC body with a sensor `Circle` shape, `collision_type == CT_ABILITY_PICKUP`.
- [ ] `collision.CT_ABILITY_PICKUP == 7`.
- [ ] A `CT_PLAYER` ↔ `CT_ABILITY_PICKUP` begin contact calls `player.unlock(pickup.ability)`, sets `pickup._collected = True`, and removes the pickup's body/shape from the space.
- [ ] Subsequent contacts after collection are no-ops (`_collected` guard).

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py -v` → all pass

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_entities.py`**

```python
from blueball.abilities import Ability
from blueball.entities.ability_pickup import AbilityPickup


def test_ability_pickup_is_sensor_with_correct_collision_type():
    w = World()
    p = AbilityPickup(w, position=(100, 200), ability=Ability.DOUBLE_JUMP)
    w.add_entity(p)
    assert p.shapes[0].sensor is True
    assert p.shapes[0].collision_type == collision.CT_ABILITY_PICKUP


def test_ability_pickup_stores_ability():
    w = World()
    p = AbilityPickup(w, position=(100, 200), ability=Ability.DOUBLE_JUMP)
    assert p.ability == Ability.DOUBLE_JUMP
    assert p._collected is False
```

- [ ] **Step 2: Add failing tests to `tests/test_collision.py`**

```python
import pytest

from blueball.abilities import Ability


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    return save_mod


def test_player_unlocks_ability_on_pickup_contact(tmp_save):
    from blueball.entities.ability_pickup import AbilityPickup
    w, p = _player_world()
    # Place the pickup directly on top of the player so contact is immediate.
    pickup = AbilityPickup(w, position=(100, 100), ability=Ability.DOUBLE_JUMP, radius=20)
    w.add_entity(pickup)

    for _ in range(5):
        w.step(1 / 60)
        if pickup._collected:
            break
    assert Ability.DOUBLE_JUMP in p.abilities
    assert pickup._collected is True
    assert pickup.shapes[0] not in w.space.shapes
    assert tmp_save.load() == {"double_jump"}
```

- [ ] **Step 3: Run, confirm failure**

Run: `pytest -q tests/test_entities.py tests/test_collision.py -v`
Expected: import errors + new test failures.

- [ ] **Step 4: Implement `src/blueball/entities/ability_pickup.py`**

```python
"""AbilityPickup — a sensor entity that grants an Ability to the Player on contact.

Mirrors the structure of Collectible: a static-body sensor circle that tears
down its physics presence once collected.
"""

from __future__ import annotations

import pymunk

from ..abilities import Ability
from ..collision import CT_ABILITY_PICKUP
from .base import Entity


class AbilityPickup(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        ability: Ability,
        radius: int = 18,
    ) -> None:
        super().__init__()
        self._world = world
        self.ability = ability
        self.position = position
        self.radius = radius

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, radius)
        shape.sensor = True
        shape.collision_type = CT_ABILITY_PICKUP
        self.bodies.append(self.body)
        self.shapes.append(shape)
        self._collected = False

    def consume(self) -> None:
        """Remove this pickup from the physics space and flag it dead."""
        if self._collected:
            return
        self._collected = True
        self.alive = False
        for shape in self.shapes:
            if shape in self._world.space.shapes:
                self._world.space.remove(shape)
        for body in self.bodies:
            if body in self._world.space.bodies:
                self._world.space.remove(body)

    def draw(self, renderer, alpha: float) -> None:
        if self.alive:
            renderer.draw_ability_pickup(self.position, self.radius, str(self.ability))
```

- [ ] **Step 5: Modify `src/blueball/collision.py`** — add the constant, the handler, and the registration.

Add the constant alongside the others:

```python
CT_ABILITY_PICKUP = 7
```

Inside `register(space, world_ref)`, add the handler before the final block of `space.on_collision(...)` calls:

```python
    def on_ability_pickup(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if not hasattr(entity, "ability") or entity._collected:
                continue
            if player is not None:
                player.unlock(entity.ability)
            entity.consume()
        return False  # sensor — no physical response
```

And register it:

```python
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_ABILITY_PICKUP, begin=on_ability_pickup)
```

- [ ] **Step 6: Run targeted tests, confirm pass**

Run: `pytest -q tests/test_entities.py tests/test_collision.py -v`
Expected: all pass.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/blueball/entities/ability_pickup.py src/blueball/collision.py tests/test_entities.py tests/test_collision.py
git commit -m "feat: AbilityPickup entity and collision handler"
```

---

## Task 4: `ability_pickup` chunk

**Goal:** A new chunk type `ability_pickup` lays a flat ground segment with an `AbilityPickup` sensor floating above it. Registered in the chunk registry so level JSON can reference it.

**Files:**
- Create: `src/blueball/levels/chunks/ability_pickup.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `src/blueball/config.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `"ability_pickup"` is in `CHUNK_REGISTRY` after importing the chunks package.
- [ ] Building an `ability_pickup` chunk adds exactly one `AbilityPickup` entity to the world; the pickup's `ability` matches the JSON-supplied string (e.g., `"double_jump"` → `Ability.DOUBLE_JUMP`).
- [ ] The chunk's reported width equals `width_tiles * TILE`, so the level loader places the next chunk correctly.
- [ ] An invalid ability string raises (the `Ability(string)` lookup error is surfaced).

**Verify:** `pytest -q tests/test_chunks.py -v` → all pass

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_chunks.py`**

```python
from blueball.abilities import Ability
from blueball.entities.ability_pickup import AbilityPickup


def test_ability_pickup_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "ability_pickup" in CHUNK_REGISTRY


def test_ability_pickup_chunk_adds_one_pickup_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    w = World()
    chunk = CHUNK_REGISTRY["ability_pickup"](width_tiles=2, ability="double_jump")
    width = chunk.build(w, x_offset=0.0)
    from blueball.levels.chunks.base import TILE
    assert width == 2 * TILE
    pickups = [e for e in w.entities if isinstance(e, AbilityPickup)]
    assert len(pickups) == 1
    assert pickups[0].ability == Ability.DOUBLE_JUMP


def test_ability_pickup_chunk_rejects_unknown_ability():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    w = World()
    with pytest.raises(ValueError):
        CHUNK_REGISTRY["ability_pickup"](width_tiles=2, ability="frobnicate").build(w, x_offset=0.0)
```

If `pytest` is not already imported at the top of the test file, add `import pytest`.

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_chunks.py -v`
Expected: 3 new failures (`KeyError: 'ability_pickup'`).

- [ ] **Step 3: Add the config constant**

Edit `src/blueball/config.py`. Append:

```python
# Abilities
ABILITY_PICKUP_DEFAULT_HEIGHT = 64    # px above ground where pickups float
```

- [ ] **Step 4: Implement `src/blueball/levels/chunks/ability_pickup.py`**

```python
"""ability_pickup chunk — a flat ground segment with a floating AbilityPickup."""

from __future__ import annotations

import pymunk

from ... import config
from ...abilities import Ability
from ...entities.ability_pickup import AbilityPickup
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("ability_pickup")
class AbilityPickupChunk(Chunk):
    def __init__(
        self,
        width_tiles: int = 2,
        ability: str = "double_jump",
        height: int = config.ABILITY_PICKUP_DEFAULT_HEIGHT,
    ) -> None:
        # Validate eagerly so a broken level JSON fails at load, not at collision time
        self.ability = Ability(ability)
        self.width_tiles = width_tiles
        self.height = height

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, GROUND_Y),
            (x_offset + w, GROUND_Y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(AbilityPickup(
            world,
            position=(x_offset + w / 2, GROUND_Y - self.height),
            ability=self.ability,
        ))
        return w
```

- [ ] **Step 5: Register the chunk module**

Edit `src/blueball/levels/chunks/__init__.py`:

```python
"""Importing this package registers every chunk type."""

from . import flat, gap, spike_pit, patrol_platform, stairs, bump, falling_hazard, goal, ability_pickup  # noqa: F401
```

- [ ] **Step 6: Run chunk tests**

Run: `pytest -q tests/test_chunks.py -v`
Expected: all pass.

- [ ] **Step 7: Run full suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/blueball/levels/chunks/ability_pickup.py src/blueball/levels/chunks/__init__.py src/blueball/config.py tests/test_chunks.py
git commit -m "feat: ability_pickup chunk type"
```

---

## Task 5: `PlayScene` reads save file at level start

**Goal:** `PlayScene._reset()` reads the save file once per level start, builds an `abilities` set, and passes it to `Player`. Unknown ability names in the save are ignored so the loader is forward-compatible.

**Files:**
- Modify: `src/blueball/scenes/play.py`
- Create: `tests/test_play_scene.py`

**Acceptance Criteria:**
- [ ] On `PlayScene.__init__`, `scene.player.abilities` is the set of `Ability` values present in the save file (filtered to known members).
- [ ] A save file containing an unknown string (e.g., `"frobnicate"`) does not raise; only known abilities end up on the player.
- [ ] An empty / missing save file results in `scene.player.abilities == set()`.

**Verify:** `pytest -q tests/test_play_scene.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write failing tests**

`tests/test_play_scene.py`:

```python
import importlib
import json

import pygame
import pytest

from blueball.abilities import Ability


@pytest.fixture
def headless_pygame():
    # Use a dummy SDL driver so pygame can run in CI without a display.
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.display.quit()


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    import blueball.save as save_mod
    importlib.reload(save_mod)
    # PlayScene imports Player which imports save — reload Player too.
    import blueball.entities.player as player_mod
    importlib.reload(player_mod)
    import blueball.scenes.play as play_mod
    importlib.reload(play_mod)
    return save_path, save_mod, play_mod


def _level_path():
    from blueball.levels import loader as _loader  # noqa: F401
    from pathlib import Path
    import blueball
    return Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"


def test_play_scene_starts_with_no_abilities_when_save_missing(headless_pygame, tmp_save):
    _path, _save_mod, play_mod = tmp_save
    scene = play_mod.PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == set()


def test_play_scene_loads_unlocked_abilities_from_save(headless_pygame, tmp_save):
    path, _save_mod, play_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["double_jump"]}))
    scene = play_mod.PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == {Ability.DOUBLE_JUMP}


def test_play_scene_ignores_unknown_abilities_in_save(headless_pygame, tmp_save):
    path, _save_mod, play_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["frobnicate", "double_jump"]}))
    scene = play_mod.PlayScene(headless_pygame, _level_path())
    assert scene.player.abilities == {Ability.DOUBLE_JUMP}
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_play_scene.py -v`
Expected: failures because `PlayScene` doesn't yet load abilities.

- [ ] **Step 3: Modify `src/blueball/scenes/play.py`**

Add at the top with the other imports:

```python
from .. import save
from ..abilities import Ability
```

In `_reset()`, after `load_level` and before constructing `Player`:

```python
        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=unlocked,
        )
```

(Replaces the existing `self.player = Player(...)` line.)

- [ ] **Step 4: Run play-scene tests**

Run: `pytest -q tests/test_play_scene.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/scenes/play.py tests/test_play_scene.py
git commit -m "feat: PlayScene loads abilities from save file"
```

---

## Task 6: Renderer + tutorial level + manual playtest

**Goal:** Wire the new pickup visual into the renderer, place an `ability_pickup` chunk into `tutorial_hill.json` in a spot where double jump is immediately useful, and verify the feature works end-to-end by playing it.

**Files:**
- Modify: `src/blueball/render/renderer.py`
- Modify: `src/blueball/levels/tutorial_hill.json`

**Acceptance Criteria:**
- [ ] `Renderer.draw_ability_pickup(pos, radius, ability: str)` draws a pulsing diamond with a per-ability color (warm yellow for `"double_jump"`).
- [ ] `tutorial_hill.json` contains an `ability_pickup` chunk followed by a gap that is wider than a single ground jump can cross, so the new ability is obviously useful.
- [ ] All tests pass.
- [ ] Manual playtest: launching `python main.py`, rolling onto the pickup makes it disappear, and pressing jump twice in mid-air now produces a second jump. Reaching the goal once and rerunning the game preserves the unlock (jumping is still possible mid-air without re-collecting the pickup). Deleting / pointing `BLUEBALL_SAVE_PATH` elsewhere restores the locked state.

**Verify:** `pytest -q` → all green; then `python main.py` and observe the behavior listed above.

**Steps:**

- [ ] **Step 1: Modify `src/blueball/render/renderer.py`**

Add module-level color constants alongside the existing ones (near the top of the file):

```python
_ABILITY_PICKUP_DEFAULT = (220, 220, 220)
_ABILITY_PICKUP_COLORS = {
    "double_jump": (255, 220, 80),
}
```

Add the draw method to the `Renderer` class:

```python
    def draw_ability_pickup(self, pos, radius, ability: str) -> None:
        color = _ABILITY_PICKUP_COLORS.get(ability, _ABILITY_PICKUP_DEFAULT)
        sx, sy = self._w2s(pos)
        pulse = 1.0 + 0.15 * math.sin(
            pygame.time.get_ticks() / 1000.0 * 2 * math.pi * 1.5
        )
        r = int(radius * pulse)
        pygame.draw.polygon(self.screen, color, [
            (int(sx), int(sy) - r),
            (int(sx) + r, int(sy)),
            (int(sx), int(sy) + r),
            (int(sx) - r, int(sy)),
        ])
```

- [ ] **Step 2: Modify `src/blueball/levels/tutorial_hill.json`**

Edit the chunks list. Replace the existing prefix:

```diff
-    {"type": "flat", "width_tiles": 8},
+    {"type": "flat", "width_tiles": 6},
+    {"type": "ability_pickup", "width_tiles": 2, "ability": "double_jump"},
+    {"type": "gap", "width_tiles": 4},
+    {"type": "flat", "width_tiles": 2},
     {"type": "bump", "width_tiles": 2, "height": 24},
```

(The original `flat width_tiles=8` is broken into `flat 6` + the pickup + a 4-tile gap that's too wide to cross with a single ground jump + a short `flat 2` to land on. Resulting horizontal extent of this prefix is identical to the original 8 tiles + new content. Iterate the gap width during playtesting if it lands too easy / too hard.)

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: all green. If `test_play_scene.py` references the JSON, make sure no test asserts a specific chunk sequence that this change would break (none currently does, per the spec's design).

- [ ] **Step 4: Manual playtest — first run (locked state)**

Run: `BLUEBALL_SAVE_PATH=/tmp/blueball_test_save.json python main.py`

Expected sequence (do all of these):
- The ball rolls forward, hits the pickup (yellow pulsing diamond floating above the ground). It disappears.
- Continuing past the pickup, the ball reaches the 4-tile gap. A single ground jump should land short.
- Holding direction into the air and pressing jump a second time produces a second jump, clearing the gap.
- Reach the goal.

- [ ] **Step 5: Manual playtest — second run (persisted unlock)**

Re-run: `BLUEBALL_SAVE_PATH=/tmp/blueball_test_save.json python main.py`

Expected:
- Save file `/tmp/blueball_test_save.json` exists and contains `"double_jump"`.
- Without picking up the diamond, jumping twice in midair still works (the unlock persisted).

- [ ] **Step 6: Manual playtest — fresh state**

Delete the tmp save and re-run with a fresh path:

```bash
rm /tmp/blueball_test_save.json
BLUEBALL_SAVE_PATH=/tmp/blueball_test_save_2.json python main.py
```

Expected: double jump is locked again; only one jump per air phase until the pickup is collected.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/render/renderer.py src/blueball/levels/tutorial_hill.json
git commit -m "feat: render ability pickups and wire double jump into tutorial level"
```

- [ ] **Step 8: (Optional) Iterate feel**

If the gap is too wide or too narrow, adjust `width_tiles` on the `gap` chunk in `tutorial_hill.json` and commit again. This is normal feel-tuning territory (per the standing memory: don't auto-commit during iterative tuning — wait for an explicit "commit").

---

## Self-review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| Slot system on Player | Task 2 |
| `Ability` StrEnum | Task 0 |
| Save file (`save.py`, env var override) | Task 0 |
| `Player.unlock()` with persistence | Task 2 |
| `JumpController` shares the abilities set by reference | Tasks 1, 2 |
| Double jump (air-jump counter, resets on grounded→airborne, primary refills) | Task 1 |
| `AbilityPickup` entity (sensor circle) | Task 3 |
| `CT_ABILITY_PICKUP = 7` and begin handler | Task 3 |
| `ability_pickup` chunk + registration | Task 4 |
| `PlayScene` loads save at `_reset()` with unknown-name filter | Task 5 |
| Renderer `draw_ability_pickup` + per-ability color map | Task 6 |
| `ABILITY_PICKUP_DEFAULT_HEIGHT` in config | Task 4 |
| `tutorial_hill.json` placement | Task 6 |
| Manual end-to-end verification | Task 6 |

No spec section is unaddressed.

**Type / name consistency check:**
- `Player(abilities=...)` ↔ `JumpController(abilities=...)` ↔ `save.load() / add_ability(name)` — all consistent across Tasks 0–2.
- `AbilityPickup.ability` ↔ `pickup.consume()` ↔ `collision.on_ability_pickup` — consistent across Task 3.
- `CT_ABILITY_PICKUP = 7` (Task 3) ↔ referenced in `entities/ability_pickup.py` import (Task 3) — consistent.

**Placeholder scan:** none.
