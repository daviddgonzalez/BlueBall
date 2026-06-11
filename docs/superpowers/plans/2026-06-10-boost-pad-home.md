# Boost-pad's Home + Box-lava Re-tune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the boost pad a real home — a dedicated boost-gap segment in the completion gym — and re-tune box-lava so it is genuinely solvable and double-jump-proof.

**Architecture:** A shared, correct double-jump test harness drives both throwaway tuning probes (which *find* max-margin geometry) and committed invariant tests (which *guard* it). A new `LavaGapChunk` (box-lava's pit minus the box) backs a new `BoostGapSegment`; box-lava keeps its box-push mechanic but loses the boost pad and gets a narrower/shallower pit. Geometry numbers are probe-determined and hard-coded.

**Tech Stack:** Python, pymunk (physics), pytest. Builds on the existing `levels/chunks/*` + `levels/segments.py` completion-gym scaffolding and the just-landed boost-pad fix (2s timer, +30% strength, seam welding).

**Spec:** `docs/superpowers/specs/2026-06-10-boost-pad-home-design.md`

---

### Task 0: Shared correct-double-jump maneuver harness

**Goal:** One module with the *correct* double-jump vault agent and the box-hop solver, imported by both probes and tests so "we tuned it" and "we guard it" run identical agents.

**Files:**
- Create: `tests/segment_maneuvers.py`
- Test: `tests/test_segment_maneuvers.py`

**Acceptance Criteria:**
- [ ] `DoubleJumpVaultAgent` performs a real two-jump arc: on a flat ledge with a gap it clears a gap a single jump cannot, and falls into a gap wider than its reach.
- [ ] `BoxHopAgent` exists with `(push_steps, jump1_x, box_run)` params and reads live `player`/`box` refs.
- [ ] `fresh_world()`, `find_entity(world, name)`, `run_segment(world, player, steps)` helpers return `"GOAL"|"DEAD"|"TIMEOUT"`.

**Verify:** `pytest -q tests/test_segment_maneuvers.py -v` → 2 tests pass

**Steps:**

- [ ] **Step 1: Write the failing test** (`tests/test_segment_maneuvers.py`)

```python
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pymunk
from blueball.abilities import Ability
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.world import World
from tests.segment_maneuvers import DoubleJumpVaultAgent, run_segment


def _ledge_gap_world(gap_px):
    """Two ledges at GROUND_Y with a lethal fall between near edge 256 and far."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    def seg(a, b):
        s = pymunk.Segment(w.space.static_body, a, b, 5)
        s.friction = 1.0
        w.space.add(s)
    seg((0, GROUND_Y), (256, GROUND_Y))                       # near ledge
    seg((256 + gap_px, GROUND_Y), (256 + gap_px + 400, GROUND_Y))  # far ledge
    return w


def test_double_jump_clears_a_gap_a_single_jump_cannot():
    # ~300px gap is clearable by a competent double jump, not a single one.
    w = _ledge_gap_world(300)
    agent = DoubleJumpVaultAgent(launch_x=250)
    p = Player(agent=agent, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    agent.player = p
    w.add_entity(p)
    reached = False
    for _ in range(400):
        w.substep()
        if p.body.position[0] > 256 + 300 + 10 and p.grounded:
            reached = True
            break
        if p.dead:
            break
    assert reached and not p.dead


def test_double_jump_falls_into_an_unreachable_gap():
    w = _ledge_gap_world(1200)  # far beyond any double-jump reach
    agent = DoubleJumpVaultAgent(launch_x=250)
    p = Player(agent=agent, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    agent.player = p
    w.add_entity(p)
    landed_far = False
    for _ in range(400):
        w.substep()
        if p.body.position[0] > 256 + 1200 and p.grounded:
            landed_far = True
            break
        if p.dead:
            break
    assert not landed_far
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_segment_maneuvers.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.segment_maneuvers`

- [ ] **Step 3: Write the harness** (`tests/segment_maneuvers.py`)

```python
"""Shared scripted agents for tuning probes AND committed segment tests.

The same agent that FOUND a geometry's safe margin also GUARDS it, so the two
can never drift. The double-jump maneuver is correct: a jump fires only on a
FRESH press (input_feel.JumpController), and releasing while ascending applies a
jump-cut — so we HOLD through the ascent and release only at apex (vy>=0, where
the cut is a no-op), then re-press for the air jump.
"""
from __future__ import annotations

from blueball.agent import Agent, Action
from blueball.world import World
from blueball.collision import register as register_collisions


def fresh_world(seed: int = 0) -> World:
    w = World(seed=seed)
    register_collisions(w.space, world_ref=w)
    return w


def find_entity(world, type_name: str):
    return next((e for e in world.entities
                if type(e).__name__ == type_name), None)


def run_segment(world, player, steps: int = 2500) -> str:
    for _ in range(steps):
        world.substep()
        if player.reached_goal:
            return "GOAL"
        if player.dead:
            return "DEAD"
    return "TIMEOUT"


class _Maneuver:
    """Mixin: one max-distance double jump to the right. Set self.player first."""
    def _start_jump(self):
        self._mj = "primary"

    def _maneuver(self):
        p = self.player
        grounded = p.grounded
        vy = p.body.velocity.y
        if self._mj == "primary":
            self._mj = "ascend"
            return Action.RIGHT_JUMP          # grounded fresh press -> primary fires
        if self._mj == "ascend":
            if (not grounded) and vy >= 0:     # apex
                self._mj = "release"
                return Action.RIGHT            # release at apex (cut is a no-op)
            return Action.RIGHT_JUMP           # hold through ascent
        if self._mj == "release":
            self._mj = "air"
            return Action.RIGHT_JUMP           # fresh airborne press -> air jump
        if self._mj == "air":
            if (not grounded) and vy >= 0:
                self._mj = "done"
            return Action.RIGHT_JUMP if self._mj == "air" else Action.RIGHT
        return Action.RIGHT                    # drift right under air control


class DoubleJumpVaultAgent(Agent, _Maneuver):
    """Roll right to launch_x, then one competent double jump. The strongest
    'cheese' attempt for anti-cheese tests, and the boost-gap solver when a pad
    sits on the run-up (the boost comes from the world, not the agent)."""
    def __init__(self, launch_x: float):
        self.launch_x = launch_x
        self.player = None
        self._mj = None

    def act(self, observation):
        if self._mj is None:
            if self.player.body.position[0] < self.launch_x:
                return Action.RIGHT
            self._start_jump()
        return self._maneuver()


class BoxHopAgent(Agent, _Maneuver):
    """Box-lava solver: shove the box into the pit, brake on the near ledge, then
    double-jump near-ledge -> box-top -> far-ledge -> goal. Set player+box."""
    def __init__(self, push_steps: int, jump1_x: float, box_run: int = 0):
        self.push_steps = push_steps
        self.jump1_x = jump1_x
        self.box_run = box_run
        self.player = None
        self.box = None
        self.phase = "SHOVE"
        self._t = 0
        self._mj = None
        self._run = 0

    def act(self, observation):
        p, box = self.player, self.box
        self._t += 1
        px = p.body.position[0]
        bx = box.body.position[0]

        if self.phase == "SHOVE":
            if self._t <= self.push_steps:
                return Action.RIGHT
            self.phase = "BRAKE"
        if self.phase == "BRAKE":
            if p.body.velocity[0] > 8.0 or px > 235.0:
                return Action.LEFT
            self.phase = "SETTLE"
            self._settle_t = self._t
        if self.phase == "SETTLE":
            if abs(box.body.velocity[0]) > 3.0 and self._t - self._settle_t < 120:
                return Action.IDLE
            self.phase = "APPROACH"
        if self.phase == "APPROACH":
            if px < self.jump1_x:
                return Action.RIGHT
            self.phase = "JUMP1"
            self._start_jump()
        if self.phase == "JUMP1":
            a = self._maneuver()
            if self._mj == "done" and p.grounded and abs(px - bx) < 40.0:
                self.phase = "ONBOX"
                self._run = 0
            return a
        if self.phase == "ONBOX":
            if self._run < self.box_run:
                self._run += 1
                return Action.RIGHT
            self.phase = "JUMP2"
            self._start_jump()
        if self.phase == "JUMP2":
            return self._maneuver()
        return Action.RIGHT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_segment_maneuvers.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/segment_maneuvers.py tests/test_segment_maneuvers.py
git commit -m "test(gym): shared correct-double-jump maneuver harness"
```

---

### Task 1: `LavaGapChunk` + shared pit-building helper

**Goal:** Extract box-lava's pit geometry into a shared helper and add a boxless lava-pit chunk (`lava_gap`) that the boost-gap segment crosses.

**Files:**
- Create: `src/blueball/levels/chunks/lava_gap.py`
- Modify: `src/blueball/levels/chunks/box_lava_gap.py` (use the shared helper; behavior unchanged)
- Test: `tests/test_chunks.py` (append 3 tests)

**Acceptance Criteria:**
- [ ] `build_lava_pit(world, x_offset, base_y, approach_tiles, pit_tiles, exit_tiles, depth)` returns `(pit_left, pit_right, floor_y, total)` and adds the two ledges, two walls, and a low-friction floor.
- [ ] `LavaGapChunk(pit_tiles=W)` is registered as `"lava_gap"`, builds a pit spanned by full-height `Lava`, and adds **no** `PushableBox`.
- [ ] `BoxLavaGap` still produces the same entities (existing `test_chunks.py`/`test_segments.py` for it pass unchanged).

**Verify:** `pytest -q tests/test_chunks.py -v` → existing + 3 new pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append to `tests/test_chunks.py`)

```python
def test_lava_gap_builds_pit_with_lava_and_no_box():
    from blueball.world import World
    from blueball.collision import register
    from blueball.levels.chunks.lava_gap import LavaGapChunk
    w = World(); register(w.space, world_ref=w)
    width = LavaGapChunk(pit_tiles=10).build(w, x_offset=0.0)
    names = [type(e).__name__ for e in w.entities]
    assert "Lava" in names
    assert "PushableBox" not in names
    assert width > 0


def test_lava_gap_registered():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "lava_gap" in CHUNK_REGISTRY


def test_build_lava_pit_returns_geometry():
    from blueball.world import World
    from blueball.collision import register
    from blueball.levels.chunks.lava_gap import build_lava_pit
    from blueball.levels.chunks.flat import GROUND_Y
    w = World(); register(w.space, world_ref=w)
    pit_left, pit_right, floor_y, total = build_lava_pit(
        w, x_offset=0.0, base_y=GROUND_Y, approach_tiles=2,
        pit_tiles=8, exit_tiles=2, depth=72)
    assert pit_right > pit_left
    assert floor_y == GROUND_Y + 72
    assert total == (2 + 8 + 2) * 32
```

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/test_chunks.py -k lava_gap` → FAIL (no module).

- [ ] **Step 3: Create `src/blueball/levels/chunks/lava_gap.py`**

```python
"""lava_gap chunk — a boxless lava pit too wide to clear without a boost.

Shares its pit geometry with box_lava_gap via build_lava_pit().
"""
from __future__ import annotations

import pymunk

from ...entities.lava import Lava
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

_DEPTH = 72


def build_lava_pit(world, x_offset, base_y, approach_tiles, pit_tiles,
                   exit_tiles, depth, floor_friction=0.1):
    """Build approach/exit ledges, the two pit walls, and a low-friction floor.
    Returns (pit_left, pit_right, floor_y, total_width_px). No lava, no box."""
    ax = approach_tiles * TILE
    px = pit_tiles * TILE
    ex = exit_tiles * TILE
    total = ax + px + ex
    pit_left = x_offset + ax
    pit_right = pit_left + px
    floor_y = base_y + depth

    def seg(a, b, friction=1.0):
        s = pymunk.Segment(world.space.static_body, a, b, 5)
        s.friction = friction
        world.space.add(s)

    seg((x_offset, base_y), (pit_left, base_y))            # approach ledge
    seg((pit_right, base_y), (x_offset + total, base_y))   # exit ledge
    seg((pit_left, base_y), (pit_left, floor_y))           # near wall
    seg((pit_right, base_y), (pit_right, floor_y))         # far wall
    seg((pit_left, floor_y), (pit_right, floor_y), floor_friction)
    return pit_left, pit_right, floor_y, total


@register_chunk("lava_gap")
class LavaGapChunk(Chunk):
    sampler_include: bool = False  # only used inside BoostGapSegment
    difficulty: int = 3

    def __init__(self, approach_tiles: int = 2, pit_tiles: int = 26,
                 exit_tiles: int = 2, depth: int = _DEPTH) -> None:
        self.approach_tiles = approach_tiles
        self.pit_tiles = pit_tiles
        self.exit_tiles = exit_tiles
        self.depth = depth

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        pit_left, pit_right, floor_y, total = build_lava_pit(
            world, x_offset, base_y, self.approach_tiles, self.pit_tiles,
            self.exit_tiles, self.depth)
        px = self.pit_tiles * TILE
        # Lava fills the pit from ledge level down: any fall is lethal.
        world.add_entity(Lava(
            world,
            position=(pit_left + px / 2, base_y + self.depth / 2),
            width=px,
            rise_speed=0.0,
            height=self.depth,
        ))
        return total
```

- [ ] **Step 4: Refactor `box_lava_gap.py`** to reuse `build_lava_pit` (replace the inline ledge/wall/floor build; keep the box + box-sized lava). Replace the body of `BoxLavaGap.build` with:

```python
    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        from .lava_gap import build_lava_pit
        pit_left, pit_right, floor_y, total = build_lava_pit(
            world, x_offset, base_y, self.approach_tiles, self.pit_tiles,
            self.exit_tiles, self.depth)
        px = self.pit_tiles * TILE
        box_rest_top = floor_y - self.box_size
        lava_surface = box_rest_top + _LAVA_BELOW_BOX
        world.add_entity(Lava(
            world, position=(pit_left + px / 2, lava_surface),
            width=px, rise_speed=0.0, height=self.depth))
        world.add_entity(PushableBox(
            world,
            position=(pit_left - self.box_size / 2 - 2, base_y - self.box_size / 2 - 1),
            size=self.box_size, mass=self.box_mass))
        return total
```

- [ ] **Step 5: Run tests** — `pytest -q tests/test_chunks.py tests/test_segments.py -v` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/levels/chunks/lava_gap.py src/blueball/levels/chunks/box_lava_gap.py tests/test_chunks.py
git commit -m "feat(gym): LavaGapChunk + shared build_lava_pit helper"
```

---

### Task 2: Re-tune box-lava (remove boost, narrow + shallow the pit)

**Goal:** Make box-lava genuinely solvable by the box-hop *and* double-jump-proof, with the boost pad removed; geometry found by a probe and hard-coded.

**Files:**
- Create: `probes/tune_box_lava.py`
- Modify: `src/blueball/levels/segments.py` (`BoxLavaSegment`, `KeyDoorBoxLavaSegment`)
- Modify: `tests/test_segments.py` (replace the `random varies` test; add a solvable test)

**Acceptance Criteria:**
- [ ] `probes/tune_box_lava.py` sweeps `(pit_tiles, depth)`; for each prints whether `BoxHopAgent` reaches the goal AND a box-removed `DoubleJumpVaultAgent` dies. It prints a non-empty safe set, or the closest miss.
- [ ] `BoxLavaSegment` and `KeyDoorBoxLavaSegment` use the probe-chosen `(pit_tiles, depth)` constants and contain **no** `BoostPadChunk`.
- [ ] New `test_boxlava_is_solvable_by_box_hop` passes; `test_boxlava_random_varies_pit_width` replaced by `test_boxlava_random_is_the_tuned_fixed_geometry`.
- [ ] **Contingency:** if the probe finds no `(pit_tiles, depth)` that is both solvable and vault-proof in `pit_tiles∈[12,22]`, `depth∈[40,80]`, STOP and report the closest miss for a human decision (do not change box size or the mechanic).

**Verify:** `python probes/tune_box_lava.py` prints a safe cell, then `pytest -q tests/test_segments.py -k boxlava -v` → pass

**Steps:**

- [ ] **Step 1: Write the probe** (`probes/tune_box_lava.py`)

```python
"""Find a box-lava (pit_tiles, depth) that BoxHopAgent solves and a box-removed
DoubleJumpVaultAgent cannot vault. Prints the safe set."""
import sys
sys.path.insert(0, "tests")
from segment_maneuvers import (fresh_world, find_entity, run_segment,
                               BoxHopAgent, DoubleJumpVaultAgent)
from blueball.abilities import Ability
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import BoxLavaSegment


def _remove(world, e):
    if e in world.entities:
        world.entities.remove(e)
    for s in list(getattr(e, "shapes", [])):
        if s in world.space.shapes:
            world.space.remove(s)
        world._shape_to_entity.pop(s, None)
    for b in list(getattr(e, "bodies", [])):
        if b in world.space.bodies:
            world.space.remove(b)


def solvable(pit, depth):
    for push in range(70, 130, 5):
        for jx in (232, 240, 248, 254):
            w = fresh_world()
            BoxLavaSegment(pit_tiles=pit, depth=depth).build(w, x_offset=0.0)
            ag = BoxHopAgent(push, jx)
            p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
                       abilities={Ability.DOUBLE_JUMP})
            ag.player = p; ag.box = find_entity(w, "PushableBox")
            w.add_entity(p)
            if run_segment(w, p) == "GOAL":
                return (push, jx)
    return None


def vaultable(pit, depth):
    for lx in (220, 240, 248, 254, 260):
        w = fresh_world()
        BoxLavaSegment(pit_tiles=pit, depth=depth).build(w, x_offset=0.0)
        _remove(w, find_entity(w, "PushableBox"))
        ag = DoubleJumpVaultAgent(lx)
        p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
                   abilities={Ability.DOUBLE_JUMP})
        ag.player = p
        w.add_entity(p)
        if run_segment(w, p, steps=600) == "GOAL":
            return True
    return False


safe = []
print("pit depth | solvable_by(push,jx) | vaultable")
for pit in range(12, 23, 2):
    for depth in (40, 48, 56, 64, 72, 80):
        s = solvable(pit, depth)
        v = vaultable(pit, depth)
        ok = "  <-- SAFE" if (s and not v) else ""
        print(f"{pit:3d} {depth:5d} | {str(s):20s} | {v}{ok}")
        if s and not v:
            safe.append((pit, depth, s))
print()
print("SAFE cells:", safe or "NONE — escalate (see Task 2 contingency)")
```

- [ ] **Step 2: Run the probe** — `python probes/tune_box_lava.py`. Pick the SAFE `(pit_tiles, depth)` with the widest margin (smallest pit that stays vault-proof, shallowest depth that stays solvable). If `SAFE` is empty → STOP, report, do not proceed.

- [ ] **Step 3: Update `segments.py`** — set the chosen constants, drop the boost pad:

```python
# probe-tuned in Task 2: solvable by box-hop, vault-proof under a competent
# double jump (probes/tune_box_lava.py).
_BOX_LAVA_PIT_TILES = 16   # replace with the probe's chosen value
_BOX_LAVA_DEPTH = 48       # replace with the probe's chosen value
```

```python
class BoxLavaSegment(SegmentTemplate):
    """Tier 2 — shove the box into the lava pit as a stepping stone, then reach
    the goal. The pit is fixed at the probe-tuned width/depth: solvable by the
    box-hop, and a competent DOUBLE_JUMP agent cannot vault it without the box."""
    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, pit_tiles=_BOX_LAVA_PIT_TILES, depth=_BOX_LAVA_DEPTH):
        self.pit_tiles = pit_tiles
        self.depth = depth

    @classmethod
    def random(cls, rng):
        return cls()  # fixed, probe-tuned geometry

    def build(self, world, x_offset):
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=self.pit_tiles, depth=self.depth), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset
```

Update `KeyDoorBoxLavaSegment.build` the same way: replace its `BoostPadChunk` + `BoxLavaGap(pit_tiles=24)` lines with `BoxLavaGap(pit_tiles=_BOX_LAVA_PIT_TILES, depth=_BOX_LAVA_DEPTH)` (keep key/door/flat chunks).

- [ ] **Step 4: Update tests** (`tests/test_segments.py`)

```python
def test_boxlava_random_is_the_tuned_fixed_geometry():
    import random
    from blueball.levels.segments import (BoxLavaSegment, _BOX_LAVA_PIT_TILES,
                                          _BOX_LAVA_DEPTH)
    s = BoxLavaSegment.random(random.Random(7))
    assert s.pit_tiles == _BOX_LAVA_PIT_TILES
    assert s.depth == _BOX_LAVA_DEPTH


def test_boxlava_is_solvable_by_box_hop():
    """Self-contained: some box-hop (push, jump1_x) reaches the goal. The sweep
    range brackets what the Task 2 probe found solvable; keep it small for speed."""
    from tests.segment_maneuvers import (fresh_world, find_entity,
                                        run_segment, BoxHopAgent)
    from blueball.abilities import Ability
    from blueball.entities.player import Player
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball.levels.segments import BoxLavaSegment
    solved = False
    for push in (88, 92, 96, 100):
        for jx in (240, 248):
            w = fresh_world()
            BoxLavaSegment().build(w, x_offset=0.0)
            ag = BoxHopAgent(push_steps=push, jump1_x=jx)
            p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
                       abilities={Ability.DOUBLE_JUMP})
            ag.player = p; ag.box = find_entity(w, "PushableBox")
            w.add_entity(p)
            if run_segment(w, p) == "GOAL":
                solved = True
                break
        if solved:
            break
    assert solved, "box-lava not solvable by any box-hop in the swept range"
```

Delete the old `test_boxlava_random_varies_pit_width`.

- [ ] **Step 5: Run** — `pytest -q tests/test_segments.py -k boxlava -v` → pass.

- [ ] **Step 6: Commit**

```bash
git add probes/tune_box_lava.py src/blueball/levels/segments.py tests/test_segments.py
git commit -m "fix(gym): box-lava solvable+vault-proof re-tune; drop the boost pad"
```

---

### Task 3: `BoostGapSegment` (boost-or-die lava gap)

**Goal:** A segment where a boost pad before a wide lava gap is the only way across; solvable with the boost, lethal without.

**Files:**
- Create: `probes/tune_boost_gap.py`
- Modify: `src/blueball/levels/segments.py` (add `BoostGapSegment`)
- Modify: `tests/test_segments.py` (add solvable + anti-cheese + composition tests)

**Acceptance Criteria:**
- [ ] `probes/tune_boost_gap.py` sweeps gap width `W`; prints for each whether `DoubleJumpVaultAgent` solves it WITH the pad and dies WITHOUT it. Prints the safe corridor.
- [ ] `BoostGapSegment` (tier 2, `min_abilities={DOUBLE_JUMP}`) builds `Flat(2) | BoostPad(3) | LavaGapChunk(W) | Goal(2)` with the probe-chosen `W`.
- [ ] `test_boostgap_is_solvable_with_boost` passes; `test_boostgap_requires_boost_not_double_jumpable` passes (pad stripped → death).

**Verify:** `python probes/tune_boost_gap.py` prints a safe corridor, then `pytest -q tests/test_segments.py -k boostgap -v` → pass

**Steps:**

- [ ] **Step 1: Write the probe** (`probes/tune_boost_gap.py`)

```python
"""Find a boost-gap width W that a boosted double jump clears but a bare
(no-boost) double jump cannot."""
import sys
sys.path.insert(0, "tests")
from segment_maneuvers import (fresh_world, find_entity, run_segment,
                               DoubleJumpVaultAgent)
from blueball.abilities import Ability
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import BoostGapSegment


def _strip_pads(world):
    for e in [e for e in world.entities if type(e).__name__ == "BoostPad"]:
        world.entities.remove(e)
        for s in list(e.shapes):
            if s in world.space.shapes:
                world.space.remove(s)
            world._shape_to_entity.pop(s, None)
        for b in list(e.bodies):
            if b in world.space.bodies:
                world.space.remove(b)


def run(W, with_boost):
    w = fresh_world()
    BoostGapSegment(gap_tiles=W).build(w, x_offset=0.0)
    if not with_boost:
        _strip_pads(w)
    # launch just before the pit edge: Flat(2)+BoostPad(3)+LavaGap approach(2)
    launch = (2 + 3 + 2) * 32 - 8
    ag = DoubleJumpVaultAgent(launch)
    p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    ag.player = p
    w.add_entity(p)
    return run_segment(w, p, steps=800)


print("W | with_boost | no_boost   (want GOAL / DEAD)")
for W in range(22, 34, 2):
    print(f"{W:2d} | {run(W, True):8s} | {run(W, False)}")
```

- [ ] **Step 2: Run the probe** — `python probes/tune_boost_gap.py`. Choose the largest `W` that is `GOAL` with boost AND `DEAD` without, with one tile of margin on each side.

- [ ] **Step 3: Add `BoostGapSegment` to `segments.py`** (import `BoostPadChunk`, `LavaGapChunk` at top):

```python
_BOOST_GAP_TILES = 28   # probe-tuned in Task 3 (boost-or-die corridor)


class BoostGapSegment(SegmentTemplate):
    """Tier 2 — a lava gap too wide for a bare double jump. A boost pad just
    before it is the only way across: cross the pad, then jump within the 2s
    boost window so the boost locks in and carries the leap. Fall = death."""
    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, gap_tiles=_BOOST_GAP_TILES):
        self.gap_tiles = gap_tiles

    @classmethod
    def random(cls, rng):
        return cls()

    def build(self, world, x_offset):
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoostPadChunk(width_tiles=3, multiplier=2.0), world, x)
        x += self._chunk(LavaGapChunk(pit_tiles=self.gap_tiles), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset
```

- [ ] **Step 4: Add tests** (`tests/test_segments.py`)

```python
def _strip_boost_pads(world):
    for e in [e for e in world.entities if type(e).__name__ == "BoostPad"]:
        world.entities.remove(e)
        for s in list(e.shapes):
            if s in world.space.shapes:
                world.space.remove(s)
            world._shape_to_entity.pop(s, None)
        for b in list(e.bodies):
            if b in world.space.bodies:
                world.space.remove(b)


def test_boostgap_composition_and_requirements():
    from blueball.world import World
    from blueball.collision import register
    from blueball.abilities import Ability
    from blueball.levels.segments import BoostGapSegment
    w = World(); register(w.space, world_ref=w)
    BoostGapSegment().build(w, x_offset=0.0)
    names = [type(e).__name__ for e in w.entities]
    assert "BoostPad" in names and "Lava" in names and "Goal" in names
    assert BoostGapSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoostGapSegment.min_abilities


def test_boostgap_is_solvable_with_boost():
    from tests.segment_maneuvers import fresh_world, run_segment, DoubleJumpVaultAgent
    from blueball.abilities import Ability
    from blueball.entities.player import Player
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball.levels.segments import BoostGapSegment
    w = fresh_world()
    BoostGapSegment().build(w, x_offset=0.0)
    launch = (2 + 3 + 2) * 32 - 8
    ag = DoubleJumpVaultAgent(launch)
    p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    ag.player = p
    w.add_entity(p)
    assert run_segment(w, p, steps=800) == "GOAL"


def test_boostgap_requires_boost_not_double_jumpable():
    from tests.segment_maneuvers import fresh_world, run_segment, DoubleJumpVaultAgent
    from blueball.abilities import Ability
    from blueball.entities.player import Player
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball.levels.segments import BoostGapSegment
    w = fresh_world()
    BoostGapSegment().build(w, x_offset=0.0)
    _strip_boost_pads(w)
    launch = (2 + 3 + 2) * 32 - 8
    ag = DoubleJumpVaultAgent(launch)
    p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    ag.player = p
    w.add_entity(p)
    assert run_segment(w, p, steps=800) != "GOAL"
```

- [ ] **Step 5: Run** — `pytest -q tests/test_segments.py -k boostgap -v` → pass.

- [ ] **Step 6: Commit**

```bash
git add probes/tune_boost_gap.py src/blueball/levels/segments.py tests/test_segments.py
git commit -m "feat(gym): BoostGapSegment (boost-or-die lava gap) + invariants"
```

---

### Task 4: Wire into the sampler + fix the false-confidence vault test

**Goal:** Register `BoostGapSegment` so the gym samples it, and replace the broken single-jump vault agent in the existing anti-cheese test with the correct one.

**Files:**
- Modify: `src/blueball/levels/segments.py` (`SEGMENT_TEMPLATES`)
- Modify: `tests/test_segments.py` (fix vault test, tier-count test, add sampler test)

**Acceptance Criteria:**
- [ ] `BoostGapSegment` is in `SEGMENT_TEMPLATES`; a sampler test shows it appears in a `{DOUBLE_JUMP}` pool and is excluded from a single-jump (`frozenset()`) pool.
- [ ] `test_all_four_tiers_registered` updated to assert **five** templates.
- [ ] `test_boxlava_pit_requires_the_box_not_vaultable` uses `DoubleJumpVaultAgent` (box removed) and still asserts no goal — replacing the `_DelayedJumpAgent` that only single-jumped.

**Verify:** `pytest -q tests/test_segments.py -v` → all pass

**Steps:**

- [ ] **Step 1: Add to `SEGMENT_TEMPLATES`** in `segments.py`:

```python
SEGMENT_TEMPLATES: list[type[SegmentTemplate]] = [
    GoalSegment,
    KeyDoorGoalSegment,
    BoxLavaSegment,
    BoostGapSegment,
    KeyDoorBoxLavaSegment,
]
```

- [ ] **Step 2: Fix the vault test + tier count** in `tests/test_segments.py`. Replace `_DelayedJumpAgent` usage in `test_boxlava_pit_requires_the_box_not_vaultable` with:

```python
def test_boxlava_pit_requires_the_box_not_vaultable():
    from tests.segment_maneuvers import fresh_world, find_entity, run_segment, DoubleJumpVaultAgent
    from blueball.abilities import Ability
    from blueball.entities.player import Player
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball.levels.segments import BoxLavaSegment
    for lx in (220, 240, 248, 254, 260):
        w = fresh_world()
        BoxLavaSegment().build(w, x_offset=0.0)
        box = find_entity(w, "PushableBox")
        w.entities.remove(box)
        for s in list(box.shapes):
            if s in w.space.shapes:
                w.space.remove(s)
            w._shape_to_entity.pop(s, None)
        for b in list(box.bodies):
            if b in w.space.bodies:
                w.space.remove(b)
        ag = DoubleJumpVaultAgent(lx)
        p = Player(agent=ag, spawn_xy=(40.0, GROUND_Y - 30.0),
                   abilities={Ability.DOUBLE_JUMP})
        ag.player = p
        w.add_entity(p)
        assert run_segment(w, p, steps=600) != "GOAL", f"vaulted at launch_x={lx}"
```

Update the tier-registration test:

```python
def test_all_five_tiers_registered():
    from blueball.levels.segments import SEGMENT_TEMPLATES
    assert len(SEGMENT_TEMPLATES) == 5
```

(Delete the now-unused `_DelayedJumpAgent` if no longer referenced.)

- [ ] **Step 3: Add the sampler test:**

```python
def test_boostgap_sampled_only_with_double_jump():
    from blueball.abilities import Ability
    from blueball.levels.segments import SegmentSampler, BoostGapSegment
    dj = SegmentSampler(granted=frozenset({Ability.DOUBLE_JUMP}))
    assert BoostGapSegment in dj._pool
    sj = SegmentSampler(granted=frozenset())
    assert BoostGapSegment not in sj._pool
```

(If `SegmentSampler`'s constructor/attribute names differ, match them — check `segments.py`.)

- [ ] **Step 4: Run** — `pytest -q tests/test_segments.py -v` → all pass; then `pytest -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/segments.py tests/test_segments.py
git commit -m "feat(gym): register BoostGapSegment; fix false-confidence vault test"
```

---

### Task 5: Weld seams in the gym stream (smooth on-screen playback)

**Goal:** Apply the existing seam weld after the gym materializes segments, so watched/visualized gym runs don't show the ball hopping at chunk joints. Optional polish; the boost already survives hops, so this is cosmetic.

**Files:**
- Modify: `src/blueball/levels/segments.py` (`SegmentStream` materialize path)
- Test: `tests/test_segment_stream.py` (append 1 test)

**Acceptance Criteria:**
- [ ] After `SegmentStream` materializes a segment, `weld_ground_seams(world.space)` has been called so consecutive collinear ground segments have neighbours set.
- [ ] A test asserts that after materializing, at least one static segment has a neighbour set (`.a`/`.b` differ from a freshly-built, unwelded control), or simply that `weld_ground_seams` is invoked (spy/refactor as fits the code).

**Verify:** `pytest -q tests/test_segment_stream.py -v` → existing + 1 new pass

**Steps:**

- [ ] **Step 1: Locate the materialize method** in `SegmentStream` (the method that calls `segment.build(world, ...)`). Append after a segment is built:

```python
from .seams import weld_ground_seams
weld_ground_seams(self.world.space)
```

(Use the stream's actual world reference attribute name.)

- [ ] **Step 2: Write the test** (`tests/test_segment_stream.py`) asserting the weld ran — e.g. monkeypatch `blueball.levels.seams.weld_ground_seams` with a counter and assert it was called after materialize, OR assert a known interior seam has non-default neighbours.

- [ ] **Step 3: Run** — `pytest -q tests/test_segment_stream.py -v` → pass.

- [ ] **Step 4: Commit**

```bash
git add src/blueball/levels/segments.py tests/test_segment_stream.py
git commit -m "feat(gym): weld ground seams on segment materialize"
```

---

## Final verification

- [ ] `pytest -q` → full suite green (481 + new tests).
- [ ] `python probes/tune_box_lava.py` and `python probes/tune_boost_gap.py` each print a non-empty safe corridor (or box-lava escalated per Task 2's contingency).
- [ ] `SEGMENT_TEMPLATES` has 5 templates; the gym samples `BoostGapSegment` only with `DOUBLE_JUMP`.
