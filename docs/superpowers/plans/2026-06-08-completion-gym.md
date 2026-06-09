# Completion Gym Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training-only "completion gym" mode — an endless chain of solvable, goal-terminated segments (keys, doors, boxes, lava) with difficulty ramping by depth — so the GA-trained generalist learns the completion mechanics Infinite Run never exercises.

**Architecture:** Reuse the existing completion *chunks* as the vocabulary for solvable *segment templates* (Approach A). A `SegmentSampler` emits templates with a depth-driven tier ramp; a `SegmentStream` (sibling of `TerrainStream`) materializes them ahead of the ball and culls behind. A new `evaluate_gym` evaluator never terminates on a goal — it counts segment clears by x-boundary crossing and clears `keys_held` per segment while tracking cumulative keys. No collision/entity changes.

**Tech Stack:** Python 3.12, pymunk (physics), numpy (FTNN/GA), pytest. Package `blueball` under `src/` (editable install; `tests/conftest.py` prepends the worktree `src/`).

**Reference spec:** `docs/superpowers/specs/2026-06-08-completion-gym-design.md`

**Conventions to mirror:**
- Chunks are built with `chunk.build(world, x_offset=x)` — **no `base_y`** (see `levels/loader.py:45`); every chunk defaults to `GROUND_Y = 600`. The gym stays on this flat baseline.
- Headless evaluators step physics with `world.substep()` (one `PHYS_DT`, no accumulator) for cross-host determinism.
- Tests import `from blueball...`; place them flat in `tests/test_*.py`.

---

### Task 1: Fitness — repeatable per-segment clear reward

**Goal:** Add a `segments_cleared` term to the fitness function so each cleared gym segment banks a goal-sized reward, with non-gym callers numerically unchanged.

**Files:**
- Modify: `src/blueball/config.py` (add `GYM_SEGMENT_BONUS`)
- Modify: `src/blueball/ai/fitness.py`
- Test: `tests/test_gym_fitness.py`

**Acceptance Criteria:**
- [ ] `FitnessInputs` has `segments_cleared: int = 0` (defaulted, last field).
- [ ] `fitness()` adds `config.GYM_SEGMENT_BONUS * segments_cleared`.
- [ ] With `segments_cleared=0`, fitness equals the previous formula (backward compatible).

**Verify:** `pytest tests/test_gym_fitness.py -v` → 2 passed

**Steps:**

- [ ] **Step 1: Write the failing test** — `tests/test_gym_fitness.py`

```python
from blueball import config
from blueball.ai.fitness import FitnessInputs, fitness


def _base(**over):
    kw = dict(progress_x=100.0, collectibles=0, reached_goal=False, died=False,
              steps_taken=0, keys_collected=0, level_width=0.0)
    kw.update(over)
    return FitnessInputs(**kw)


def test_segments_cleared_defaults_to_zero_backward_compatible():
    # progress only, no time/death/goal → exactly progress_x
    assert fitness(_base()) == 100.0


def test_each_cleared_segment_adds_the_bonus():
    f0 = fitness(_base())
    f3 = fitness(_base(segments_cleared=3))
    assert f3 - f0 == 3 * config.GYM_SEGMENT_BONUS
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_gym_fitness.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'segments_cleared'`.

- [ ] **Step 3: Add the config constant** — in `src/blueball/config.py`, near the other training constants (after `GOAL_MULT`):

```python
# --- Completion Gym ---
# Flat reward banked per cleared gym segment. ~ GOAL_MULT (2.0) * a typical
# segment width (~600 px), so reward-per-completion is in the same range as the
# campaign goal bonus (aids transfer). Tunable.
GYM_SEGMENT_BONUS = 1200.0
```

- [ ] **Step 4: Extend `FitnessInputs` and `fitness()`** — `src/blueball/ai/fitness.py`

Add the field (last, defaulted) to the dataclass:

```python
    keys_collected: int  # popcount of player.keys_held (cumulative in the gym)
    level_width: float   # level total width; 0.0 for goalless (infinite) modes
    segments_cleared: int = 0  # gym: count of solved segments; 0 elsewhere
```

Add the term to `fitness()`:

```python
def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + config.GOAL_MULT * inputs.level_width * (1.0 if inputs.reached_goal else 0.0)
        + config.GYM_SEGMENT_BONUS * inputs.segments_cleared
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
    )
```

- [ ] **Step 5: Run it, expect pass**

Run: `pytest tests/test_gym_fitness.py -v`
Expected: 2 passed.

- [ ] **Step 6: Confirm no regression in existing fitness/eval tests**

Run: `pytest tests/test_ai_multiepisode.py tests/test_ai_smoke.py -q`
Expected: all pass (the defaulted field leaves existing call sites unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/fitness.py tests/test_gym_fitness.py
git commit -m "feat(gym): repeatable per-segment clear reward in fitness"
```

---

### Task 2: Segment templates — interface + Tier 0 & 1

**Goal:** Create `segments.py` with the `SegmentTemplate` interface and the two no-ability tiers (`GoalSegment`, `KeyDoorGoalSegment`), each solvable by rolling right.

**Files:**
- Create: `src/blueball/levels/segments.py`
- Test: `tests/test_segments.py`

**Acceptance Criteria:**
- [ ] `SegmentTemplate.build(world, x_offset) -> float` returns positive width; builds chunks via `chunk.build(world, x_offset=...)` (no `base_y`).
- [ ] `GoalSegment` (tier 0) builds a `Goal`; `KeyDoorGoalSegment` (tier 1) builds `Key`, `Door`, `Goal` with the key positioned left of the door.
- [ ] Both declare `min_abilities = frozenset()` and are solvable by a roll-right agent (reach the goal under physics).

**Verify:** `pytest tests/test_segments.py -v` → all passed

**Steps:**

- [ ] **Step 1: Write the failing tests** — `tests/test_segments.py`

```python
import pytest

from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import GoalSegment, KeyDoorGoalSegment


def _fresh_world():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    return w


def _names(world):
    return [type(e).__name__ for e in world.entities]


class _RightAgent(Agent):
    """Always rolls right — solves any flat, jump-free segment."""
    def act(self, observation):
        return Action.RIGHT


def _rolls_to_goal(world, player, max_steps=3000):
    for _ in range(max_steps):
        world.substep()
        if player.reached_goal:
            return True
    return False


def test_goal_segment_builds_a_goal_with_positive_width():
    w = _fresh_world()
    width = GoalSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)
    assert GoalSegment.tier == 0
    assert GoalSegment.min_abilities == frozenset()


def test_keydoorgoal_builds_key_left_of_door_plus_goal():
    w = _fresh_world()
    width = KeyDoorGoalSegment().build(w, x_offset=0.0)
    names = _names(w)
    assert names.count("Key") == 1 and names.count("Door") == 1 and "Goal" in names
    key = next(e for e in w.entities if type(e).__name__ == "Key")
    door = next(e for e in w.entities if type(e).__name__ == "Door")
    assert key.position[0] < door.position[0]
    assert width > 0


def test_goal_segment_is_solvable_by_rolling_right():
    w = _fresh_world()
    GoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    assert _rolls_to_goal(w, p)


def test_keydoorgoal_is_solvable_by_rolling_right():
    w = _fresh_world()
    KeyDoorGoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    # Rolling right collects the low key, opens the door, reaches the goal.
    assert _rolls_to_goal(w, p)
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blueball.levels.segments'`.

- [ ] **Step 3: Create `src/blueball/levels/segments.py`**

```python
"""Completion-gym segment templates.

A *segment* is a small, self-contained, solvable unit ending in a goal, built
from the existing completion chunks on the flat GROUND_Y baseline. Templates
mirror the loader's calling convention — `chunk.build(world, x_offset=...)` with
no base_y — so the per-chunk base_y signature inconsistency never bites.
Segments are the gym's analogue of Infinite Run's chunks.
"""

from __future__ import annotations

import random

from ..abilities import Ability
from .chunks.flat import Flat
from .chunks.key import KeyChunk
from .chunks.door import DoorChunk
from .chunks.goal import GoalChunk


class SegmentTemplate:
    """Base class. `build` lays chunks left-to-right from `x_offset` on the
    GROUND_Y baseline and returns the segment's total width in px."""

    tier: int = 0
    min_abilities: frozenset = frozenset()

    @classmethod
    def random(cls, rng: random.Random) -> "SegmentTemplate":
        """Instantiate with any per-segment randomization. Default: no params."""
        return cls()

    def build(self, world, x_offset: float) -> float:
        raise NotImplementedError

    @staticmethod
    def _chunk(chunk, world, x_offset: float) -> float:
        # Mirror loader.py: no base_y kwarg; every chunk defaults to GROUND_Y.
        return chunk.build(world, x_offset=x_offset)


class GoalSegment(SegmentTemplate):
    """Tier 0 — flat approach then a goal. The 'run to the goal' lesson."""

    tier = 0
    min_abilities = frozenset()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=4), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class KeyDoorGoalSegment(SegmentTemplate):
    """Tier 1 — collect a key, pass the door it unlocks, reach the goal. The key
    sits low (y_offset=40) so a rolling ball (radius 16) collects it without
    jumping; the door chunk seals the gap above the doorway, so the key is the
    only way through."""

    tier = 1
    min_abilities = frozenset()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


# Registry the sampler draws from. Extended with higher tiers in Task 3.
SEGMENT_TEMPLATES: list[type[SegmentTemplate]] = [
    GoalSegment,
    KeyDoorGoalSegment,
]
```

- [ ] **Step 4: Run it, expect pass**

Run: `pytest tests/test_segments.py -v`
Expected: 4 passed. (If `test_keydoorgoal_is_solvable_by_rolling_right` fails because the ball stalls at the door, the door is not opening on a low rolling contact — lower `door_height` is NOT the fix; verify the key is actually collected first by asserting `p.keys_held != 0` mid-run. The key/door chunks are proven in `tests/test_chunks.py`, so a failure here means the segment geometry, not the chunks.)

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/segments.py tests/test_segments.py
git commit -m "feat(gym): segment template interface + tier 0/1 templates"
```

---

### Task 3: Segment templates — Tier 2 & 3 (box/lava)

**Goal:** Add the box/lava tiers (`BoxLavaSegment`, `KeyDoorBoxLavaSegment`) and register all four templates. These require `DOUBLE_JUMP` (granted by default), so their internal solvability rides on the proven `box_lava_gap` chunk.

**Files:**
- Modify: `src/blueball/levels/segments.py`
- Test: `tests/test_segments.py` (append)

**Acceptance Criteria:**
- [ ] `BoxLavaSegment` (tier 2) builds `Lava`, `PushableBox`, `Goal`; `min_abilities == {DOUBLE_JUMP}`; `random()` varies the pit width.
- [ ] `KeyDoorBoxLavaSegment` (tier 3) builds `Key`, `Door`, `Lava`, `PushableBox`, `Goal`.
- [ ] `SEGMENT_TEMPLATES` covers tiers `{0, 1, 2, 3}`.

**Verify:** `pytest tests/test_segments.py -v` → all passed

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_segments.py`

```python
from blueball.abilities import Ability
from blueball.levels.segments import (
    BoxLavaSegment, KeyDoorBoxLavaSegment, SEGMENT_TEMPLATES,
)


def test_boxlava_segment_composition_and_requirements():
    w = _fresh_world()
    width = BoxLavaSegment(pit_tiles=6).build(w, x_offset=0.0)
    names = _names(w)
    assert "Lava" in names and "PushableBox" in names and "Goal" in names
    assert BoxLavaSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoxLavaSegment.min_abilities
    assert width > 0


def test_boxlava_random_varies_pit_width():
    import random
    widths = {BoxLavaSegment.random(random.Random(s)).pit_tiles for s in range(20)}
    assert len(widths) > 1  # not constant


def test_tier3_combo_composition():
    w = _fresh_world()
    KeyDoorBoxLavaSegment().build(w, x_offset=0.0)
    names = _names(w)
    for kind in ("Key", "Door", "Lava", "PushableBox", "Goal"):
        assert kind in names, kind
    assert KeyDoorBoxLavaSegment.tier == 3
    assert Ability.DOUBLE_JUMP in KeyDoorBoxLavaSegment.min_abilities


def test_all_four_tiers_registered():
    assert {t.tier for t in SEGMENT_TEMPLATES} == {0, 1, 2, 3}
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_segments.py -v`
Expected: FAIL — `ImportError: cannot import name 'BoxLavaSegment'`.

- [ ] **Step 3: Add the templates** — in `src/blueball/levels/segments.py`, add the import and classes, then extend the registry.

Add to the imports at the top:

```python
from .chunks.box_lava_gap import BoxLavaGap
```

Add the classes (after `KeyDoorGoalSegment`):

```python
class BoxLavaSegment(SegmentTemplate):
    """Tier 2 — shove the box into the lava pit as a stepping stone, then reach
    the goal. Requires DOUBLE_JUMP (granted by default); the box_lava_gap chunk's
    own solvability is covered by tests/test_chunks.py."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, pit_tiles: int = 6) -> None:
        self.pit_tiles = pit_tiles

    @classmethod
    def random(cls, rng: random.Random) -> "BoxLavaSegment":
        return cls(pit_tiles=rng.randint(5, 7))

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=self.pit_tiles), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class KeyDoorBoxLavaSegment(SegmentTemplate):
    """Tier 3 — unlock a door, then cross a box/lava pit, then the goal."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoxLavaGap(), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset
```

Replace the registry list at the bottom with all four:

```python
SEGMENT_TEMPLATES: list[type[SegmentTemplate]] = [
    GoalSegment,
    KeyDoorGoalSegment,
    BoxLavaSegment,
    KeyDoorBoxLavaSegment,
]
```

- [ ] **Step 4: Run it, expect pass**

Run: `pytest tests/test_segments.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/segments.py tests/test_segments.py
git commit -m "feat(gym): box/lava tier 2-3 segment templates"
```

---

### Task 4: SegmentSampler — deterministic, depth-ramped emitter

**Goal:** Add `SegmentSampler` to `segments.py` — picks templates by a depth-driven tier target (Gaussian-weighted, anti-repeat), filtered to those solvable under the granted abilities. Mirrors `ChunkSampler`.

**Files:**
- Modify: `src/blueball/config.py` (add `GYM_RAMP_PER_SEGMENT`, `GYM_SIGMA`)
- Modify: `src/blueball/levels/segments.py`
- Test: `tests/test_segment_sampler.py`

**Acceptance Criteria:**
- [ ] Same seed → identical template sequence (deterministic).
- [ ] Average tier rises with depth (the ramp).
- [ ] With no abilities granted, tier 2/3 (DOUBLE_JUMP) templates never appear.
- [ ] No two consecutive emissions are the same template type.

**Verify:** `pytest tests/test_segment_sampler.py -v` → 4 passed

**Steps:**

- [ ] **Step 1: Write the failing tests** — `tests/test_segment_sampler.py`

```python
import pytest

from blueball.abilities import Ability
from blueball.levels.segments import SegmentSampler

ALL = frozenset({Ability.DOUBLE_JUMP})


def _names(sampler, n):
    return [type(sampler.emit_next()).__name__ for _ in range(n)]


def test_deterministic_for_a_fixed_seed():
    assert _names(SegmentSampler(123, ALL), 30) == _names(SegmentSampler(123, ALL), 30)


def test_difficulty_ramps_with_depth():
    s = SegmentSampler(7, ALL)
    early = [s.emit_next().tier for _ in range(5)]
    for _ in range(40):
        s.emit_next()
    late = [s.emit_next().tier for _ in range(10)]
    assert sum(early) / len(early) < sum(late) / len(late)


def test_ability_filter_excludes_doublejump_tiers_when_not_granted():
    s = SegmentSampler(1, frozenset())  # single jump only
    tiers = {s.emit_next().tier for _ in range(60)}
    assert 2 not in tiers and 3 not in tiers


def test_no_immediate_duplicate_template_type():
    s = SegmentSampler(99, ALL)
    seq = [type(s.emit_next()).__name__ for _ in range(50)]
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def test_empty_pool_raises():
    # A hypothetical grant that satisfies nothing is impossible today (tier 0/1
    # need no abilities), so this guards the future: monkeypatch an all-DJ pool.
    import blueball.levels.segments as seg
    only_dj = [t for t in seg.SEGMENT_TEMPLATES if t.min_abilities]
    saved = seg.SEGMENT_TEMPLATES
    seg.SEGMENT_TEMPLATES = only_dj
    try:
        with pytest.raises(ValueError):
            SegmentSampler(0, frozenset())
    finally:
        seg.SEGMENT_TEMPLATES = saved
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_segment_sampler.py -v`
Expected: FAIL — `ImportError: cannot import name 'SegmentSampler'`.

- [ ] **Step 3: Add config constants** — `src/blueball/config.py`, in the Completion Gym block:

```python
GYM_RAMP_PER_SEGMENT = 0.15  # target tier climbs by this per segment of depth
GYM_SIGMA = 1.0              # Gaussian spread mixing adjacent tiers
```

- [ ] **Step 4: Add `SegmentSampler`** — in `src/blueball/levels/segments.py`.

Add to the top imports:

```python
import math
from typing import Iterator

from .. import config
```

Add the class (after the template classes, before `SEGMENT_TEMPLATES` is fine, or after — `emit_next` reads the module global at call time):

```python
class SegmentSampler:
    """Deterministic, depth-ramped segment emitter. Tier target rises with
    depth; templates are Gaussian-weighted by closeness to the target tier and
    immediate repeats are suppressed. Only templates whose `min_abilities` are
    all granted are eligible. Mirrors levels/sampler.py:ChunkSampler."""

    def __init__(
        self,
        seed: int,
        granted_abilities: frozenset,
        *,
        ramp_per_segment: float = config.GYM_RAMP_PER_SEGMENT,
        sigma: float = config.GYM_SIGMA,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.ramp = ramp_per_segment
        self.sigma = sigma
        self.depth = 0
        self._last_name: str | None = None
        granted = frozenset(granted_abilities)
        self._pool = sorted(
            (t for t in SEGMENT_TEMPLATES if t.min_abilities <= granted),
            key=lambda t: t.__name__,
        )
        if not self._pool:
            raise ValueError(
                "no segment templates are solvable under the granted abilities"
            )
        self._max_tier = max(t.tier for t in self._pool)

    def emit_next(self) -> SegmentTemplate:
        target = min(float(self._max_tier), self.depth * self.ramp)
        weights = [
            math.exp(-((t.tier - target) ** 2) / (2 * self.sigma ** 2))
            for t in self._pool
        ]
        idx = self._weighted_pick(weights)
        if self._pool[idx].__name__ == self._last_name and len(self._pool) > 1:
            weights[idx] = 0.0
            idx = self._weighted_pick(weights)
        cls = self._pool[idx]
        self._last_name = cls.__name__
        self.depth += 1
        return cls.random(self.rng)

    def __iter__(self) -> Iterator[SegmentTemplate]:
        while True:
            yield self.emit_next()

    def _weighted_pick(self, weights: list[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1
```

- [ ] **Step 5: Run it, expect pass**

Run: `pytest tests/test_segment_sampler.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/config.py src/blueball/levels/segments.py tests/test_segment_sampler.py
git commit -m "feat(gym): depth-ramped deterministic SegmentSampler"
```

---

### Task 5: SegmentStream — materialize ahead, cull behind, record boundaries

**Goal:** Add `SegmentStream` (sibling of `TerrainStream`) that streams segments from a `SegmentSampler`, culls those behind, and records every segment's end-x in `segment_ends` (never culled, so the evaluator can count crossings).

**Files:**
- Modify: `src/blueball/config.py` (add `GYM_LOAD_AHEAD`, `GYM_LOAD_BEHIND`, `GYM_INITIAL_SEGMENTS`)
- Create: `src/blueball/levels/segment_stream.py`
- Test: `tests/test_segment_stream.py`

**Acceptance Criteria:**
- [ ] Construction lays a spawn-footing `Flat` at x=0 (not counted) and builds `GYM_INITIAL_SEGMENTS` segments; `segment_ends` is non-empty and strictly increasing.
- [ ] `maintain(player_x)` builds ahead to keep `load_ahead` px materialized.
- [ ] `maintain` culls units fully behind `player_x - load_behind`; no stale `_shape_to_entity` links; `segment_ends` history is preserved (only grows).

**Verify:** `pytest tests/test_segment_stream.py -v` → 4 passed

**Steps:**

- [ ] **Step 1: Write the failing tests** — `tests/test_segment_stream.py`

```python
import pytest

from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.abilities import Ability
from blueball.levels.segment_stream import SegmentStream

ALL = frozenset({Ability.DOUBLE_JUMP})


def _world():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    return w


def test_construction_builds_initial_segments_with_increasing_boundaries():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    assert len(s.segment_ends) >= 4
    assert s.segment_ends == sorted(s.segment_ends)
    assert len(set(s.segment_ends)) == len(s.segment_ends)  # strictly increasing
    assert len(w.entities) > 0


def test_maintain_builds_ahead():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    n0 = len(s.segment_ends)
    s.maintain(player_x=s.build_x)  # ask for terrain at the frontier
    assert len(s.segment_ends) > n0


def test_maintain_culls_units_fully_behind_the_player():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    s.maintain(player_x=6000.0)
    ends_snapshot = list(s.segment_ends)
    s.maintain(player_x=12000.0)
    cutoff = 12000.0 - s.load_behind
    assert all(u["x_end"] >= cutoff for u in s.built)          # nothing behind remains
    assert s.segment_ends[: len(ends_snapshot)] == ends_snapshot  # history preserved
    assert all(sh in w.space.shapes for sh in list(w._shape_to_entity))  # no stale links


def test_cull_removes_entities_and_shapes_from_space():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    s.maintain(player_x=4000.0)
    ents_peak = len(w.entities)
    s.maintain(player_x=20000.0)  # leave everything earlier far behind
    # Far-behind units are gone, so live entity count is bounded well under peak.
    assert len(w.entities) < ents_peak
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_segment_stream.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blueball.levels.segment_stream'`.

- [ ] **Step 3: Add config constants** — `src/blueball/config.py`, Completion Gym block:

```python
GYM_LOAD_AHEAD = 2000.0    # px of segments kept materialized ahead of the ball
GYM_LOAD_BEHIND = 800.0    # px behind the ball before a unit is culled
GYM_INITIAL_SEGMENTS = 4   # segments built at construction (after spawn footing)
```

- [ ] **Step 4: Create `src/blueball/levels/segment_stream.py`**

```python
"""Completion-gym segment streaming — pygame-free sibling of TerrainStream.

Materializes solvable goal-segments from a SegmentSampler ahead of the ball and
culls those behind it. Records the cumulative end-x of every segment ever built
(cheap floats, never culled) in `segment_ends`, so the evaluator can count how
many segments the ball has passed even after their physics objects are removed.
Builds on the flat GROUND_Y baseline (no base_y threading needed).
"""

from __future__ import annotations

from .. import config
from .chunks.flat import Flat
from .segments import SegmentSampler


class SegmentStream:
    def __init__(
        self,
        world,
        seed: int,
        granted_abilities: frozenset,
        *,
        load_ahead: float = config.GYM_LOAD_AHEAD,
        load_behind: float = config.GYM_LOAD_BEHIND,
        initial_segments: int = config.GYM_INITIAL_SEGMENTS,
    ) -> None:
        self.world = world
        self.load_ahead = load_ahead
        self.load_behind = load_behind
        self.sampler = SegmentSampler(int(seed), frozenset(granted_abilities))
        self.build_x: float = 0.0
        self.built: list[dict] = []
        self.segment_ends: list[float] = []

        # Spawn footing: a guaranteed flat at x=0 (recorded for culling, but NOT
        # a counted segment — it has no goal).
        self._materialize(Flat(width_tiles=4))
        for _ in range(initial_segments):
            self._build_next_segment()

    def _materialize(self, builder) -> float:
        """Build `builder` (a chunk or segment exposing
        `build(world, x_offset)`) at the cursor, recording exactly what got added
        to the space so it can be culled later. Returns the width."""
        pre_shapes = set(self.world.space.shapes)
        pre_bodies = set(self.world.space.bodies)
        pre_entities = set(self.world.entities)
        pre_constraints = set(self.world.space.constraints)

        width = builder.build(self.world, x_offset=self.build_x)

        self.built.append({
            "x_end": self.build_x + width,
            "shapes": set(self.world.space.shapes) - pre_shapes,
            "bodies": set(self.world.space.bodies) - pre_bodies,
            "entities": set(self.world.entities) - pre_entities,
            "constraints": set(self.world.space.constraints) - pre_constraints,
        })
        self.build_x += width
        return width

    def _build_next_segment(self) -> None:
        template = self.sampler.emit_next()
        self._materialize(template)
        self.segment_ends.append(self.build_x)  # cursor now sits at the segment end

    def maintain(self, player_x: float) -> None:
        """Per-tick: build ahead to keep load_ahead px materialized, cull units
        fully behind player_x - load_behind."""
        while self.build_x < player_x + self.load_ahead:
            self._build_next_segment()
        cutoff = player_x - self.load_behind
        while self.built and self.built[0]["x_end"] < cutoff:
            info = self.built.pop(0)
            for shape in info["shapes"]:
                if shape in self.world.space.shapes:
                    self.world.space.remove(shape)
                self.world._shape_to_entity.pop(shape, None)
            for constraint in info["constraints"]:
                if constraint in self.world.space.constraints:
                    self.world.space.remove(constraint)
            for body in info["bodies"]:
                if body is self.world.space.static_body:
                    continue
                if body in self.world.space.bodies:
                    self.world.space.remove(body)
            for entity in info["entities"]:
                if entity in self.world.entities:
                    self.world.entities.remove(entity)
```

- [ ] **Step 5: Run it, expect pass**

Run: `pytest tests/test_segment_stream.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/config.py src/blueball/levels/segment_stream.py tests/test_segment_stream.py
git commit -m "feat(gym): SegmentStream materialize/cull with boundary history"
```

---

### Task 6: evaluate_gym + episode dispatch

**Goal:** Add the gym evaluator (count-and-continue, per-segment key clear, cumulative-key tracking, never stops on a goal) and wire `kind="gym"` through `EpisodeSpec` / `evaluate_episodes`, plus a `gym_episodes` constructor.

**Files:**
- Modify: `src/blueball/config.py` (add `GYM_SPAWN`, `GYM_MAX_STEPS`)
- Modify: `src/blueball/ai/episodes.py` (`EpisodeSpec.abilities`, `gym_episodes`)
- Modify: `src/blueball/ai/trainer.py` (`evaluate_gym`, dispatch)
- Test: `tests/test_gym_eval.py`

**Acceptance Criteria:**
- [ ] `evaluate_gym((idx, genome, seed, world_seed, max_steps, abilities))` returns `(idx, finite_float)` and never breaks on a reached goal.
- [ ] A roll-right agent over a single-jump gym crosses ≥2 boundaries (chain continues past the first goal) and `keys_held` is cleared between segments while cumulative keys still accrue.
- [ ] `EpisodeSpec(kind="gym")` is dispatched to `evaluate_gym` by `evaluate_episodes`.

**Verify:** `pytest tests/test_gym_eval.py -v` → all passed

**Steps:**

- [ ] **Step 1: Write the failing tests** — `tests/test_gym_eval.py`

```python
import bisect

import numpy as np
import pytest

from blueball import config
from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.levels.segment_stream import SegmentStream
from blueball.ai.trainer import evaluate_gym
from blueball.ai.episodes import gym_episodes, EpisodeSpec


class _RightAgent(Agent):
    def act(self, observation):
        return Action.RIGHT


def _drive(stream, world, player, max_steps):
    """The exact bookkeeping loop evaluate_gym runs, exposed for assertions."""
    max_x = player.body.position.x
    cleared = 0
    cumulative_keys = 0
    prev = 0
    for _ in range(max_steps):
        stream.maintain(player.body.position.x)
        world.substep()
        max_x = max(max_x, player.body.position.x)
        cur = bin(player.keys_held).count("1")
        if cur > prev:
            cumulative_keys += cur - prev
        prev = cur
        n = bisect.bisect_right(stream.segment_ends, max_x)
        if n > cleared:
            cleared = n
            player.keys_held = 0
            prev = 0
        if player.dead:
            break
    return cleared, cumulative_keys


def test_chain_continues_past_first_goal_and_clears_keys():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    granted = frozenset()  # single jump → only roll-solvable tier 0/1 segments
    stream = SegmentStream(w, seed=3, granted_abilities=granted)
    p = Player(agent=_RightAgent(), spawn_xy=config.GYM_SPAWN, abilities=set(granted))
    w.add_entity(p)
    cleared, cumulative_keys = _drive(stream, w, p, max_steps=8000)
    assert cleared >= 2                       # did NOT stop at the first goal
    assert cumulative_keys >= 1               # collected ≥1 key across the chain
    assert bin(p.keys_held).count("1") <= 1   # keys cleared between segments


def test_evaluate_gym_smoke_returns_finite_fitness():
    genome = np.random.default_rng(0).standard_normal(
        __import__("blueball.ai.ftnn", fromlist=["GENOME_SIZE"]).GENOME_SIZE
    ).astype(np.float32)
    idx, f = evaluate_gym((0, genome, 11, config.DEFAULT_SEED, 500, ("double_jump",)))
    assert idx == 0
    assert np.isfinite(f)


def test_evaluate_episodes_dispatches_gym():
    # A gym EpisodeSpec routed through evaluate_episodes should not raise and
    # should return a finite score.
    from blueball.ai.trainer import evaluate_episodes
    genome = np.random.default_rng(1).standard_normal(
        __import__("blueball.ai.ftnn", fromlist=["GENOME_SIZE"]).GENOME_SIZE
    ).astype(np.float32)
    eps = gym_episodes([11], world_seed=config.DEFAULT_SEED, max_steps=400,
                       abilities=("double_jump",))
    idx, score = evaluate_episodes((0, genome, tuple(eps), 1.0, "mean_std"))
    assert idx == 0 and np.isfinite(score)
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_gym_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_gym'` / `gym_episodes`.

- [ ] **Step 3: Add config constants** — `src/blueball/config.py`, Completion Gym block:

```python
GYM_SPAWN = (80.0, 540.0)  # ball spawn; lands on the streamer's x=0 flat footing
GYM_MAX_STEPS = 6000       # higher than Infinite Run: puzzles cover less x/step
```

- [ ] **Step 4: Extend `EpisodeSpec` and add `gym_episodes`** — `src/blueball/ai/episodes.py`

Add the field (last, defaulted) to the dataclass:

```python
    norm: float = 1.0         # divisor applied to this episode's raw fitness
    abilities: tuple[str, ...] = ()  # gym: granted ability names; () elsewhere
```

Add the constructor (next to `infinite_episodes`):

```python
def gym_episodes(seeds: Sequence[int], world_seed: int, max_steps: int,
                 abilities: Sequence[str]) -> list[EpisodeSpec]:
    """One completion-gym EpisodeSpec per chain seed. `abilities` is the granted
    ability-name set, shared across all seeds (norm=1.0: all gym chains share the
    same reward scale)."""
    ab = tuple(str(a) for a in abilities)
    return [
        EpisodeSpec(kind="gym", seed=int(s), level_path=None,
                    world_seed=int(world_seed), max_steps=int(max_steps),
                    abilities=ab)
        for s in seeds
    ]
```

- [ ] **Step 5: Add `evaluate_gym` and dispatch** — `src/blueball/ai/trainer.py`

Add to the module imports near the top:

```python
import bisect

from ..abilities import Ability
from ..levels.segment_stream import SegmentStream
```

Add the spawn constant near `INFINITE_SPAWN`:

```python
GYM_SPAWN = config.GYM_SPAWN
```

Add the evaluator (after `evaluate_infinite`):

```python
def evaluate_gym(args: tuple) -> tuple[int, float]:
    """One genome -> one fitness on a streamed completion-gym chain. Picklable
    in/out for multiprocessing.Pool. Args is
    (idx, genome, seed, world_seed, max_steps, abilities), where `abilities` is a
    tuple of Ability *name* strings.

    Unlike the goal-terminal evaluators, this NEVER stops on a goal. It counts
    segment clears by how far the ball's max_x has passed the segment boundaries
    (a locked door can't be passed without its key, so crossing a boundary ==
    solving that segment), clears keys_held at each crossed boundary (so a reused
    key_id behind the next door must be re-earned), and tracks cumulative keys
    across those clears so the key reward survives the clearing.
    """
    idx, genome, seed, world_seed, max_steps, abilities = args
    granted = frozenset(Ability(a) for a in abilities)

    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    stream = SegmentStream(world, int(seed), granted)

    spawn_x, spawn_y = config.GYM_SPAWN
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y),
                    abilities=set(granted))
    world.add_entity(player)

    max_x = spawn_x
    cleared = 0
    cumulative_keys = 0
    prev_keys_popcount = 0
    steps = 0
    while steps < max_steps:
        stream.maintain(player.body.position.x)
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x

        # Accumulate newly collected keys (popcount only rises between clears).
        cur = bin(player.keys_held).count("1")
        if cur > prev_keys_popcount:
            cumulative_keys += cur - prev_keys_popcount
        prev_keys_popcount = cur

        # Count boundary crossings; reset the key scope when a segment is cleared.
        new_cleared = bisect.bisect_right(stream.segment_ends, max_x)
        if new_cleared > cleared:
            cleared = new_cleared
            player.keys_held = 0
            prev_keys_popcount = 0

        if player.dead:
            break

    f = fitness(FitnessInputs(
        progress_x=float(max_x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=False,
        died=bool(player.dead),
        steps_taken=steps,
        keys_collected=int(cumulative_keys),
        level_width=0.0,
        segments_cleared=int(cleared),
    ))
    return idx, float(f)
```

Wire the dispatch in `evaluate_episodes` — replace the `if/else` with an `if/elif/else`:

```python
    for ep in episodes:
        if ep.kind == "infinite":
            _, raw = evaluate_infinite(
                (idx, genome, ep.seed, ep.world_seed, ep.max_steps))
        elif ep.kind == "gym":
            _, raw = evaluate_gym(
                (idx, genome, ep.seed, ep.world_seed, ep.max_steps, ep.abilities))
        else:
            _, raw = evaluate(
                (idx, genome, ep.world_seed, ep.level_path, ep.max_steps))
        scores.append(raw / ep.norm)
```

- [ ] **Step 6: Run it, expect pass**

Run: `pytest tests/test_gym_eval.py -v`
Expected: all passed. (If `test_chain_continues_past_first_goal_and_clears_keys` clears fewer than 2, raise `max_steps` or check the door opens — see Task 2 Step 4 note. If `cumulative_keys` is 0, seed 3 produced no early tier-1 segment; pick a seed whose first ~6 segments include ≥1 `KeyDoorGoalSegment`.)

- [ ] **Step 7: Regression check on the multi-episode path**

Run: `pytest tests/test_ai_multiepisode.py -q`
Expected: pass (new `EpisodeSpec.abilities` defaults to `()`; infinite/static dispatch unchanged).

- [ ] **Step 8: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/episodes.py src/blueball/ai/trainer.py tests/test_gym_eval.py
git commit -m "feat(gym): evaluate_gym (count-and-continue) + episode dispatch"
```

---

### Task 7: CLI + persistence + reproducibility smoke

**Goal:** Add `train_completion_gym.py` (mirrors `train_infinite.py`) with a `gym` run-folder key, and a smoke test that a tiny gym run trains and is reproducible.

**Files:**
- Modify: `src/blueball/config.py` (add `GYM_SEED`, `GYM_DEFAULT_NUM_SEEDS`)
- Modify: `src/blueball/ai/persistence.py` (`run_dir_name` gym key)
- Create: `train_completion_gym.py`
- Test: `tests/test_gym_smoke.py`

**Acceptance Criteria:**
- [ ] `run_dir_name(gym_seed=7, world_seed=1, timestamp="T", num_seeds=1)` → starts with `gym7_w1_`; `num_seeds=3` → `gym7x3_w1_`.
- [ ] A `train(episodes=gym_episodes(...), pop_size=4, generations=2, map_fn=map)` run returns a finite best and is byte-identical across two runs with the same seeds.
- [ ] `python train_completion_gym.py --help` exits 0.

**Verify:** `pytest tests/test_gym_smoke.py -v` → 2 passed; `python train_completion_gym.py --help` → exit 0

**Steps:**

- [ ] **Step 1: Write the failing tests** — `tests/test_gym_smoke.py`

```python
import numpy as np

from blueball import config
from blueball.ai.episodes import gym_episodes
from blueball.ai.persistence import run_dir_name
from blueball.ai.trainer import train


def test_run_dir_name_gym_key():
    assert run_dir_name(gym_seed=7, world_seed=1, timestamp="T",
                        num_seeds=1).startswith("gym7_w1_")
    assert run_dir_name(gym_seed=7, world_seed=1, timestamp="T",
                        num_seeds=3).startswith("gym7x3_w1_")


def test_gym_training_runs_and_is_reproducible():
    eps = gym_episodes([7], world_seed=config.DEFAULT_SEED, max_steps=400,
                       abilities=("double_jump",))
    r1 = train(pop_size=4, generations=2, episodes=eps, ga_seed=0,
               world_seed=config.DEFAULT_SEED, map_fn=map)
    r2 = train(pop_size=4, generations=2, episodes=eps, ga_seed=0,
               world_seed=config.DEFAULT_SEED, map_fn=map)
    assert np.array_equal(r1.best_genome, r2.best_genome)
    assert np.isfinite(r1.history[-1]["best"])
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_gym_smoke.py -v`
Expected: FAIL — `run_dir_name() got an unexpected keyword argument 'gym_seed'`.

- [ ] **Step 3: Add config constants** — `src/blueball/config.py`, Completion Gym block:

```python
GYM_SEED = 4242            # default base gym chain seed
GYM_DEFAULT_NUM_SEEDS = 8  # multi-seed by default: gym chains must generalize
```

- [ ] **Step 4: Add the gym key to `run_dir_name`** — `src/blueball/ai/persistence.py`

Add a `gym_seed` parameter and a branch. Update the signature and the branch chain:

```python
def run_dir_name(
    *,
    world_seed: int,
    timestamp: str,
    infinite_seed: int | None = None,
    gym_seed: int | None = None,
    level_name: str | None = None,
    num_seeds: int = 1,
    num_levels: int | None = None,
    curriculum: bool = False,
) -> str:
    if curriculum:
        key = f"{level_name or 'level'}curr"
    elif num_levels is not None:
        key = f"lvls{num_levels}"
    elif gym_seed is not None:
        key = f"gym{gym_seed}" if num_seeds <= 1 else f"gym{gym_seed}x{num_seeds}"
    elif infinite_seed is not None:
        key = f"inf{infinite_seed}" if num_seeds <= 1 else f"inf{infinite_seed}x{num_seeds}"
    else:
        key = level_name or "level"
    return f"{key}_w{world_seed}_{timestamp}"
```

Also add the `gym<seed>` line to the docstring's example block (keep docs honest):

```
    gym4242x8_w1_<ts>    multi-seed completion-gym run
```

- [ ] **Step 5: Run the unit test for the key, expect pass**

Run: `pytest tests/test_gym_smoke.py::test_run_dir_name_gym_key -v`
Expected: 1 passed.

- [ ] **Step 6: Create `train_completion_gym.py`** (repo root)

```python
"""Headless training on the Completion Gym — an endless chain of solvable,
goal-terminated segments (keys, doors, boxes, lava) with difficulty ramping by
depth. Trains the completion mechanics that Infinite Run never exercises.

    python train_completion_gym.py                    # default multi-seed run
    python train_completion_gym.py --num-seeds 16     # more chains -> generalize
    python train_completion_gym.py --seeds 3,7,11     # explicit gym seeds
    python train_completion_gym.py --abilities ''     # single-jump gym (tier 0/1)
    python train_completion_gym.py --gens 80

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import generate_seeds, gym_episodes
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.GYM_MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--gym-seed", type=int, default=config.GYM_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--num-seeds", type=int, default=config.GYM_DEFAULT_NUM_SEEDS,
                        help="train across N gym seeds derived from --gym-seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="explicit comma-separated gym seeds (overrides --num-seeds)")
    parser.add_argument("--abilities", type=str, default="double_jump",
                        help="comma-separated granted abilities; '' for single jump")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = generate_seeds(args.gym_seed, args.num_seeds)

    abilities = tuple(a.strip() for a in args.abilities.split(",") if a.strip())

    episodes = gym_episodes(seeds, world_seed=args.world_seed,
                            max_steps=args.max_steps, abilities=abilities)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        gym_seed=seeds[0], world_seed=args.world_seed,
        timestamp=timestamp, num_seeds=len(seeds),
    )

    print(
        f"Training {args.pop}x{args.gens} on Completion Gym seeds={seeds} "
        f"abilities={abilities or '(single jump)'} world={args.world_seed}\n  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train(
            pop_size=args.pop,
            generations=args.gens,
            episodes=episodes,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the smoke tests + CLI help**

Run: `pytest tests/test_gym_smoke.py -v`
Expected: 2 passed.

Run: `python train_completion_gym.py --help`
Expected: argparse help printed, exit 0.

- [ ] **Step 8: Full suite regression**

Run: `pytest -q`
Expected: the whole suite passes (no existing test broken by the gym additions).

- [ ] **Step 9: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/persistence.py train_completion_gym.py tests/test_gym_smoke.py
git commit -m "feat(gym): train_completion_gym CLI + gym run-dir key"
```

---

## Post-implementation (not tasks — for the human)

- **First real run:** `python train_completion_gym.py --num-seeds 8 --gens 100` and watch `mean`/`best` climb; inspect `genomes/gym4242x8_w1_<ts>/run.json`.
- **Tuning knobs** (all in `config.py`): `GYM_SEGMENT_BONUS`, `GYM_RAMP_PER_SEGMENT`, `GYM_SIGMA`, `GYM_MAX_STEPS`, `GYM_LOAD_AHEAD/BEHIND`, seeds-per-genome.
- **Transfer check:** evaluate a gym-trained `final_best.npy` on the campaign levels to see if completion mechanics carried over (the whole point).
- **Deferred (YAGNI for v1):** a plain pushable-box-step segment (no lava) and `goal_vault` multi-key segments — add as new templates later; the sampler/stream/evaluator need no changes.
