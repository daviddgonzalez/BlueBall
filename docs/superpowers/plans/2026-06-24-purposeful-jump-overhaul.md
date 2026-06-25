# Purposeful-Jump Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop GA-trained agents from bounce-spamming by (a) giving the network jump-state perception, (b) rewarding efficiency in goal modes and gap-gated purposeful jumps everywhere, then A/B-validating against the stuck maze-curriculum and gym specialists.

**Architecture:** Keep the FTNN trainer; the NEAT escape hatch stays deferred behind the `Agent` interface. Add 2 jump-state inputs to the 35-input observation (→37, retrain from scratch). Add two additive fitness terms: an efficiency ratio `progress_x/steps` active only when `level_width > 0`, and a `JUMP_GAP_BONUS * purposeful_jumps` term where a purposeful jump is a ground takeoff with a real ledge ahead (forward-down probe). The `Player` accumulates per-episode counters (`purposeful_jumps`, `jumps_fired`, `airborne_steps`); every evaluator reads them off the player.

**Tech Stack:** Python, numpy (pure-numpy FTNN), pymunk (physics + raycasts), pytest. Spec: `docs/superpowers/specs/2026-06-24-purposeful-jump-overhaul-design.md`.

---

### Task 0: Expose jump-state query on JumpController

**Goal:** Add read-only accessors so callers can ask "is an air-jump available?" and "would a jump fire right now?" without mutating controller state.

**Files:**
- Modify: `src/blueball/input_feel.py` (class `JumpController`)
- Test: `tests/test_input_feel.py`

**Acceptance Criteria:**
- [ ] `JumpController.air_jumps_remaining` property returns the current air-jump stock.
- [ ] `JumpController.can_fire(grounded: bool)` returns True iff grounded, coyote window open, or an air-jump is available; it does not mutate state.

**Verify:** `python -m pytest tests/test_input_feel.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_input_feel.py`:

```python
def test_can_fire_when_grounded():
    jc = JumpController(abilities=set())
    assert jc.can_fire(grounded=True) is True


def test_can_fire_false_airborne_no_doublejump():
    jc = JumpController(abilities=set())
    jc._air_jumps_remaining = 0
    jc._coyote_remaining = 0.0
    assert jc.can_fire(grounded=False) is False


def test_can_fire_true_with_air_jump():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    jc._coyote_remaining = 0.0
    assert jc._air_jumps_remaining == 1
    assert jc.can_fire(grounded=False) is True


def test_can_fire_does_not_mutate():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    before = jc._air_jumps_remaining
    jc.can_fire(grounded=False)
    assert jc._air_jumps_remaining == before


def test_air_jumps_remaining_property():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    assert jc.air_jumps_remaining == 1
    jc._air_jumps_remaining = 0
    assert jc.air_jumps_remaining == 0
```

(`Ability` is already imported in `tests/test_input_feel.py`; confirm at top, add `from blueball.abilities import Ability` if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_input_feel.py -q`
Expected: FAIL — `AttributeError: 'JumpController' object has no attribute 'can_fire'`

- [ ] **Step 3: Implement** — in `src/blueball/input_feel.py`, add inside `class JumpController` (e.g. just after `_max_air_jumps`):

```python
    @property
    def air_jumps_remaining(self) -> int:
        """Current air-jump stock (read-only view of internal counter)."""
        return self._air_jumps_remaining

    def can_fire(self, grounded: bool) -> bool:
        """True iff a jump would be initiable this tick: on the ground, inside
        the coyote window, or with an air-jump still in stock. Pure read — does
        not consume timers or counters. Used to feed the observation; it is NOT
        the firing authority (that stays in `tick`)."""
        return bool(grounded) or self._coyote_remaining > 0.0 or self._air_jumps_remaining > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_input_feel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blueball/input_feel.py tests/test_input_feel.py
git commit -m "feat(ai): expose JumpController.can_fire + air_jumps_remaining"
```

---

### Task 1: Add jump-state inputs to the observation vector (35→37)

**Goal:** Extend `Observation` with two jump-state fields and pack them into the FTNN input vector, growing `INPUT_SIZE` 35→37 and `GENOME_SIZE` 510→534.

**Files:**
- Modify: `src/blueball/agent.py` (`Observation` dataclass)
- Modify: `src/blueball/ai/observation.py` (layout doc, offsets, `INPUT_SIZE`, `observation_to_inputs`)
- Modify: `src/blueball/ai/ftnn.py` (docstring numbers only — `FTNN_INPUTS`/`GENOME_SIZE` auto-derive)
- Test: `tests/test_observation.py` (new)

**Acceptance Criteria:**
- [ ] `Observation` has `air_jumps_remaining: float = 0.0` and `can_jump_now: bool = False` (defaulted so existing constructors keep working).
- [ ] `INPUT_SIZE == 37` and `ai.ftnn.GENOME_SIZE == 534`.
- [ ] `observation_to_inputs` writes the normalized air-jump value at index 35 and the can-jump flag (0.0/1.0) at index 36.

**Verify:** `python -m pytest tests/test_observation.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — create `tests/test_observation.py`:

```python
import numpy as np

from blueball.agent import HitType, Observation
from blueball.ai.observation import INPUT_SIZE, observation_to_inputs
from blueball.ai.ftnn import GENOME_SIZE


def _blank_obs(**overrides):
    base = dict(
        rays=np.ones(8, dtype=np.float32),
        ray_hit_types=np.zeros(8, dtype=np.int8),
        vel=np.zeros(2, dtype=np.float32),
        ang_vel=0.0,
        grounded=False,
        nearest_pickup=None,
        nearest_hazard=None,
        abilities=0,
        keys_held=0,
    )
    base.update(overrides)
    return Observation(**base)


def test_input_and_genome_sizes():
    assert INPUT_SIZE == 37
    assert GENOME_SIZE == 534


def test_jump_state_packed_at_tail():
    obs = _blank_obs(air_jumps_remaining=1.0, can_jump_now=True)
    x = observation_to_inputs(obs)
    assert x.shape == (37,)
    assert x[35] == 1.0   # air_jumps_remaining (normalized)
    assert x[36] == 1.0   # can_jump_now flag


def test_jump_state_defaults_zero():
    x = observation_to_inputs(_blank_obs())
    assert x[35] == 0.0
    assert x[36] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_observation.py -q`
Expected: FAIL — `assert INPUT_SIZE == 37` fails (currently 35) / `TypeError` on unexpected kwargs.

- [ ] **Step 3a: Extend the dataclass** — in `src/blueball/agent.py`, add two fields at the end of `Observation` (keep them last, with defaults):

```python
@dataclass(frozen=True)
class Observation:
    rays: np.ndarray              # shape (8,), float32, in [0, 1]; 1.0 = miss
    ray_hit_types: np.ndarray     # shape (8,), int8 HitType values
    vel: np.ndarray               # shape (2,), float32
    ang_vel: float
    grounded: bool
    nearest_pickup: Optional[tuple[float, float]]
    nearest_hazard: Optional[tuple[float, float]]
    abilities: int                # bitfield, ability enum ordinal
    keys_held: int                # bitfield
    air_jumps_remaining: float = 0.0  # normalized [0,1]; 1.0 = an air-jump is in stock
    can_jump_now: bool = False        # True iff a jump would be initiable this tick
```

- [ ] **Step 3b: Add offsets + sizing** — in `src/blueball/ai/observation.py`, after the `INPUT_SIZE = _KEYS_OFFSET + KEY_BITS` line, replace that single line with:

```python
_AIRJUMP_OFFSET = _KEYS_OFFSET + KEY_BITS    # 35
_CANJUMP_OFFSET = _AIRJUMP_OFFSET + 1        # 36

INPUT_SIZE = _CANJUMP_OFFSET + 1             # 37
```

- [ ] **Step 3c: Pack the values** — in `observation_to_inputs`, just before `return x`, add:

```python
    x[_AIRJUMP_OFFSET] = _clamp_unit(obs.air_jumps_remaining)
    x[_CANJUMP_OFFSET] = 1.0 if obs.can_jump_now else 0.0
```

- [ ] **Step 3d: Update layout docs** — extend the module docstring layout table in `observation.py` with rows `35 air_jumps_remaining` and `36 can_jump_now`, and update the header `INPUT_SIZE = 35` note to 37. In `ftnn.py`, update the docstring numbers (`35 inputs` → `37`, `35*12=420` → `37*12=444`, `GENOME_SIZE = ... = 510` → `534`). These are comments only; code derives the sizes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_observation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blueball/agent.py src/blueball/ai/observation.py src/blueball/ai/ftnn.py tests/test_observation.py
git commit -m "feat(ai): add jump-state inputs to observation (INPUT_SIZE 35->37)"
```

---

### Task 2: Populate jump-state in Player._observe

**Goal:** Have `Player._observe` fill the two new observation fields from its `JumpController` so live play and evaluation feed the network real jump-state.

**Files:**
- Modify: `src/blueball/entities/player.py` (`_observe`)
- Test: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] A grounded player with `DOUBLE_JUMP` produces `can_jump_now=True` and `air_jumps_remaining==1.0`.
- [ ] A player without `DOUBLE_JUMP` produces `air_jumps_remaining==0.0`.

**Verify:** `python -m pytest tests/test_player.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — append to `tests/test_player.py`:

```python
def test_observe_reports_jump_state_with_double_jump():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100),
               abilities={Ability.DOUBLE_JUMP})
    obs = p._observe()
    assert obs.air_jumps_remaining == 1.0
    assert obs.can_jump_now is True  # grounded defaults True off-world? see note


def test_observe_reports_no_air_jump_without_ability():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    obs = p._observe()
    assert obs.air_jumps_remaining == 0.0
```

NOTE on `can_jump_now`: off-world `p.grounded` may be False. `can_jump_now` is still True here because a fresh `JumpController` with `DOUBLE_JUMP` starts with an air-jump in stock (`_air_jumps_remaining == 1`), so `can_fire(False)` is True. If `p.grounded` semantics make this brittle, assert on `air_jumps_remaining` only and drop the `can_jump_now` assertion — the packing is already covered in Task 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player.py -k observe_reports -q`
Expected: FAIL — `air_jumps_remaining` is the default `0.0` (not yet populated).

- [ ] **Step 3: Implement** — in `src/blueball/entities/player.py`, in the `Observation(...)` constructed at the end of `_observe`, add the two fields:

```python
        return Observation(
            rays=rays,
            ray_hit_types=hit_types,
            vel=np.array(
                [self.body.velocity.x, self.body.velocity.y], dtype=np.float32
            ),
            ang_vel=self.body.angular_velocity,
            grounded=self.grounded,
            nearest_pickup=nearest_pickup,
            nearest_hazard=nearest_hazard,
            abilities=_abilities_to_bitfield(self.abilities),
            keys_held=self.keys_held,
            air_jumps_remaining=float(min(1, self.jump_ctrl.air_jumps_remaining)),
            can_jump_now=self.jump_ctrl.can_fire(self.grounded),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player.py -k observe_reports -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat(ai): Player._observe reports jump-state to the network"
```

---

### Task 3: Gap-probe + per-episode counters on Player

**Goal:** Add a forward-down ledge probe and per-episode counters (`purposeful_jumps`, `jumps_fired`, `airborne_steps`) so a ground takeoff over a ledge is counted, flat-ground bouncing is not, and bounce metrics are measurable.

**Files:**
- Modify: `src/blueball/config.py` (new constants)
- Modify: `src/blueball/entities/player.py` (`__init__`, `update`, new `_ledge_ahead`)
- Test: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] New player exposes `purposeful_jumps == 0`, `jumps_fired == 0`, `airborne_steps == 0`.
- [ ] `_ledge_ahead(direction)` returns True when the forward-down probe finds no ground, False when it hits ground, and False when `direction == 0` or no world.
- [ ] In a world with a floor that has a gap, a right-moving ground jump taken at the ledge edge increments `purposeful_jumps`; a jump over solid floor does not.

**Verify:** `python -m pytest tests/test_player.py -q` → all pass

**Steps:**

- [ ] **Step 1: Add config constants** — in `src/blueball/config.py`, under the `# Jump` section (after `JUMP_CUT_FACTOR`):

```python
# Gap-gated "purposeful jump" probe. At a ground takeoff we cast a short
# downward ray this far ahead (in the travel direction); if it finds no ground
# within the depth, the jump was taken at a ledge/gap and earns JUMP_GAP_BONUS.
# Un-farmable on flat ground (probe hits ground -> no credit). Tunable.
GAP_PROBE_AHEAD_PX = 48.0
GAP_PROBE_DEPTH_PX = 120.0
```

- [ ] **Step 2: Write the failing tests** — append to `tests/test_player.py`:

```python
def _make_world_with_gap(gap_start=200.0, gap_end=400.0):
    """Floor at y=600 from -2000..gap_start and gap_end..2000, gap in between."""
    w = World()
    static = w.space.static_body
    left = pymunk.Segment(static, (-2000, 600), (gap_start, 600), 5)
    right = pymunk.Segment(static, (gap_end, 600), (2000, 600), 5)
    for seg in (left, right):
        seg.friction = 1.0
        w.space.add(seg)
    return w


def test_player_counters_start_zero():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.purposeful_jumps == 0
    assert p.jumps_fired == 0
    assert p.airborne_steps == 0


def test_ledge_ahead_true_over_gap():
    w = _make_world_with_gap()
    # Stand so that pos + AHEAD lands inside the gap (200..400).
    x = 200.0 - config.GAP_PROBE_AHEAD_PX + 10.0
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(x, 580))
    w.add_entity(p)
    assert p._ledge_ahead(direction=1) is True


def test_ledge_ahead_false_over_floor():
    w = _make_world_with_gap()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(0.0, 580))
    w.add_entity(p)
    assert p._ledge_ahead(direction=1) is False


def test_ledge_ahead_false_without_direction():
    w = _make_world_with_gap()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(0.0, 580))
    w.add_entity(p)
    assert p._ledge_ahead(direction=0) is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_player.py -k "counters_start_zero or ledge_ahead" -q`
Expected: FAIL — `AttributeError: 'Player' object has no attribute 'purposeful_jumps'` / `_ledge_ahead`.

- [ ] **Step 4a: Init counters** — in `src/blueball/entities/player.py` `__init__`, alongside `self.jump_ctrl = JumpController(...)`:

```python
        # Per-episode behavior counters (read by the GA evaluators / fitness).
        self.purposeful_jumps = 0   # ground takeoffs taken at a real ledge/gap
        self.jumps_fired = 0        # total jumps that fired
        self.airborne_steps = 0     # update() ticks spent not grounded
```

- [ ] **Step 4b: Add the probe helper** — add a method on `Player` (near `_observe`):

```python
    def _ledge_ahead(self, direction: int) -> bool:
        """True if a short forward-down probe finds NO ground ahead — i.e. the
        player is at a ledge/gap edge in the travel `direction` (+1 right, -1
        left). False on flat ground, when `direction == 0`, or off-world."""
        world = getattr(self, "_world", None)
        if world is None or direction == 0:
            return False
        pos = self.body.position
        start = pos + pymunk.Vec2d(direction * config.GAP_PROBE_AHEAD_PX, 0.0)
        end = start + pymunk.Vec2d(0.0, config.GAP_PROBE_DEPTH_PX)  # y-down: +y is down
        hit = world.space.segment_query_first(start, end, 0.5, self._ray_filter)
        return hit is None
```

- [ ] **Step 4c: Count in update()** — in `Player.update`, after `grounded = self.grounded` (the existing line ~289), add:

```python
        if not grounded:
            self.airborne_steps += 1
        if action in _MOVE_RIGHT:
            probe_dir = 1
        elif action in _MOVE_LEFT:
            probe_dir = -1
        else:
            probe_dir = 1 if vx > 0 else (-1 if vx < 0 else 0)
```

Note: `vx` is already defined just above as `vx = self.body.velocity.x`. Place this block after that assignment.

Then in the existing `if decision.fire:` block (after the impulse is applied), add:

```python
            self.jumps_fired += 1
            if grounded and self._ledge_ahead(probe_dir):
                self.purposeful_jumps += 1
```

- [ ] **Step 5: Add the integration test** — append to `tests/test_player.py`:

```python
def test_ground_jump_over_gap_counts_purposeful():
    w = _make_world_with_gap()
    x = 200.0 - config.GAP_PROBE_AHEAD_PX + 10.0
    # Idle to settle on the floor, then jump-right at the ledge.
    p = Player(agent=_ScriptedAgent([Action.IDLE] * 20 + [Action.RIGHT_JUMP]),
               spawn_xy=(x, 560))
    w.add_entity(p)
    for _ in range(25):
        w.step(1 / 60)
    assert p.jumps_fired >= 1
    assert p.purposeful_jumps >= 1


def test_ground_jump_on_flat_not_purposeful():
    w = _make_world_with_floor()  # solid floor, no gap
    p = Player(agent=_ScriptedAgent([Action.IDLE] * 20 + [Action.RIGHT_JUMP]),
               spawn_xy=(0.0, 560))
    w.add_entity(p)
    for _ in range(25):
        w.step(1 / 60)
    assert p.jumps_fired >= 1
    assert p.purposeful_jumps == 0
```

- [ ] **Step 6: Run all player tests**

Run: `python -m pytest tests/test_player.py -q`
Expected: PASS. If the jump doesn't fire within the window, widen the IDLE settle count or confirm `RIGHT_JUMP` lands on a grounded tick (the controller fires on a fresh press while grounded).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/config.py src/blueball/entities/player.py tests/test_player.py
git commit -m "feat(ai): gap-probe purposeful-jump + bounce-metric counters on Player"
```

---

### Task 4: Fitness — efficiency term + gap bonus

**Goal:** Add `purposeful_jumps` to `FitnessInputs` and two additive terms to `fitness()`: a gap bonus (all modes) and an efficiency ratio active only when `level_width > 0`.

**Files:**
- Modify: `src/blueball/config.py` (`SPEED_W`, `JUMP_GAP_BONUS`)
- Modify: `src/blueball/ai/fitness.py` (`FitnessInputs`, `fitness`)
- Test: `tests/test_fitness.py` (new)
- Possibly update: `tests/test_climb_fitness.py`, `tests/test_ai_curriculum.py` (efficiency term changes level_width>0 values)

**Acceptance Criteria:**
- [ ] `JUMP_GAP_BONUS * purposeful_jumps` is added in all modes.
- [ ] The efficiency term `SPEED_W * progress_x / max(steps_taken, 1)` is added iff `level_width > 0`; it is exactly 0 when `level_width == 0`.
- [ ] With `level_width > 0` and equal progress, fewer steps yields strictly higher fitness.

**Verify:** `python -m pytest tests/test_fitness.py tests/test_gym_fitness.py tests/test_climb_fitness.py -q` → all pass

**Steps:**

- [ ] **Step 1: Add config constants** — in `src/blueball/config.py`, just under `GOAL_MULT`:

```python
# Efficiency reward: forward progress per step, added ONLY in goal-terminal
# modes (level_width > 0), where reaching the goal ends the episode early so
# steps_taken is informative. Penalizes bounce-spam (airtime is dead
# acceleration time; AIR_CONTROL=0) without a knife-edge per-step penalty.
# Off for goalless Infinite/gym (progress/steps is degenerate there). Tunable.
SPEED_W = 50.0
# Bonus per "purposeful" jump: a ground takeoff at a real ledge/gap. Un-farmable
# on flat ground. Tunable; set to 0.0 to rely on the efficiency term alone.
JUMP_GAP_BONUS = 5.0
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_fitness.py`:

```python
from blueball import config
from blueball.ai.fitness import FitnessInputs, fitness


def _inputs(**over):
    base = dict(
        progress_x=1000.0, collectibles=0, reached_goal=False, died=False,
        steps_taken=1000, keys_collected=0, level_width=0.0,
    )
    base.update(over)
    return FitnessInputs(**base)


def test_gap_bonus_applied_all_modes():
    no_gap = fitness(_inputs(level_width=0.0, purposeful_jumps=0))
    gap = fitness(_inputs(level_width=0.0, purposeful_jumps=3))
    assert gap - no_gap == config.JUMP_GAP_BONUS * 3


def test_efficiency_off_when_goalless():
    a = fitness(_inputs(level_width=0.0, steps_taken=500))
    b = fitness(_inputs(level_width=0.0, steps_taken=2000))
    assert a == b  # steps only enters via the tiny -0.01 term... see below


def test_efficiency_on_with_level_width():
    fast = fitness(_inputs(level_width=2000.0, steps_taken=500))
    slow = fitness(_inputs(level_width=2000.0, steps_taken=2000))
    assert fast > slow
```

NOTE: `test_efficiency_off_when_goalless` must isolate the new term from the existing `-0.01 * steps_taken`. Rewrite that test to compare the *efficiency contribution* directly:

```python
def test_efficiency_off_when_goalless():
    # Difference between two step counts at level_width=0 must equal ONLY the
    # pre-existing -0.01*steps term (no efficiency term added).
    a = fitness(_inputs(level_width=0.0, steps_taken=500))
    b = fitness(_inputs(level_width=0.0, steps_taken=1500))
    assert abs((a - b) - (-0.01 * (500 - 1500))) < 1e-6
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_fitness.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'purposeful_jumps'`.

- [ ] **Step 4a: Add the field** — in `src/blueball/ai/fitness.py`, add to `FitnessInputs` (after `climb_height`):

```python
    purposeful_jumps: int = 0  # ground takeoffs taken at a real ledge/gap
```

- [ ] **Step 4b: Add the terms** — replace the `return (...)` in `fitness()` with:

```python
def fitness(inputs: FitnessInputs) -> float:
    # Efficiency reward only in goal-terminal modes, where the episode ends on
    # goal so steps_taken distinguishes a fast finisher from a dawdler/bouncer.
    # Degenerate for goalless modes (every survivor uses all max_steps), so off.
    efficiency = 0.0
    if inputs.level_width > 0.0:
        efficiency = config.SPEED_W * inputs.progress_x / max(inputs.steps_taken, 1)
    return (
        inputs.progress_x
        + inputs.climb_height
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + config.GOAL_MULT * inputs.level_width * (1.0 if inputs.reached_goal else 0.0)
        + config.GYM_SEGMENT_BONUS * inputs.segments_cleared
        + config.JUMP_GAP_BONUS * inputs.purposeful_jumps
        + efficiency
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
    )
```

- [ ] **Step 5: Run fitness + adjacent tests; fix expected values**

Run: `python -m pytest tests/test_fitness.py tests/test_gym_fitness.py tests/test_climb_fitness.py tests/test_ai_curriculum.py -q`
Expected: `test_fitness.py` PASS. `test_gym_fitness.py` PASS (gym uses `level_width=0.0`, efficiency off; `purposeful_jumps` defaults 0). `test_climb_fitness.py` / `test_ai_curriculum.py` may FAIL on exact-value assertions because they use `level_width > 0` and now include the efficiency term. For each failing assertion, recompute the expected number to include `SPEED_W * progress_x / max(steps,1)` and update it. Do NOT weaken the assertions — update the literals to the correct new totals.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/fitness.py tests/test_fitness.py tests/test_climb_fitness.py tests/test_ai_curriculum.py
git commit -m "feat(ai): efficiency reward (goal modes) + gap-gated jump bonus"
```

---

### Task 5: Wire purposeful_jumps through every evaluator + playback

**Goal:** Pass `player.purposeful_jumps` into every `FitnessInputs` construction so the gap bonus actually scores during training and the watch HUD matches.

**Files:**
- Modify: `src/blueball/ai/trainer.py` (`_episode_fitness` at :74, `evaluate_gym` at :224)
- Modify: `src/blueball/ai/curriculum.py` (`FitnessInputs` at :212)
- Modify: `src/blueball/scenes/playback.py` (`fitness` property at :280)
- Test: `tests/test_fitness_wiring.py` (new)

**Acceptance Criteria:**
- [ ] `_episode_fitness`, `evaluate_gym`, curriculum, and playback all set `purposeful_jumps` from the player.
- [ ] A stub player with `purposeful_jumps=2` raises `_episode_fitness` by `2 * JUMP_GAP_BONUS` vs a stub with 0.

**Verify:** `python -m pytest tests/test_fitness_wiring.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — create `tests/test_fitness_wiring.py`:

```python
from types import SimpleNamespace

from blueball import config
from blueball.ai.trainer import _episode_fitness


def _stub_player(purposeful):
    return SimpleNamespace(
        collectibles_collected=0, dead=False, keys_held=0,
        purposeful_jumps=purposeful,
    )


def test_episode_fitness_includes_purposeful_jumps():
    base = _episode_fitness(_stub_player(0), spawn_x=0.0, max_x=500.0,
                            steps=500, reached_goal=False, level_width=1000.0)
    more = _episode_fitness(_stub_player(2), spawn_x=0.0, max_x=500.0,
                            steps=500, reached_goal=False, level_width=1000.0)
    assert abs((more - base) - 2 * config.JUMP_GAP_BONUS) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fitness_wiring.py -q`
Expected: FAIL — `more - base == 0` (purposeful_jumps not yet wired).

- [ ] **Step 3a: trainer `_episode_fitness`** — in `src/blueball/ai/trainer.py`, add to the `FitnessInputs(...)` at :74:

```python
        keys_collected=bin(player.keys_held).count("1"),
        level_width=float(level_width),
        purposeful_jumps=int(player.purposeful_jumps),
    ))
```

- [ ] **Step 3b: trainer `evaluate_gym`** — in the `FitnessInputs(...)` at :224, add:

```python
        segments_cleared=int(cleared),
        purposeful_jumps=int(player.purposeful_jumps),
    ))
```

- [ ] **Step 3c: curriculum** — in `src/blueball/ai/curriculum.py` `FitnessInputs(...)` at :212, add:

```python
        climb_height=float(max(0.0, spawn_y - min_y)) if climb_shaping else 0.0,
        purposeful_jumps=int(player.purposeful_jumps),
    ))
```

- [ ] **Step 3d: playback** — in `src/blueball/scenes/playback.py` `fitness` property at :280, add:

```python
            segments_cleared=int(segments_cleared),
            purposeful_jumps=int(self.player.purposeful_jumps),
        ))
```

- [ ] **Step 4: Run the wiring test + full AI suite**

Run: `python -m pytest tests/test_fitness_wiring.py tests/test_ai_curriculum.py tests/test_gym_fitness.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/trainer.py src/blueball/ai/curriculum.py src/blueball/scenes/playback.py tests/test_fitness_wiring.py
git commit -m "feat(ai): thread purposeful_jumps into all evaluators + playback HUD"
```

---

### Task 6: Bounce metrics in the watch HUD

**Goal:** Surface jumps-per-100px and airtime-% in the playback HUD so the A/B is measured, not just eyeballed, when watching the best genome.

**Files:**
- Create: `src/blueball/ai/metrics.py` (pure helper)
- Modify: `src/blueball/scenes/playback.py` (HUD/text — wherever fitness is rendered)
- Test: `tests/test_metrics.py` (new)

**Acceptance Criteria:**
- [ ] `behavior_metrics(jumps_fired, airborne_steps, steps, progress_x)` returns a dict with `jumps_per_100px` and `airtime_pct`, safe at zero denominators.
- [ ] The playback HUD shows both values for the running genome.

**Verify:** `python -m pytest tests/test_metrics.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — create `tests/test_metrics.py`:

```python
from blueball.ai.metrics import behavior_metrics


def test_basic_rates():
    m = behavior_metrics(jumps_fired=10, airborne_steps=300, steps=600,
                         progress_x=1000.0)
    assert abs(m["jumps_per_100px"] - 1.0) < 1e-6      # 10 jumps / 1000px * 100
    assert abs(m["airtime_pct"] - 50.0) < 1e-6         # 300/600


def test_zero_denominators_safe():
    m = behavior_metrics(jumps_fired=0, airborne_steps=0, steps=0,
                         progress_x=0.0)
    assert m["jumps_per_100px"] == 0.0
    assert m["airtime_pct"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — module `blueball.ai.metrics` does not exist.

- [ ] **Step 3a: Implement the helper** — create `src/blueball/ai/metrics.py`:

```python
"""Bounce/efficiency behavior metrics for A/B comparison.

Pure functions over the per-episode counters Player accumulates
(jumps_fired, airborne_steps) plus steps and progress_x. Used by the watch
HUD so a trained genome can be measured, not just eyeballed."""

from __future__ import annotations


def behavior_metrics(jumps_fired: int, airborne_steps: int, steps: int,
                     progress_x: float) -> dict[str, float]:
    jumps_per_100px = (100.0 * jumps_fired / progress_x) if progress_x > 0 else 0.0
    airtime_pct = (100.0 * airborne_steps / steps) if steps > 0 else 0.0
    return {"jumps_per_100px": jumps_per_100px, "airtime_pct": airtime_pct}
```

- [ ] **Step 3b: Show in HUD** — in `src/blueball/scenes/playback.py`, where the HUD renders the live fitness text, also render the metrics. Add near the fitness draw (use the sim's counters: `self.player.jumps_fired`, `self.player.airborne_steps`, `self.steps`, `self.max_x - self.spawn_x`):

```python
        from blueball.ai.metrics import behavior_metrics
        m = behavior_metrics(self.player.jumps_fired, self.player.airborne_steps,
                             self.steps, self.max_x - self.spawn_x)
        # render alongside the existing fitness line, e.g.:
        #   f"jumps/100px {m['jumps_per_100px']:.2f}  air% {m['airtime_pct']:.0f}"
```

Match the exact HUD text-draw pattern already used for fitness in this file (same font/`blit` call). If the counters live on the sim wrapper rather than `self.player`, read them from there to mirror how `fitness` reads `self.player`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Manual smoke (optional)** — launch a watch session on any saved genome and confirm the HUD shows the two metrics. Document the exact command in Task 7's runbook.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/ai/metrics.py src/blueball/scenes/playback.py tests/test_metrics.py
git commit -m "feat(ai): bounce metrics (jumps/100px, airtime%) in watch HUD"
```

---

### Task 7: A/B runbook + baseline capture

**Goal:** Document the exact control-vs-treatment procedure and capture baseline numbers so the reward/perception change can be judged, not assumed.

**Files:**
- Create: `docs/superpowers/runbooks/2026-06-24-purposeful-jump-ab.md`

**Acceptance Criteria:**
- [ ] Runbook lists exact commands for the maze-curriculum and gym runs (control = pre-change baseline captured before merge; treatment = this branch), with identical `--pop`, `--gens`, `--ga-seed`, `--world-seed`.
- [ ] Runbook states the success criteria from the spec (higher completion / fewer gens-to-first-goal, lower jumps-per-100px + airtime-%, bounce gone on watch) and a results table to fill in.

**Verify:** `test -f docs/superpowers/runbooks/2026-06-24-purposeful-jump-ab.md && echo OK`

**Steps:**

- [ ] **Step 1: Capture baseline (before this branch's behavior changes are exercised).** Since genome size changed, "control" = the pre-overhaul training behavior. Capture it from `master` prior to merge (or a baseline tag). Record: gens-to-first-goal and final completion on the maze curriculum and gym, plus a watched best genome's jumps-per-100px / airtime-%.

```bash
# CONTROL (run on pre-overhaul code / baseline checkout):
python main.py train maze --level maze --pop 80 --gens 200 --ga-seed 0 --world-seed 1
python main.py train gym  --pop 80 --gens 200 --max-steps 6000 --ga-seed 0 --world-seed 1 --abilities double_jump
```

- [ ] **Step 2: Run treatment (this branch).** Same flags. Record the same numbers.

- [ ] **Step 3: Write the runbook** with both command blocks, the success criteria, and a results table:

```markdown
| metric                | control | treatment |
|-----------------------|---------|-----------|
| maze gens-to-goal     |         |           |
| maze completion       |         |           |
| gym segments cleared  |         |           |
| best jumps/100px      |         |           |
| best airtime %        |         |           |
```

- [ ] **Step 4: Tuning loop.** `SPEED_W`, `JUMP_GAP_BONUS`, `GAP_PROBE_AHEAD_PX`, `GAP_PROBE_DEPTH_PX` are the knobs. Per the project's commit-cadence convention, do NOT auto-commit during tuning — adjust, re-run, and commit only on an explicit "commit". If the treatment plateaus, the NEAT escape hatch is the next step (deferred, still architecturally ready).

- [ ] **Step 5: Commit (the runbook only; on request).**

```bash
git add docs/superpowers/runbooks/2026-06-24-purposeful-jump-ab.md
git commit -m "docs(ai): A/B runbook for purposeful-jump overhaul"
```

---

## Full-suite gate

After Task 5 (and again after Task 6), run the whole suite to catch genome-size / fitness-shape fallout across modes:

```bash
python -m pytest -q
```

Expected: green. Likely touch-ups: any test asserting exact fitness with `level_width > 0` (efficiency term) or exact `INPUT_SIZE`/`GENOME_SIZE`/genome arrays (35→37, 510→534). Update literals to correct values — do not weaken assertions. Saved genomes from before this change are invalid by design (no migration).
