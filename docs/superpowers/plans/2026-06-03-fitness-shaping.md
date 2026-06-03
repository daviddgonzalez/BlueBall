# Fitness Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the GA fitness function so the agent optimizes furthest progress (matching the in-game Infinite Run score), earns intermediate credit for keys (so it can learn gated levels), and treats death as a modest tiebreaker.

**Architecture:** One change to `fitness()` (add a keys term, bump the death penalty) plus a small `_episode_fitness()` helper in the trainer that both evaluators call — it tracks furthest-x as the progress signal and counts keys from the player's bitfield. The helper deduplicates the `FitnessInputs` construction currently copied in both evaluators.

**Tech Stack:** Python, numpy, pymunk, pytest (headless via `SDL_VIDEODRIVER=dummy`).

**Spec:** `docs/superpowers/specs/2026-06-03-fitness-shaping-design.md`

**Conventions:** Run tests from the repo root with the venv: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest …`. Work on branch `feature/fitness-shaping`.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/blueball/ai/fitness.py` | fitness formula + inputs dataclass | Add `keys_collected` field + `100·keys` term; death `−100`→`−200` |
| `src/blueball/ai/trainer.py` | headless evaluators | Add `_episode_fitness()` helper; both evaluators track furthest-x and use it |
| `tests/test_ai_smoke.py` | AI tests | Update 3 fitness tests; add keys test + `_episode_fitness` test |

---

### Task 1: Reshape fitness — furthest-x, keys, modest death

**Goal:** Update `fitness()` to add a keys term and a `−200` death penalty, and route both evaluators through a shared `_episode_fitness()` helper that uses furthest-x as progress and counts keys.

**Files:**
- Modify: `src/blueball/ai/fitness.py`
- Modify: `src/blueball/ai/trainer.py` (`evaluate` ~lines 61-91, `evaluate_infinite` ~lines 94-130)
- Test: `tests/test_ai_smoke.py` (fitness tests ~lines 303-332)

**Acceptance Criteria:**
- [ ] `FitnessInputs` has a `keys_collected: int` field (required, last).
- [ ] `fitness()` adds `100 · keys_collected` and uses `−200 · died` (was `−100`).
- [ ] `trainer._episode_fitness(player, spawn_x, max_x, steps, reached_goal)` builds `FitnessInputs` with `progress_x = max_x − spawn_x` and `keys_collected = bin(player.keys_held).count("1")`.
- [ ] `evaluate` and `evaluate_infinite` track the furthest x reached and call `_episode_fitness` (no inline `FitnessInputs` construction left).
- [ ] Fitness formula tests updated; new keys + `_episode_fitness` tests pass.
- [ ] Full suite green.

**Verify:** `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/ -q` → all pass

**Steps:**

- [ ] **Step 1: Update the fitness tests (RED)**

In `tests/test_ai_smoke.py`, replace the three existing fitness tests (`test_fitness_all_zero_returns_zero`, `test_fitness_shape_matches_spec_formula`, `test_fitness_penalizes_death_and_charges_step_cost`) with:

```python
def test_fitness_all_zero_returns_zero():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=0.0, collectibles=0, reached_goal=False,
        died=False, steps_taken=0, keys_collected=0,
    ))
    assert f == 0.0


def test_fitness_shape_matches_spec_formula():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=500.0, collectibles=3, reached_goal=True,
        died=False, steps_taken=1000, keys_collected=0,
    ))
    # 500 + 50*3 + 200 - 0.01*1000 - 0 + 100*0 = 500 + 150 + 200 - 10 = 840
    assert f == pytest.approx(840.0)


def test_fitness_penalizes_death_and_charges_step_cost():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=10.0, collectibles=0, reached_goal=False,
        died=True, steps_taken=500, keys_collected=0,
    ))
    # 10 + 0 + 0 - 5 - 200 + 0 = -195
    assert f == pytest.approx(-195.0)


def test_fitness_rewards_keys():
    """Each key collected adds exactly 100."""
    from blueball.ai.fitness import fitness, FitnessInputs
    base = dict(progress_x=0.0, collectibles=0, reached_goal=False,
                died=False, steps_taken=0)
    f0 = fitness(FitnessInputs(keys_collected=0, **base))
    f2 = fitness(FitnessInputs(keys_collected=2, **base))
    assert f0 == 0.0
    assert f2 == pytest.approx(200.0)
```

Also add this evaluator-helper test (near the trainer tests, after `test_evaluate_runs_one_genome_to_completion`):

```python
def test_episode_fitness_uses_furthest_x_and_counts_keys():
    """_episode_fitness scores progress on the furthest x reached (not final)
    and credits each held key (popcount of the bitfield)."""
    from blueball.ai.trainer import _episode_fitness

    class _StubPlayer:
        def __init__(self, keys_held, dead=False, collectibles=0):
            self.keys_held = keys_held
            self.dead = dead
            self.collectibles_collected = collectibles

    # keys 0 and 2 set -> 2 keys -> +200; progress = max_x(300) - spawn(80) = 220
    player = _StubPlayer(keys_held=(1 << 0) | (1 << 2))
    f = _episode_fitness(player, spawn_x=80.0, max_x=300.0, steps=0, reached_goal=False)
    # 220 + 100*2 = 420
    assert f == pytest.approx(420.0)
```

- [ ] **Step 2: Run tests to confirm RED**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k "fitness or episode_fitness" -q`
Expected: FAIL — `FitnessInputs` has no `keys_collected` (TypeError) and `_episode_fitness` doesn't exist (ImportError).

- [ ] **Step 3: Update `fitness.py`**

Replace the dataclass and `fitness()` in `src/blueball/ai/fitness.py` with:

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float    # furthest x reached - spawn_x
    collectibles: int    # player.collectibles_collected
    reached_goal: bool   # player.reached_goal
    died: bool           # player.dead
    steps_taken: int     # the loop counter from the evaluator
    keys_collected: int  # popcount of player.keys_held


def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + 200.0 * (1.0 if inputs.reached_goal else 0.0)
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
    )
```

(Update the module docstring's "v1 spec's starting shape" line to note progress is now furthest-x and keys are rewarded, if you like — optional.)

- [ ] **Step 4: Add `_episode_fitness` + furthest-x tracking in `trainer.py`**

In `src/blueball/ai/trainer.py`, add this helper just above `evaluate` (after the `INFINITE_SPAWN`/`_INFINITE_SPAWN` constant and `TrainingResult`):

```python
def _episode_fitness(player, spawn_x, max_x, steps, reached_goal):
    """Build the per-episode fitness from an evaluated player. Shared by both
    evaluators. `max_x` is the furthest x the player reached (>= spawn_x), so
    progress is robust to knockback / falling back before death."""
    return fitness(FitnessInputs(
        progress_x=float(max_x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
        keys_collected=bin(player.keys_held).count("1"),
    ))
```

In `evaluate`, replace the step loop and fitness construction (everything from `steps = 0` through the `return idx, float(f)`) with:

```python
    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator
        # residual, so long headless runs are bit-identical across machines.
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.dead or player.reached_goal:
            break

    f = _episode_fitness(player, spawn_x, max_x, steps,
                         reached_goal=bool(player.reached_goal))
    return idx, float(f)
```

In `evaluate_infinite`, replace the step loop and fitness construction similarly:

```python
    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Extend/cull terrain ahead of the ball, then advance one substep.
        terrain.maintain(player.body.position.x)
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.dead:
            break

    f = _episode_fitness(player, spawn_x, max_x, steps, reached_goal=False)
    return idx, float(f)
```

(In `evaluate`, `spawn_x` is `float(meta.spawn[0])`; in `evaluate_infinite` it's `INFINITE_SPAWN[0]` — both already bound as `spawn_x` in the existing code. Confirm `FitnessInputs`/`fitness` are still imported in trainer.py — they are, via `from .fitness import FitnessInputs, fitness`.)

- [ ] **Step 5: Run tests to verify GREEN**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k "fitness or episode_fitness" -q`
Expected: PASS.

- [ ] **Step 6: Full suite**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/ -q`
Expected: all green (the determinism / Pool-equality tests assert finiteness + run-to-run equality, not absolute fitness values, so they still hold).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/fitness.py src/blueball/ai/trainer.py tests/test_ai_smoke.py
git commit -m "feat(ai): fitness rewards furthest-x + keys; modest death penalty

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- Stay on branch `feature/fitness-shaping`.
- Don't change the GA operators, network, trainer control flow, or TrainScene — only the fitness formula, the two evaluators' progress/keys wiring, and the tests.
- `progress_x`'s *meaning* changes (furthest, not final) but the field name stays — don't rename it.
