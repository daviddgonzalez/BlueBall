# Box-Push Shaping + Box-Lava Specialist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in box-push reward term to the curriculum fitness path and a single-stage "box-lava" specialist curriculum + CLI flag, so the GA gets a gradient up the maze's box-push maneuver — without touching `trainer.py` or any existing behavior.

**Architecture:** Four additive changes, each back-compat via defaults. (1) `FitnessInputs` gains a defaulted `box_progress` field and `fitness()` gains a `+ config.BOX_PUSH_MULT * box_progress` term. (2) `evaluate_curriculum` tracks the `PushableBox`'s max rightward displacement and feeds it into its local fitness. (3) A new `build_box_lava_curriculum(level)` returns one fixed-spawn stage just left of the box (all keys granted), and `train_curriculum` gains an optional `stages=None` override. (4) `train_maze_curriculum.py` gains a `--box-lava` flag that wires the specialist curriculum into a distinct run dir. Because `box_progress` defaults to `0.0` and `stages` defaults to `None`, the Infinite-Run path, static `train_levels`, the normal curriculum path, and all 409 current tests are byte-identical.

**Tech Stack:** Python 3.12, NumPy, pymunk (physics), pytest. GA trainer in `src/blueball/ai/`. Tests run with `.venv/bin/python -m pytest -q` from the repo root.

---

## Grounding facts (verified by reading master, 2026-06-06)

- **Baseline:** `.venv/bin/python -m pytest -q --co` → **409 tests collected**. After this plan: ~420.
- **`config`** is `src/blueball/config.py` (imported everywhere as `from .. import config`), **not** `src/blueball/ai/config.py`.
- **`fitness.py`** (`src/blueball/ai/fitness.py`): frozen `FitnessInputs(progress_x, collectibles, reached_goal, died, steps_taken, keys_collected, level_width)`; `fitness()` reads `config.GOAL_MULT` dynamically (so monkeypatching `blueball.config` attributes affects it at call time).
- **`curriculum.py`** (`src/blueball/ai/curriculum.py`): has `CurriculumStage(spawn_xy, granted_keys, label)`, module constant `SPAWN_MARGIN`, `granted_keys_before(keys, spawn_x)`, `build_spawn_curriculum(level)`, `make_curriculum_player(world, genome, spawn_xy, granted_keys)`, `evaluate_curriculum(args)` (args = `(idx, genome, world_seed, level_path, max_steps, spawn_xy, granted_keys)` → `(idx, fitness, reached_goal)`), and `train_curriculum(*, level_path, pop_size, generations, ga_seed=0, world_seed=config.DEFAULT_SEED, max_steps=config.MAX_STEPS, map_fn=map, save_dir=None)`. `evaluate_curriculum` builds fitness locally via `fitness()`/`FitnessInputs` with `keys_collected = popcount(keys_held & ~granted_keys)` and `level_width = meta.total_width`. The module name constant `_KEY_NAME = "Key"` already exists. The module already imports `World`, `register_collisions`, `load_level`.
- **`PushableBox`** (`src/blueball/entities/pushable_box.py`): has `.body` (pymunk Body, `.position.x`) and `.size` (float). Found via `type(e).__name__ == "PushableBox"` in `world.entities`.
- **Maze (`src/blueball/levels/maze.json`), loaded:** spawn `(80, 540)`, `total_width = 4224.0`; `Key` id 0 @ x=1056, id 1 @ x=2432 (both-keys mask = `0b11`); `PushableBox` @ x=3294, size 64 (left face = 3262); `Goal` @ x=4192; `Lava` x:[3328, 4096]. **Box-lava spawn x = 3294 − 32 − 12 = 3250** (left of box and left of lava). Verified frame-1 safe (player not dead after one substep). Verified: genome `random_genome(default_rng(20))` rolls right from this spawn and pushes the box (`box_progress ≈ 99.7`, `dead=False`) within 600 steps.
- **`tutorial_hill.json`** has **0** `PushableBox` entities (use it as the box-less control level); spawn `(80, 540)`.
- **`persistence.run_dir_name(..., curriculum=True)`** → key `f"{level_name}curr"`. So `level_name="mazeboxlava"` → `mazeboxlavacurr_w<seed>_<ts>`. No persistence change needed.
- **`train_maze_curriculum.py`** already imports `build_spawn_curriculum, evaluate_curriculum, train_curriculum` from `blueball.ai.curriculum`, and `run_dir_name, GENOMES_ROOT` from `blueball.ai.persistence`. Its verdict re-evaluates `result.best_genome` from `stages[-1]`.

## File Structure

- **Modify** `src/blueball/config.py` — add one tunable constant `BOX_PUSH_MULT = 1.0` (Task 1).
- **Modify** `src/blueball/ai/fitness.py` — add defaulted `box_progress` field + one fitness term (Task 1).
- **Modify** `src/blueball/ai/curriculum.py` — box tracking in `evaluate_curriculum` (Task 2); new `BOX_LAVA_SPAWN_MARGIN` + `build_box_lava_curriculum` (Task 3); `stages=None` param on `train_curriculum` (Task 4).
- **Modify** `train_maze_curriculum.py` — `--box-lava` flag (Task 5).
- **Modify** `tests/test_ai_smoke.py` — fitness box-term tests (Task 1).
- **Modify** `tests/test_ai_curriculum.py` — box tracking, box-lava curriculum, custom-stages, CLI tests (Tasks 2–5).

`trainer.py` is **NOT** touched. No new source files are created.

---

### Task 1: Box-push fitness term (`config.py` + `fitness.py`)

**Goal:** `FitnessInputs` gains a defaulted `box_progress` field and `fitness()` adds `config.BOX_PUSH_MULT * box_progress`, with `BOX_PUSH_MULT = 1.0` in config; every existing caller (which omits the field) is unchanged.

**Files:**
- Modify: `src/blueball/config.py` (AI / GA training section, near `GOAL_MULT`)
- Modify: `src/blueball/ai/fitness.py:14-33`
- Test: `tests/test_ai_smoke.py` (append to the "Task 3: Fitness" section)

**Acceptance Criteria:**
- [ ] `FitnessInputs(... )` with no `box_progress` still constructs; the field defaults to `0.0`.
- [ ] A positive `box_progress` increases `fitness()` by exactly `config.BOX_PUSH_MULT * box_progress`.
- [ ] `config.BOX_PUSH_MULT == 1.0`.
- [ ] All pre-existing fitness tests stay green (they omit `box_progress`).

**Verify:** `.venv/bin/python -m pytest tests/test_ai_smoke.py -q` → all pass (existing + 2 new).

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_smoke.py`, after `test_fitness_no_goal_is_independent_of_width` (end of the "Task 3: Fitness" block, ~line 364):

```python
def test_fitness_box_progress_adds_box_push_term():
    """A positive box_progress adds exactly config.BOX_PUSH_MULT * box_progress
    on top of otherwise-identical inputs."""
    from blueball import config
    from blueball.ai.fitness import fitness, FitnessInputs
    base = dict(progress_x=100.0, collectibles=0, reached_goal=False,
                died=False, steps_taken=0, keys_collected=0, level_width=0.0)
    f_no_box = fitness(FitnessInputs(**base))                 # box_progress defaults 0.0
    f_box = fitness(FitnessInputs(box_progress=250.0, **base))
    assert f_box - f_no_box == pytest.approx(config.BOX_PUSH_MULT * 250.0)
    assert config.BOX_PUSH_MULT == 1.0


def test_fitness_box_progress_defaults_to_zero():
    """Omitting box_progress (every existing caller) leaves fitness unchanged:
    the field defaults to 0.0 so the new term vanishes."""
    from blueball.ai.fitness import fitness, FitnessInputs
    kw = dict(progress_x=300.0, collectibles=1, reached_goal=False, died=False,
              steps_taken=10, keys_collected=1, level_width=0.0)
    assert fitness(FitnessInputs(box_progress=0.0, **kw)) == fitness(FitnessInputs(**kw))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai_smoke.py::test_fitness_box_progress_adds_box_push_term -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'box_progress'` (and `AttributeError`/no `BOX_PUSH_MULT`).

- [ ] **Step 3: Add `BOX_PUSH_MULT` to `config.py`** — insert immediately after the `GOAL_MULT` block (after line 121, `GOAL_MULT              = 2.0`):

```python
# Box-push shaping: fitness reward per pixel of net rightward PushableBox
# displacement. Applied ONLY in the curriculum evaluator (the box-lava
# specialist) — callers that don't pass box_progress are unaffected. 0.0
# reduces to progress-only; 1.0 is the starting guess (selection is comparative,
# so the exact value isn't load-bearing).
BOX_PUSH_MULT          = 1.0
```

- [ ] **Step 4: Add the defaulted field to `FitnessInputs`** — `src/blueball/ai/fitness.py`, append the field at the end of the dataclass (after `level_width: float`):

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float    # furthest x reached - spawn_x
    collectibles: int    # player.collectibles_collected
    reached_goal: bool   # player.reached_goal
    died: bool           # player.dead
    steps_taken: int     # the loop counter from the evaluator
    keys_collected: int  # popcount of player.keys_held
    level_width: float   # level total width; 0.0 for goalless (infinite) modes
    box_progress: float = 0.0  # net rightward PushableBox displacement (px); 0 when no box
```

- [ ] **Step 5: Add the term to `fitness()`** — `src/blueball/ai/fitness.py`, add the final line inside the return:

```python
def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + config.GOAL_MULT * inputs.level_width * (1.0 if inputs.reached_goal else 0.0)
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
        + config.BOX_PUSH_MULT * inputs.box_progress
    )
```

- [ ] **Step 6: Run the fitness tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_smoke.py -q`
Expected: PASS (all existing fitness tests + the 2 new ones).

- [ ] **Step 7: Run the full suite (back-compat guard)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, count = 411 (409 + 2).

- [ ] **Step 8: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/fitness.py tests/test_ai_smoke.py
git commit -m "feat(fitness): add defaulted box_progress term (BOX_PUSH_MULT)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Box tracking in `evaluate_curriculum` (`curriculum.py`)

**Goal:** `evaluate_curriculum` finds the `PushableBox`, tracks its max rightward displacement over the substep loop, and passes `box_progress` into its local `FitnessInputs`. Box-less levels are unaffected (`box_progress = 0.0`).

**Files:**
- Modify: `src/blueball/ai/curriculum.py:144-173` (the `evaluate_curriculum` body, after `make_curriculum_player` through the `fitness(...)` call)
- Test: `tests/test_ai_curriculum.py` (append after `test_evaluate_curriculum_granted_keys_dont_inflate_fitness`)

**Acceptance Criteria:**
- [ ] On maze, with a box-pushing genome spawned just left of the box, `evaluate_curriculum` scored with `BOX_PUSH_MULT=1.0` returns strictly higher fitness than the identical run scored with `BOX_PUSH_MULT=0.0` (the box moved right → positive box term).
- [ ] On a box-less level (`tutorial_hill`), `evaluate_curriculum` fitness is identical regardless of `BOX_PUSH_MULT` (`box_progress` is `0.0`).
- [ ] Determinism and the `(idx, fitness, reached_goal)` return shape are unchanged.

**Verify:** `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_curriculum.py`:

```python
def test_evaluate_curriculum_tracks_box_progress(monkeypatch):
    """On maze, spawning just left of the PushableBox and pushing it right makes
    box-push shaping raise fitness: evaluate_curriculum with BOX_PUSH_MULT=1.0
    exceeds the identical run scored with 0.0 by the (positive) box term."""
    import blueball.config as bbconfig
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    # Just left of the maze box (x=3294, size 64 -> left face 3262); both maze
    # keys (ids 0,1) granted. Genome 20 rolls right and shoves the box.
    spawn_xy = (3250.0, 540.0)
    granted = (1 << 0) | (1 << 1)
    g = random_genome(np.random.default_rng(20))
    args = (0, g, 1, path, 600, spawn_xy, granted)

    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 0.0)
    _, fit_off, reached_off = evaluate_curriculum(args)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 1.0)
    _, fit_on, reached_on = evaluate_curriculum(args)

    assert fit_on > fit_off          # box moved right -> positive box term
    assert reached_off == reached_on # box term doesn't change the goal verdict


def test_evaluate_curriculum_no_box_unaffected_by_box_mult(monkeypatch):
    """On a box-less level, box_progress is 0.0 so BOX_PUSH_MULT has no effect:
    fitness is identical whether shaping is on or off."""
    import blueball.config as bbconfig
    from pathlib import Path
    import blueball
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.genome import random_genome
    level = Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"
    g = random_genome(np.random.default_rng(0))
    args = (0, g, 1, level, 200, (80.0, 540.0), 0)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 0.0)
    _, fit_off, _ = evaluate_curriculum(args)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 5.0)
    _, fit_on, _ = evaluate_curriculum(args)
    assert fit_off == fit_on
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_evaluate_curriculum_tracks_box_progress -q`
Expected: FAIL — `fit_on == fit_off` (box term not yet wired; assertion `fit_on > fit_off` fails).

- [ ] **Step 3: Add box tracking to `evaluate_curriculum`** — replace the body from the `make_curriculum_player(...)` line through the `fitness(...)` call (`src/blueball/ai/curriculum.py:146-172`) with:

```python
    player = make_curriculum_player(world, genome, spawn_xy, granted_keys)

    # Track the PushableBox's rightward displacement (the box-push reward
    # gradient). Levels with no box leave box_progress at 0.0 -> no behavior
    # change. Mirrors the player's max_x high-water mark (robust to knockback).
    box = next((e for e in world.entities
                if type(e).__name__ == "PushableBox"), None)
    box_start_x = float(box.body.position.x) if box is not None else None
    box_max_x = box_start_x

    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator residual,
        # so long headless runs are bit-identical across machines (see trainer).
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if box is not None and box.body.position.x > box_max_x:
            box_max_x = box.body.position.x
        if player.dead or player.reached_goal:
            break

    box_progress = max(0.0, box_max_x - box_start_x) if box is not None else 0.0

    # Granted keys are training scaffolding, not achievements: count only the
    # keys actually collected this episode (bits set that were NOT granted), so
    # the curriculum can't hand out free fitness.
    collected = bin(player.keys_held & ~int(granted_keys)).count("1")
    f = fitness(FitnessInputs(
        progress_x=float(max_x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(player.reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
        keys_collected=collected,
        level_width=float(meta.total_width),
        box_progress=float(box_progress),
    ))
    return idx, float(f), bool(player.reached_goal)
```

- [ ] **Step 4: Run the curriculum tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (existing curriculum tests + the 2 new ones).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, count = 413 (411 + 2).

- [ ] **Step 6: Commit**

```bash
git add src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(curriculum): track PushableBox displacement as box_progress

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `build_box_lava_curriculum` + `BOX_LAVA_SPAWN_MARGIN` (`curriculum.py`)

**Goal:** A new `build_box_lava_curriculum(level)` returns exactly one `CurriculumStage` labelled `"box_lava"`, spawning the agent `BOX_LAVA_SPAWN_MARGIN` px left of the box's left face with every key granted. Raises `ValueError` on a box-less level.

**Files:**
- Modify: `src/blueball/ai/curriculum.py` (add the constant near `SPAWN_MARGIN` ~line 37; add the function after `build_spawn_curriculum`, ~line 117)
- Test: `tests/test_ai_curriculum.py` (append three tests)

**Acceptance Criteria:**
- [ ] `build_box_lava_curriculum(maze)` returns a list of length 1; its single stage has `label == "box_lava"`.
- [ ] The stage's spawn x equals `box_x - box.size/2 - BOX_LAVA_SPAWN_MARGIN` and is `< box_x`; spawn y equals the level's true spawn y.
- [ ] `granted_keys` equals both maze keys (== `granted_keys_before(keys, spawn_x)`).
- [ ] The spawn is frame-1 safe (player not dead after one substep).
- [ ] On a box-less level it raises `ValueError` mentioning `PushableBox`.

**Verify:** `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_curriculum.py`:

```python
def test_build_box_lava_curriculum_single_stage_left_of_box():
    from blueball.ai.curriculum import (build_box_lava_curriculum,
                                        BOX_LAVA_SPAWN_MARGIN, granted_keys_before)
    path, world, meta = _maze_world()
    keys = _maze_keys(world)
    box = next(e for e in world.entities if type(e).__name__ == "PushableBox")
    box_x = float(box.body.position.x)
    all_bits = 0
    for kid, _ in keys:
        all_bits |= (1 << kid)

    stages = build_box_lava_curriculum(path)
    assert len(stages) == 1
    s = stages[0]
    assert s.label == "box_lava"
    assert s.spawn_xy[0] == pytest.approx(box_x - box.size / 2.0 - BOX_LAVA_SPAWN_MARGIN)
    assert s.spawn_xy[0] < box_x
    assert s.spawn_xy[1] == pytest.approx(float(meta.spawn[1]))
    assert s.granted_keys == all_bits
    assert s.granted_keys == granted_keys_before(keys, s.spawn_xy[0])


def test_build_box_lava_curriculum_spawn_is_frame1_safe():
    """The box-lava spawn lands on the approach ledge, not in lava/geometry."""
    from blueball.ai.curriculum import (build_box_lava_curriculum,
                                        make_curriculum_player)
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    stage = build_box_lava_curriculum(path)[0]
    world = World(seed=1)
    register_collisions(world.space, world_ref=world)
    load_level(path, world)
    p = make_curriculum_player(world, g, stage.spawn_xy, stage.granted_keys)
    world.substep()
    assert not p.dead


def test_build_box_lava_curriculum_requires_box():
    from pathlib import Path
    import blueball
    from blueball.ai.curriculum import build_box_lava_curriculum
    level = Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"
    with pytest.raises(ValueError, match="PushableBox"):
        build_box_lava_curriculum(level)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_build_box_lava_curriculum_single_stage_left_of_box -q`
Expected: FAIL — `ImportError: cannot import name 'build_box_lava_curriculum'`.

- [ ] **Step 3: Add the constant** — `src/blueball/ai/curriculum.py`, right after the `SPAWN_MARGIN = 96.0` definition (~line 37):

```python
# How far left of the box's left face (px) the box-lava specialist spawns, so
# rolling right immediately contacts and pushes the box. Module constant for
# easy iteration; selection is comparative so the exact value isn't load-bearing.
BOX_LAVA_SPAWN_MARGIN = 12.0
```

- [ ] **Step 4: Add the function** — `src/blueball/ai/curriculum.py`, after `build_spawn_curriculum` returns (after line 117) and before `make_curriculum_player`:

```python
def build_box_lava_curriculum(level: Union[str, Path, dict]) -> list[CurriculumStage]:
    """Single-stage curriculum for the box-lava section: spawn just left of the
    PushableBox (so rolling right pushes it into the lava as a stepping stone)
    with every key granted (both gates are behind this spawn). Used to train a
    box-lava specialist.

    Returns exactly one CurriculumStage labelled "box_lava". Raises ValueError
    if the level has no PushableBox.
    """
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(level, world)

    box = next((e for e in world.entities
                if type(e).__name__ == "PushableBox"), None)
    if box is None:
        raise ValueError("build_box_lava_curriculum: level has no PushableBox")

    keys: list[tuple[int, float]] = [
        (int(e.key_id), float(e.position[0]))
        for e in world.entities if type(e).__name__ == _KEY_NAME
    ]

    box_x = float(box.body.position.x)
    spawn_x = box_x - box.size / 2.0 - BOX_LAVA_SPAWN_MARGIN
    spawn_y = float(meta.spawn[1])
    return [CurriculumStage(
        spawn_xy=(spawn_x, spawn_y),
        granted_keys=granted_keys_before(keys, spawn_x),
        label="box_lava",
    )]
```

- [ ] **Step 5: Run the curriculum tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (existing + the 3 new).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, count = 416 (413 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(curriculum): build_box_lava_curriculum single-stage specialist

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `stages=None` override on `train_curriculum` (`curriculum.py`)

**Goal:** `train_curriculum` gains an optional `stages` parameter (default `None`). When `None`, it builds `build_spawn_curriculum(level_path)` exactly as today; when given a stages list (e.g. from `build_box_lava_curriculum`), it trains at that fixed curriculum and never recedes past the last stage.

**Files:**
- Modify: `src/blueball/ai/curriculum.py:176-205` (the `train_curriculum` signature + the `stages = build_spawn_curriculum(level_path)` line)
- Test: `tests/test_ai_curriculum.py` (append three tests)

**Acceptance Criteria:**
- [ ] `train_curriculum(..., stages=build_box_lava_curriculum(maze))` records `curriculum.stages == ["box_lava"]` in `run.json` and every history entry stays on stage 0.
- [ ] With custom stages it is deterministic run-to-run (identical `best_genome`).
- [ ] `train_curriculum(...)` with default `stages=None` is unchanged: `run.json` `curriculum.stages[-1] == "start"`.

**Verify:** `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_curriculum.py`:

```python
def test_train_curriculum_custom_stages_used(tmp_path):
    """Passing stages=build_box_lava_curriculum(maze) trains at that single fixed
    stage: run.json records exactly ['box_lava'] and the loop never recedes."""
    import json
    from blueball.ai.curriculum import train_curriculum, build_box_lava_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    stages = build_box_lava_curriculum(path)
    run_dir = tmp_path / "boxlava_run"
    train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                     world_seed=1, max_steps=60, save_dir=run_dir, stages=stages)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["curriculum"]["stages"] == ["box_lava"]
    assert all(h["stage"] == 0 for h in meta["history"])


def test_train_curriculum_custom_stages_deterministic():
    from blueball.ai.curriculum import train_curriculum, build_box_lava_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    stages = build_box_lava_curriculum(path)
    a = train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                         world_seed=1, max_steps=60, stages=stages)
    b = train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                         world_seed=1, max_steps=60, stages=stages)
    assert np.array_equal(a.best_genome, b.best_genome)


def test_train_curriculum_default_stages_unchanged(tmp_path):
    """stages=None (default) still builds the full reverse curriculum:
    run.json's last stage is 'start' as before."""
    import json
    from blueball.ai.curriculum import train_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    run_dir = tmp_path / "default_run"
    train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                     world_seed=1, max_steps=60, save_dir=run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["curriculum"]["stages"][-1] == "start"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_train_curriculum_custom_stages_used -q`
Expected: FAIL — `TypeError: train_curriculum() got an unexpected keyword argument 'stages'`.

- [ ] **Step 3: Add the `stages` parameter** — `src/blueball/ai/curriculum.py`, in the `train_curriculum` signature add the keyword-only parameter after `save_dir` (~line 185):

```python
def train_curriculum(
    *,
    level_path: Union[str, Path],
    pop_size: int,
    generations: int,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable = map,
    save_dir: Union[Path, str, None] = None,
    stages: Union[list[CurriculumStage], None] = None,
) -> "TrainingResult":
```

- [ ] **Step 4: Use the override** — in the same function, replace the existing line `stages = build_spawn_curriculum(level_path)` (~line 205) with:

```python
    # Default None preserves today's behavior exactly; a custom one-element list
    # (e.g. build_box_lava_curriculum) trains a fixed-spawn specialist that never
    # recedes (it's already the last stage).
    stages = stages if stages is not None else build_spawn_curriculum(level_path)
```

- [ ] **Step 5: Run the curriculum tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (existing + the 3 new).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, count = 419 (416 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(curriculum): optional stages override on train_curriculum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `--box-lava` CLI flag (`train_maze_curriculum.py`)

**Goal:** `train_maze_curriculum.py --box-lava` trains the box-lava specialist: it builds `build_box_lava_curriculum(level)`, passes it as `train_curriculum(stages=…)`, writes to a `mazeboxlavacurr_w<seed>_<ts>` run dir, and the verdict re-evaluates the final genome from the box-lava spawn. Without the flag, the CLI is unchanged.

**Files:**
- Modify: `train_maze_curriculum.py` (import, argparse flag, stage selection, `train_curriculum(stages=…)`, run-dir name)
- Test: `tests/test_ai_curriculum.py` (append one subprocess test)

**Acceptance Criteria:**
- [ ] `train_maze_curriculum.py --box-lava --pop 4 --gens 2 --max-steps 60 --workers 1` exits 0.
- [ ] It writes exactly one `mazeboxlavacurr_w1_*` run dir containing `final_best.npy` + `run.json` whose `curriculum.stages == ["box_lava"]`.
- [ ] stdout contains `reached_goal=` (the verdict line).
- [ ] Without `--box-lava`, the existing CLI test (`mazecurr_w1_*`, stages end `"start"`) still passes.

**Verify:** `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing test** — append to `tests/test_ai_curriculum.py`:

```python
def test_train_maze_curriculum_cli_box_lava_writes_run(tmp_path):
    import json, os, subprocess, sys
    import blueball
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_maze_curriculum.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "--box-lava", "--pop", "4",
         "--gens", "2", "--max-steps", "60", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("mazeboxlavacurr_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert meta["curriculum"]["stages"] == ["box_lava"]
    assert "reached_goal" in r.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_train_maze_curriculum_cli_box_lava_writes_run -q`
Expected: FAIL — non-zero exit (`unrecognized arguments: --box-lava`).

- [ ] **Step 3: Add the import** — `train_maze_curriculum.py`, extend the curriculum import (lines 22-23):

```python
from blueball.ai.curriculum import (build_box_lava_curriculum,
                                    build_spawn_curriculum, evaluate_curriculum,
                                    train_curriculum)
```

- [ ] **Step 4: Add the argparse flag** — after the `--workers` argument (~line 36), add:

```python
    parser.add_argument("--box-lava", action="store_true",
                        help="train a box-lava specialist: single fixed stage "
                             "spawned just left of the PushableBox, all keys "
                             "granted (writes a mazeboxlavacurr_* run dir).")
```

- [ ] **Step 5: Select stages + run-dir name** — replace the run-dir + stages block (current lines 44-57, from `timestamp = ...` through the `print(...)` call) with:

```python
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.box_lava:
        stages = build_box_lava_curriculum(level_path)
        level_name = "mazeboxlava"
    else:
        stages = build_spawn_curriculum(level_path)
        level_name = args.level

    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        world_seed=args.world_seed, timestamp=timestamp,
        level_name=level_name, curriculum=True,
    )
    print(
        f"Curriculum training {args.pop}x{args.gens} on {args.level} "
        f"({len(stages)} stages) world={args.world_seed} ga={args.ga_seed}\n"
        f"  -> {run_dir}"
    )
```

- [ ] **Step 6: Pass the stages into `train_curriculum`** — in the `train_curriculum(...)` call (~lines 61-70), add the `stages=stages` argument:

```python
        result = train_curriculum(
            level_path=level_path,
            pop_size=args.pop,
            generations=args.gens,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
            stages=stages,
        )
```

(The verdict block below already uses `stages[-1]` — for `--box-lava` that is the single `box_lava` stage; for the default path it remains the true-start stage. No change needed there.)

- [ ] **Step 7: Run the curriculum tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS — both the new `--box-lava` test and the pre-existing `test_train_maze_curriculum_cli_writes_run` (default path, `mazecurr_w1_*`, stages end `"start"`).

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, count = 420 (419 + 1).

- [ ] **Step 9: Commit**

```bash
git add train_maze_curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(cli): --box-lava flag trains the box-lava specialist

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (`docs/superpowers/specs/2026-06-06-box-push-shaping-design.md`):
- §1 box-push shaping (`fitness.py` + `config.py`, `BOX_PUSH_MULT=1.0`) → **Task 1**.
- §2 box tracking in `evaluate_curriculum` (max-x high-water mark, `box_progress = max(0, box_max_x - box_start_x)`, box-less → 0.0) → **Task 2**.
- §3 `build_box_lava_curriculum` + `BOX_LAVA_SPAWN_MARGIN=12.0` + `train_curriculum(stages=None)` → **Tasks 3 & 4**.
- §4 CLI `--box-lava` (run dir `mazeboxlavacurr_*`, verdict from box-lava spawn) → **Task 5**.
- §5 determinism & back-compat (defaults preserve all existing paths; `trainer.py` untouched) → guarded by the full-suite step at the end of every task and the explicit default-path tests in Tasks 2 & 4.
- Testing section items map 1:1 onto the tests in Tasks 1–5.

**Placeholder scan:** none — every step shows the exact code/command.

**Type consistency:** `box_progress` (float) is the single name used across Tasks 1–2; `BOX_LAVA_SPAWN_MARGIN` (Task 3) and `stages` param (Task 4) are consumed unchanged by Task 5; `build_box_lava_curriculum` returns `list[CurriculumStage]` and is used identically in Tasks 4–5. `run_dir_name(level_name="mazeboxlava", curriculum=True)` → `mazeboxlavacurr` key, asserted in Task 5's test.
</content>
</invoke>
