# Start-First (Forward) Spawn Curriculum — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give start-gated levels a forward spawn-curriculum (spawn fixed at the true start, finish-line checkpoint advances goal-ward) so the maze trains on its deadly opening every generation instead of never.

**Architecture:** Mirror the existing reverse curriculum inside the same trio — `CurriculumStage`, `evaluate_curriculum`, `build_spawn_curriculum`. A `CurriculumStage.checkpoint_x` field makes the evaluator terminate early at a moving finish line; a level-declared `start_gated` flag makes `build_spawn_curriculum` emit forward stages (all spawning at the true start). No new training command or CLI path.

**Tech Stack:** Python, numpy, pymunk (headless physics), pytest. Spec: `docs/superpowers/specs/2026-06-27-start-first-curriculum-design.md`.

**Maze geometry (verified, drives the checkpoint values):** spawn `x=80`; deadly `spike_wall` spans `x=[96..224]`; keys at `x=1056, 2432`; goal at `x=4192`; total width `4224`.

**Ordering rationale:** Task 1 changes the evaluator's args-tuple arity (touches all callers/tests) but leaves the maze on the reverse curriculum, so the suite stays green. Task 2 adds the forward build path, tested via in-memory level dicts, with the maze still reverse. Task 3 flips `maze.json` to forward and migrates the maze-reverse tests onto a flag-stripped dict so the reverse code path keeps its only regression guard.

---

### Task 1: Add `checkpoint_x` to `CurriculumStage` + checkpoint-aware `evaluate_curriculum`

**Goal:** The evaluator can terminate an episode early when the ball crosses a moving finish line, reporting that crossing as the stage-advancement signal; with no checkpoint it behaves exactly as today.

**Files:**
- Modify: `src/blueball/ai/curriculum.py` (`CurriculumStage`, `evaluate_curriculum`, `train_curriculum` args tuple)
- Modify: `src/blueball/cli.py` (`cmd_train_maze` verdict call)
- Test: `tests/test_ai_curriculum.py` (update 3 tuple call sites; add 2 new tests)
- Test: `tests/test_climb_fitness.py` (update 2 tuple call sites)

**Acceptance Criteria:**
- [ ] `CurriculumStage` has `checkpoint_x: float | None = None`; existing constructions (which omit it) compile and default to `None`.
- [ ] `evaluate_curriculum` accepts an 8-element args tuple ending in `checkpoint_x`; when set, the episode breaks the first frame `position.x >= checkpoint_x` and returns `reached=True`.
- [ ] When `checkpoint_x is None`, `evaluate_curriculum` is behaviorally unchanged (the `reached` value reflects real goal-reaching only).
- [ ] All existing direct `evaluate_curriculum(...)` tuple callers pass the new 8th element.
- [ ] Full suite green.

**Verify:** `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py tests/test_climb_fitness.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** in `tests/test_ai_curriculum.py` (append near the other `evaluate_curriculum` tests):

```python
def test_evaluate_curriculum_checkpoint_crossing_reports_reached():
    """A checkpoint at/behind the spawn is crossed on frame 1 -> the episode
    terminates early and reports reached=True (the advancement signal)."""
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    spawn = (80.0, 540.0)
    # checkpoint just behind spawn => crossed immediately, deterministically
    _, fit, reached = evaluate_curriculum((0, g, 1, path, 500, spawn, 0, spawn[0] - 1.0))
    assert reached is True
    assert isinstance(fit, float) and np.isfinite(fit)


def test_evaluate_curriculum_none_checkpoint_unchanged():
    """checkpoint_x=None: behaves as before — a random genome from the maze start
    cannot reach the far goal in 120 steps, so reached is False."""
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    _, _, reached = evaluate_curriculum((0, g, 1, path, 120, (80.0, 540.0), 0, None))
    assert reached is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py::test_evaluate_curriculum_checkpoint_crossing_reports_reached -q`
Expected: FAIL — `evaluate_curriculum` unpacks a 7-tuple, so an 8-tuple raises `ValueError: too many values to unpack`.

- [ ] **Step 3: Add the `checkpoint_x` field** to `CurriculumStage` in `src/blueball/ai/curriculum.py`:

```python
@dataclass(frozen=True)
class CurriculumStage:
    """One staged spawn. `granted_keys` is OR'd into player.keys_held at spawn.
    `checkpoint_x` (forward curriculum) is a finish line: when set, the episode
    ends and counts as reached the frame the ball's x crosses it. None => the
    stage runs to the real goal (reverse curriculum / final forward stage)."""

    spawn_xy: tuple[float, float]
    granted_keys: int
    label: str   # "near_goal" | "before_key<id>" | "start" | "to_cp<i>" | "to_goal"
    checkpoint_x: float | None = None
```

- [ ] **Step 4: Make `evaluate_curriculum` checkpoint-aware** in `src/blueball/ai/curriculum.py`. Change the unpack line and the substep loop, and feed `reached` into fitness + return.

Replace the args unpack:

```python
    idx, genome, world_seed, level_path, max_steps, spawn_xy, granted_keys, checkpoint_x = args
```

Replace the substep loop (currently `while steps < max_steps: ... if player.dead or player.reached_goal: break`) with:

```python
    reached_checkpoint = False
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator residual,
        # so long headless runs are bit-identical across machines (see trainer).
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.body.position.y < min_y:
            min_y = player.body.position.y
        if checkpoint_x is not None and player.body.position.x >= checkpoint_x:
            reached_checkpoint = True
            break
        if player.dead or player.reached_goal:
            break

    # Crossing the forward finish line counts as "reached" for both the fitness
    # terminal bonus and the stage-advancement signal — mirroring how a reverse
    # near-goal stage is credited for reaching its (then-fixed) goal.
    reached = bool(player.reached_goal) or reached_checkpoint
```

Then update the `fitness(...)` call to use `reached_goal=reached` (replace `reached_goal=bool(player.reached_goal)`), and update the return:

```python
    return idx, float(f), bool(reached)
```

- [ ] **Step 5: Thread `checkpoint_x` through `train_curriculum`'s args** in `src/blueball/ai/curriculum.py` (the `args_iter` list comprehension):

```python
        args_iter = [
            (i, population[i], world_seed, str(level_path), max_steps,
             stage.spawn_xy, stage.granted_keys, stage.checkpoint_x)
            for i in range(pop_size)
        ]
```

- [ ] **Step 6: Update the CLI verdict call** in `src/blueball/cli.py` (`cmd_train_maze`, the `evaluate_curriculum((...))` call). `start` is `stages[-1]`, whose `checkpoint_x` is `None`:

```python
    _, fit, reached = evaluate_curriculum(
        (0, result.best_genome, args.world_seed, level_path, args.max_steps,
         start.spawn_xy, start.granted_keys, start.checkpoint_x))
```

- [ ] **Step 7: Fix the existing tuple call sites** (add the 8th element).

In `tests/test_ai_curriculum.py`:
- Line ~90 (`test_evaluate_curriculum_returns_idx_fitness_reached`): `(7, g, 1, path, 120, start.spawn_xy, start.granted_keys, start.checkpoint_x)`
- Line ~127 (`test_evaluate_curriculum_granted_keys_dont_inflate_fitness`): `evaluate_curriculum((0, g, 1, path, 120, start.spawn_xy, 0, None))`
- Line ~128: `evaluate_curriculum((0, g, 1, path, 120, start.spawn_xy, 0xFF, None))`

In `tests/test_climb_fitness.py`: find both `evaluate_curriculum((...))` calls (around lines 54 and 84) and append `, None` as the final tuple element before the closing `)`.

- [ ] **Step 8: Run the full suite to verify green**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest -q`
Expected: PASS (no failures; same count as before plus the 2 new tests).

- [ ] **Step 9: Commit** (only if the user has authorized committing — per the project's commit-cadence convention, otherwise leave staged for an explicit "commit")

```bash
git add src/blueball/ai/curriculum.py src/blueball/cli.py tests/test_ai_curriculum.py tests/test_climb_fitness.py
git commit -m "feat(ai): checkpoint_x finish-line + early termination in curriculum evaluator"
```

---

### Task 2: Forward-curriculum build path + `LevelMeta` fields

**Goal:** A level can declare `start_gated: true` (with optional `curriculum_checkpoints`), and `build_spawn_curriculum` emits forward stages — all spawning at the true start, ascending checkpoints, final stage running to the real goal.

**Files:**
- Modify: `src/blueball/levels/loader.py` (`LevelMeta` fields + `load_level` reads them)
- Modify: `src/blueball/ai/curriculum.py` (`_forward_stages` helper + branch in `build_spawn_curriculum`)
- Test: `tests/test_ai_curriculum.py` (forward-build tests via in-memory dicts)
- Test: `tests/test_levels.py` (loader reads the new fields) — create the test there if the file exists; otherwise add to `tests/test_ai_curriculum.py`.

**Acceptance Criteria:**
- [ ] `LevelMeta` has `start_gated: bool = False` and `curriculum_checkpoints: tuple[float, ...] = ()`, populated by `load_level` from JSON keys `"start_gated"` / `"curriculum_checkpoints"` (defaulting false / empty).
- [ ] `build_spawn_curriculum` on a `start_gated` level returns: N+1 stages for N checkpoints; every stage `spawn_xy == (start_x, start_y)`; `granted_keys == 0` throughout; `checkpoint_x` equals the declared checkpoints in order; the final stage has `checkpoint_x is None` and label `"to_goal"`.
- [ ] When `curriculum_checkpoints` is omitted, checkpoints fall back to the level's key x-positions ascending.
- [ ] A non-`start_gated` level is unaffected (reverse path).
- [ ] Full suite green.

**Verify:** `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** in `tests/test_ai_curriculum.py`:

```python
def _maze_dict_start_gated(checkpoints=None):
    """The real maze level as a dict, flagged start_gated (optionally with
    explicit checkpoints). Used to exercise the forward path without flipping
    the on-disk maze.json."""
    import json
    from blueball.ai.episodes import resolve_level_paths
    d = json.loads(Path(resolve_level_paths(["maze"])[0]).read_text())
    d["start_gated"] = True
    if checkpoints is not None:
        d["curriculum_checkpoints"] = checkpoints
    else:
        d.pop("curriculum_checkpoints", None)
    return d


def test_build_spawn_curriculum_forward_stages_for_start_gated():
    from blueball.ai.curriculum import build_spawn_curriculum
    d = _maze_dict_start_gated([300.0, 1056.0, 2432.0])
    stages = build_spawn_curriculum(d)
    start_xy = (float(d["spawn"][0]), float(d["spawn"][1]))
    assert len(stages) == 4                       # 3 checkpoints + final goal
    assert all(s.spawn_xy == start_xy for s in stages)   # all spawn at true start
    assert all(s.granted_keys == 0 for s in stages)
    assert [s.checkpoint_x for s in stages] == [300.0, 1056.0, 2432.0, None]
    assert stages[-1].label == "to_goal"
    assert stages[0].checkpoint_x == 300.0        # isolates the spike_wall [96..224]


def test_build_spawn_curriculum_forward_fallback_uses_keys():
    from blueball.ai.curriculum import build_spawn_curriculum
    d = _maze_dict_start_gated(checkpoints=None)   # no explicit checkpoints
    stages = build_spawn_curriculum(d)
    # maze keys are at x=1056, 2432 -> two checkpoint stages + final goal
    assert [s.checkpoint_x for s in stages] == [1056.0, 2432.0, None]


def test_loader_reads_start_gated_and_checkpoints():
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    d = _maze_dict_start_gated([300.0, 1056.0])
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(d, world)
    assert meta.start_gated is True
    assert meta.curriculum_checkpoints == (300.0, 1056.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py::test_loader_reads_start_gated_and_checkpoints -q`
Expected: FAIL — `LevelMeta` has no `start_gated` attribute.

- [ ] **Step 3: Add `LevelMeta` fields** in `src/blueball/levels/loader.py`:

```python
@dataclass(frozen=True)
class LevelMeta:
    name: str
    spawn: tuple[float, float]
    background: tuple[int, int, int]
    ground: tuple[int, int, int]
    total_width: float
    starting_abilities: frozenset[Ability] = frozenset()
    # Optional terrain-aware reverse-curriculum spawn waypoints; absent → empty.
    curriculum_spawns: tuple[dict, ...] = ()
    # Start-gated (forward curriculum): difficulty is at the start, so training
    # spawns at the true start and advances a finish line goal-ward. Optional
    # explicit finish-line x's; absent → derived from key positions.
    start_gated: bool = False
    curriculum_checkpoints: tuple[float, ...] = ()
```

- [ ] **Step 4: Populate the fields in `load_level`** in `src/blueball/levels/loader.py` (alongside the existing `curriculum_spawns` read, and add to the `LevelMeta(...)` return):

```python
    start_gated = bool(data.get("start_gated", False))
    curriculum_checkpoints = tuple(float(x) for x in data.get("curriculum_checkpoints", []))
```

and in the `return LevelMeta(...)`:

```python
        start_gated=start_gated,
        curriculum_checkpoints=curriculum_checkpoints,
```

- [ ] **Step 5: Add `_forward_stages` and the branch** in `src/blueball/ai/curriculum.py`. Add the helper near `_stages_from_waypoints`:

```python
def _forward_stages(meta, world) -> list[CurriculumStage]:
    """Start-first (forward) curriculum: every stage spawns at the true start
    (so the opening hazard is always trained); the finish-line checkpoint
    advances goal-ward across the list. granted_keys is 0 everywhere — the agent
    spawns at the start and collects keys by traversing forward. The final stage
    has checkpoint_x=None, so it is the real task (start -> real goal)."""
    start_xy = (float(meta.spawn[0]), float(meta.spawn[1]))
    if meta.curriculum_checkpoints:
        checkpoints = [float(x) for x in meta.curriculum_checkpoints]
    else:
        checkpoints = sorted(
            float(e.position[0]) for e in world.entities
            if type(e).__name__ == _KEY_NAME)
    stages = [
        CurriculumStage(spawn_xy=start_xy, granted_keys=0,
                        label=f"to_cp{i}", checkpoint_x=cp)
        for i, cp in enumerate(checkpoints)
    ]
    stages.append(CurriculumStage(spawn_xy=start_xy, granted_keys=0,
                                  label="to_goal", checkpoint_x=None))
    return stages
```

Then in `build_spawn_curriculum`, immediately after `meta = load_level(level, world)` and BEFORE the `if meta.curriculum_spawns:` check, add:

```python
    # Start-gated levels use the forward curriculum (takes precedence over the
    # reverse waypoint/entity paths below).
    if meta.start_gated:
        return _forward_stages(meta, world)
```

- [ ] **Step 6: Run the forward tests**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (new forward tests pass; existing maze-reverse tests still pass — maze.json is not yet flipped).

- [ ] **Step 7: Run the full suite**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit** (subject to commit-cadence authorization)

```bash
git add src/blueball/levels/loader.py src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(ai): forward (start-gated) spawn-curriculum build path + LevelMeta fields"
```

---

### Task 3: Flip `maze.json` to start-gated + migrate maze-reverse tests

**Goal:** The maze trains forward by default (the A/B-proven direction), and the reverse code path keeps a regression guard via a flag-stripped maze dict.

**Files:**
- Modify: `src/blueball/levels/maze.json` (`start_gated` + `curriculum_checkpoints`)
- Test: `tests/test_ai_curriculum.py` (migrate 2 reverse tests onto a stripped dict; add a maze-forward test)

**Acceptance Criteria:**
- [ ] `maze.json` declares `"start_gated": true` and `"curriculum_checkpoints": [300, 1056, 1700, 2432, 3300]`.
- [ ] `build_spawn_curriculum(maze)` returns 6 forward stages; first `checkpoint_x == 300.0` (just past the `spike_wall` at `x=[96..224]`); all spawn at `x=80`; last `checkpoint_x is None`.
- [ ] The two reverse-assertion tests (`..._maze_orders_and_grants`, `..._maze_unchanged_without_waypoints`) now build from a flag-stripped maze dict and still assert the reverse (receding) shape — i.e. the reverse path is still covered.
- [ ] A tiny `train_curriculum` smoke run on the maze advances through forward stages without error.
- [ ] Full suite green.

**Verify:** `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest -q` → all pass

**Steps:**

- [ ] **Step 1: Add a reverse-dict helper** to `tests/test_ai_curriculum.py` (near `_maze_world`):

```python
def _maze_reverse_dict():
    """The maze level as a dict with the start_gated flag stripped, so it
    exercises the entity-derived REVERSE curriculum on real maze geometry
    regardless of what maze.json declares on disk."""
    import json
    from blueball.ai.episodes import resolve_level_paths
    d = json.loads(Path(resolve_level_paths(["maze"])[0]).read_text())
    d.pop("start_gated", None)
    d.pop("curriculum_checkpoints", None)
    return d
```

- [ ] **Step 2: Migrate the two reverse tests** to use the stripped dict.

In `test_build_spawn_curriculum_maze_orders_and_grants`, replace the line `stages = build_spawn_curriculum(path)` with:

```python
    stages = build_spawn_curriculum(_maze_reverse_dict())
```

(Keep the rest of that test — `_maze_keys(world)`, the receding/grant assertions — unchanged; the geometry is identical.)

In `test_build_spawn_curriculum_maze_unchanged_without_waypoints`, replace its `build_spawn_curriculum(resolve_level_paths(["maze"])[0])` call with `build_spawn_curriculum(_maze_reverse_dict())`.

- [ ] **Step 3: Add a maze-forward test** to `tests/test_ai_curriculum.py`:

```python
def test_maze_file_is_start_gated_forward():
    """The on-disk maze now trains forward: first checkpoint clears the opening
    spike_wall (x=[96..224]); all stages spawn at the true start; final stage
    runs to the real goal."""
    from blueball.ai.curriculum import build_spawn_curriculum
    from blueball.ai.episodes import resolve_level_paths
    stages = build_spawn_curriculum(resolve_level_paths(["maze"])[0])
    assert [s.checkpoint_x for s in stages] == [300.0, 1056.0, 1700.0, 2432.0, 3300.0, None]
    assert all(s.spawn_xy[0] == 80.0 for s in stages)
    assert stages[-1].label == "to_goal"
```

- [ ] **Step 4: Add a forward smoke test** to `tests/test_ai_curriculum.py`:

```python
def test_train_curriculum_runs_forward_on_maze():
    """A tiny forward-curriculum run on the real maze completes and finishes on a
    forward stage label."""
    from blueball.ai.curriculum import train_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    result = train_curriculum(level_path=path, pop_size=6, generations=3,
                              ga_seed=0, world_seed=1, max_steps=400)
    last = result.history[-1]
    assert last["stage_label"].startswith("to_")   # forward label, not "start"/"near_goal"
```

- [ ] **Step 5: Run to verify the new forward tests fail** (maze.json not yet flipped)

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest tests/test_ai_curriculum.py::test_maze_file_is_start_gated_forward -q`
Expected: FAIL — maze is still reverse, so `checkpoint_x` values are all `None` / labels differ.

- [ ] **Step 6: Flip `maze.json`** — add the two keys at the top level (alongside `"spawn"`, `"starting_abilities"`):

```json
  "start_gated": true,
  "curriculum_checkpoints": [300, 1056, 1700, 2432, 3300],
```

(Checkpoints: 300 just past the `spike_wall` opening; 1056 & 2432 the two keys; 1700 between them; 3300 between the second key and the goal at 4192. The final goal stage is appended automatically.)

- [ ] **Step 7: Run the full suite**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m pytest -q`
Expected: PASS — forward tests pass; migrated reverse tests pass; train_curriculum/CLI maze tests (which compute stage counts dynamically) pass on the new forward stages.

- [ ] **Step 8: Manual sanity check** — confirm the static path is untouched (start_gated only affects the curriculum trainer, not `train levels`):

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python main.py train maze --pop 8 --gens 3 --world-seed 1`
Expected: prints a curriculum run with forward stage labels (`to_cp0` … `to_goal`) and a `Verdict @ true start: reached_goal=... fitness=...` line, no errors.

- [ ] **Step 9: Commit** (subject to commit-cadence authorization)

```bash
git add src/blueball/levels/maze.json tests/test_ai_curriculum.py
git commit -m "feat(ai): maze adopts forward start-gated curriculum; reverse tests use stripped dict"
```

---

## Self-Review

**Spec coverage:**
- Detection via `start_gated` flag + optional `curriculum_checkpoints` → Task 2 (loader) + Task 3 (maze opts in). ✓
- `CurriculumStage.checkpoint_x` + early-termination lever → Task 1. ✓
- Forward stages spawn at true start, `granted_keys=0`, final `checkpoint_x=None` → Task 2. ✓
- Keys-fallback when checkpoints omitted → Task 2 (`_forward_stages` + test). ✓
- Reverse output unchanged / regression guard → Task 2 (maze still reverse) + Task 3 (stripped-dict migration). ✓
- True-start verdict stays honest (final stage `checkpoint_x=None`) → Task 1 (CLI passes `start.checkpoint_x`). ✓
- Known wart (cross-stage `best_genome` inflation) → documented in spec; no code change, consistent with reverse. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; checkpoint values are concrete (verified geometry). ✓

**Type consistency:** `checkpoint_x: float | None` used consistently across `CurriculumStage`, the args tuple (8th element), `_forward_stages`, and all call sites. `LevelMeta.start_gated: bool`, `curriculum_checkpoints: tuple[float, ...]` consistent between loader and `_forward_stages`. Labels `to_cp<i>` / `to_goal` consistent between `_forward_stages` and tests. ✓

**Out of scope (deferred, per spec):** checkpoint tuning beyond the maze default; generalist maze episode wiring; the gens/seeds push from 70% → goal.
