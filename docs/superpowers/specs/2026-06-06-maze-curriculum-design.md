# Maze Reverse Spawn-Curriculum — Design Spec

**Date:** 2026-06-06
**Branch:** `feature/maze-curriculum`
**Status:** Design approved; ready for implementation plan.

## Problem

The completion-fitness + min-aggregation change shipped (master `713c3b9`), and a
full `train_levels.py --aggregate=min` run was done. It **completed no levels**;
the best generalist reaches only ~10–15% (normalized) on every level and is, on
average, slightly worse than the pre-feature mean-based genome.

A diagnostic (single-level specialists, `train_levels.py --levels <one>`) split
the failure cleanly:

- **The network is capable.** A `tutorial_hill`-only specialist reaches the goal
  (norm 0.984); a `speed_run`-only specialist reaches the goal (norm 0.987). So
  the FTNN controller + fitness shaping can solve levels.
- **maze is a genuine wall — even solo.** A `maze`-only specialist still dies at
  ~37% of the level (norm 0.105 ≈ the multi-level generalist's 0.082). No amount
  of aggregation tuning helps a level that a dedicated specialist can't crack.
- **The generalist is a separate, harder problem.** Even the old *mean*-based
  generalist reached only 0.16 on `tutorial_hill` (a level a specialist solves at
  0.98), so one stateless reactive 510-param genome can't jointly serve five
  disparate levels.

Decision (user): **crack maze first via a curriculum, then return to the
generalist.** This spec covers only the maze curriculum.

maze is a *linear* left-to-right gauntlet (width 4224, true spawn x=80) with two
`key→door` gates — Key1 at x≈1056, Key2 at x≈2432, each key on the path before
its door — plus dense hazards. So completing it is a survival + distance problem,
reachable without recurrence; the blocker is that the GA never stumbles onto a
full-traversal run to get reward signal for the late, hard segments.

## Goals

- Train a **maze specialist** that, spawned at the **true start (x=80)**, reaches
  the goal — i.e. crack the wall the min run could not.
- Do it with a **reverse spawn-curriculum**: spawn the agent partway through
  (near the goal first), and recede the spawn toward the true start as the
  population masters each stage — so the agent always has a reachable goal to get
  reward from, then learns to finish from progressively farther back.
- Keep the curriculum **training-only**: the saved genome is an ordinary genome
  evaluated from the real spawn, so it plays maze from x=80 with no scaffolding.
- Keep everything **opt-in and isolated**: the existing `trainer.py`, `evaluate`,
  Infinite-Run golden, and all current tests stay byte-identical.
- Design the mechanism so it can be **reused** for other hard levels and the
  later generalist cycle.

## Non-goals

- The generalist (one agent across all levels) — that is the next cycle.
- Solving levels other than maze in this spec (the mechanism is reusable, but only
  maze is wired and validated here).
- Network capacity / recurrence changes — out of scope.
- Raycast-to-ground spawn probing, forward/level-simplification curricula,
  reward reshaping — explicitly deferred (see Risks).
- Changing GA operators, encoding, fitness, or aggregation.

## Design

A new, self-contained subsystem. `trainer.py` is **not modified**.

### 1. Curriculum stages (key-structure-aware) + key-granting (`ai/curriculum.py`)

```python
@dataclass(frozen=True)
class CurriculumStage:
    spawn_xy: tuple[float, float]   # where the agent spawns this stage
    granted_keys: int               # bitmask OR'd into player.keys_held at spawn
    label: str                      # e.g. "near_goal", "before_key2", "start"
```

`build_spawn_curriculum(level_path) -> list[CurriculumStage]` loads the level
once, reads the `Key` entities (each exposes `key_id` and a world x), the `Goal`
entity x, and the level's true spawn, and derives an ordered stage list
**easiest → hardest** (spawn receding start-ward). For maze (Key1≈1056,
Key2≈2432, start=80) with a `SPAWN_MARGIN` (so the agent spawns just *before* a
key, on solid ground, and must move right to collect it):

| order | label | spawn x | granted_keys | the agent must |
|------:|-------|--------:|--------------|----------------|
| 0 | `near_goal`    | between last key and goal | Key1 \| Key2 | survive the final stretch to the goal |
| 1 | `before_key2`  | `key2.x - SPAWN_MARGIN`   | Key1        | collect Key2 → open Door2 → goal |
| 2 | `before_key1`  | `key1.x - SPAWN_MARGIN`   | (none)      | collect Key1→Door1→Key2→Door2→goal |
| 3 | `start`        | the level's true spawn    | (none)      | the full level from x=80 |

**Granting rule (exact):** for a stage spawning at x = S,
`granted_keys = OR of (1 << k.key_id) for each Key k with k.x < S`. A door behind
the spawn opens because its `key_id`'s pickup is also behind S (keys precede their
doors), so the bit is granted. Spawn-y reuses the level's true spawn-y (maze is a
flat-floored gauntlet; see Risks for the safety net).

Stage x-values are computed from real entity positions at load time, never
hard-coded; `near_goal.x = (max(key.x) + goal.x) / 2`.

### 2. Curriculum evaluator (`ai/curriculum.py`)

`evaluate_curriculum(args)` mirrors `trainer.evaluate` but (a) spawns at a stage
override, (b) grants that stage's keys, and (c) returns whether the goal was
reached — the success signal the adaptive loop needs. Picklable for `Pool`.

```python
# args = (idx, genome, world_seed, level_path, max_steps, spawn_xy, granted_keys)
# returns (idx, fitness, reached_goal)
```

It builds a fresh World, loads the level, spawns `Player(agent, spawn_xy)`, then
sets the granted keys via `player.collect_key(key_id)` for each granted bit,
runs the same drift-free `world.substep()` loop as `evaluate` (break on
dead/`reached_goal`), and returns `(idx, fitness, bool(player.reached_goal))`.
Fitness uses the existing `_episode_fitness`/`FitnessInputs` with
`level_width = meta.total_width` (so the goal bonus is identical to normal eval;
`progress_x` is measured from the stage spawn, which is correct — it rewards
distance covered this stage).

### 3. Adaptive, success-gated training loop (`ai/curriculum.py`)

`train_curriculum(*, level_path, pop_size, generations, ...) -> TrainingResult`
runs a GA loop that holds **curriculum state = a current stage index** (starts at
0, `near_goal`). Each generation:

1. Evaluate the whole population at the **current stage** (spawn + granted keys)
   via `map_fn(evaluate_curriculum, ...)`.
2. Selection uses each genome's `fitness` exactly as today (`breed`, elitism,
   tournament — unchanged GA).
3. **Advancement trigger:** if the generation's **best genome reached the goal**
   from the current spawn, advance `stage_index += 1` (recede toward the start).
   Elitism preserves the best, so the capability is locked in and the signal is
   stable. If not, hold at this stage and keep evolving.
4. Stop conditions: reaching and clearing the final `start` stage (maze cracked),
   or exhausting `generations`.

It reuses `random_genome`, `breed` (with the same `config.GA_*` knobs), and
`TrainingRunWriter` for per-gen snapshots + final genome. `run.json` gains a
**`curriculum` block**: the ordered stage labels and the **stage trajectory**
(the generation index at which each stage was first reached and first cleared),
so a reader can see how the curriculum progressed.

**The saved genome is curriculum-free at eval:** `final_best.npy` is selected by
in-loop fitness, but its headline quality is reported by evaluating it once from
the **true start** (stage `start`, no grants) — that number is the "cracked maze?"
verdict.

### 4. CLI + persistence (`train_maze_curriculum.py`)

A repo-root CLI mirroring `train_levels.py`'s shape: `--pop`, `--gens`,
`--max-steps`, `--ga-seed`, `--world-seed`, `--workers`. Trains maze via
`train_curriculum`, persists to `genomes/mazecurr_w<seed>_<ts>/`
(`run_dir_name` gains a `mazecurr` variant), prints the final true-start verdict
(reached goal? + normalized fitness) and the stage trajectory.

### 5. Determinism & back-compat

- Given `(ga_seed, world_seed)`, the population, every evaluation, and therefore
  the **stage trajectory** are deterministic → run-to-run identical `best_genome`
  and identical curriculum progression.
- Entirely new module + CLI. `trainer.py`, `evaluate`, `evaluate_episodes`,
  `episodes.py`, fitness, and Infinite-Run golden are untouched → all current
  tests stay green unchanged.

## Testing

`tests/test_ai_curriculum.py`:

- **`build_spawn_curriculum`:** for maze, returns stages ordered easiest→hardest;
  spawn x's strictly **decrease** toward the start; the `start` stage's spawn ==
  the level's true spawn with `granted_keys == 0`; `near_goal` grants **both**
  keys; `before_key2` grants Key1 only; `before_key1` grants none. (Assert against
  the real Key `key_id`s, not literals.)
- **Granting rule:** a unit test that the mask for a given S equals the OR of
  `1<<key_id` over keys with `x < S`.
- **`evaluate_curriculum`:** spawns at the override and the granted keys are set
  on the player (`keys_held` has the expected bits before stepping); returns a
  3-tuple `(idx, fitness, reached_goal)`; with a granted key, a door behind the
  spawn is passable (agent isn't hard-blocked at frame 1).
- **Frame-1 spawn safety:** at **every** maze stage spawn, the player is **not
  dead after one substep** (catches a stage spawned into geometry/over a pit).
- **`train_curriculum` determinism:** same seeds → byte-identical `best_genome`
  and identical recorded stage trajectory.
- **Adaptive logic (stubbed evaluator):** inject a fake `evaluate_curriculum` that
  reports `reached_goal=True` → the stage index advances next generation; reports
  `False` → it holds. (Isolates the state machine from physics; fast.)
- **CLI smoke:** `train_maze_curriculum.py --pop 4 --gens 2 --workers 1` exits 0,
  writes `mazecurr_w1_*` with `final_best.npy` + `run.json` containing the
  `curriculum` block.

Backward-compat: existing suite (395) stays green unchanged — no shared code is
modified.

## Risks

- **Spawn-y on a non-flat segment.** v1 reuses the start-y; the frame-1 safety
  test guards it for maze. If a stage ever spawns into geometry/a pit, the fix is
  to nudge that stage's x to the key's on-surface position; the general
  raycast-to-ground probe is deferred to the generalist cycle.
- **Curriculum stalls on an intermediate stage.** If the population can't clear a
  stage within the generation budget, training never reaches `start`. Levers
  (documented, not built here): more generations, a finer stage between the two
  it's stuck on, or a small per-stage patience before giving up. The stage
  trajectory in `run.json` makes a stall obvious.
- **Cracking from near-goal ≠ cracking from start.** Mastering an easy stage
  doesn't guarantee the next recede succeeds; this is expected, and the adaptive
  trigger is exactly what prevents receding before the agent is ready.
- **`SPAWN_MARGIN` is a small tunable** (so the agent spawns just before a key on
  solid ground). It lives as a module constant for easy iteration; selection is
  comparative so the exact value isn't load-bearing.
