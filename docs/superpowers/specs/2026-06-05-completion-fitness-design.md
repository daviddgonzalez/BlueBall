# Completion-Oriented Fitness + Min Aggregation — Design Spec

**Date:** 2026-06-05
**Branch:** `feature/completion-fitness`
**Status:** Design approved; ready for implementation plan.

## Problem

The multi-episode harness shipped, and the first static-level generalist
(`genomes/lvls5_w1_20260605-121158/`) makes partial progress on every level but
**completes none** (normalized per level: tutorial_hill .44, vertical_climb .42,
speed_run .38, lava_rising .32, maze .23). Two root causes:

1. **The goal reward is drowned out.** Fitness is
   `progress_x + 100·keys + 50·collectibles + 200·goal − 0.01·steps − 200·died`.
   On a ~2000–4200-wide level, `progress_x` alone reaches ≈`level_width`, so
   actually *touching the goal* adds only +200 on top — a ~5–10% bump. Selection
   barely distinguishes a finisher from an agent that just travels far.

2. **The aggregation fights "complete all levels."** `train_levels` scores a
   genome across the levels with `mean − λ·std`. Both λ regimes fail the goal:
   - **λ=0 (pure mean):** a lucky clear of an easy level (e.g. `speed_run`) pumps
     the average while a hard level (`maze`) is neglected — the GA farms easy
     levels.
   - **λ>0 (mean − λ·std):** clearing one level raises variance, so it's
     penalized; the GA can satisfy the penalty by *leveling down* (getting worse
     at the easy level) instead of improving the hard one, converging to uniform
     mediocrity.

   The objective we actually want — "raise the floor: improve the *worst* level
   until all are cleared" — is the **min** (worst-case) objective.

(Investigating during design also corrected an earlier assumption: `maze.json`
is a *linear* left-to-right gauntlet — dense hazards plus two `key→door` gates
where each key sits on the path before its door — **not** a 2-D maze needing
spatial memory. So completing it is a fitness + survival problem, reachable
without network recurrence.)

## Goals

- Make **reaching the goal dominate** any unfinished run, on every level,
  without a magic constant that breaks when a level gets longer.
- Make the levels trainer **relentlessly target the weakest level** so it pushes
  toward completing *all* of them, and cannot be gamed by neglecting a hard
  level or by leveling down a strength.
- Keep Infinite-Run training **byte-identical** (the committed golden and 3-seed
  generalist must still reproduce).

## Non-goals

- Reverse curriculum / staged spawns — deferred (cycle 1b if fitness alone
  doesn't yield completions).
- Network capacity / recurrence — deferred (cycle 2; and likely reframed given
  the levels are linear).
- Survival/death-penalty retuning, distance-to-goal shaping — out of scope; the
  goal-dominance + min objective is the focused change.
- Changing GA operators, encoding, or `TrainScene`.

## Design

### 1. Width-scaled goal bonus (`config.py`, `fitness.py`)

Add one config constant:

```python
GOAL_MULT = 2.0  # goal is worth GOAL_MULT full level-lengths (completion bonus)
```

`FitnessInputs` gains one field, and the goal term scales with level width:

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float
    collectibles: int
    reached_goal: bool
    died: bool
    steps_taken: int
    keys_collected: int
    level_width: float   # NEW — the level's total width; 0.0 for goalless modes
```

```
fitness =
      progress_x
    + 100.0 · keys_collected
    +  50.0 · collectibles
    + GOAL_MULT · level_width · reached_goal      # was: 200.0 · reached_goal
    −   0.01 · steps_taken
    − 200.0 · died
```

`fitness.py` picks up `from .. import config` (pure constants, no weight).

**Why it dominates:** a non-completer's `progress_x` maxes at ≈`level_width`, so
it scores ≤ `width + (key/collectible bonuses)`. A completer scores
`≈ width + GOAL_MULT·width = (1 + GOAL_MULT)·width`, winning by ≈`GOAL_MULT·width`
on every level — and it auto-scales, so no constant breaks when a level grows.

### 2. Evaluator passes the width (`trainer.py`)

`_episode_fitness` gains a `level_width` argument, threaded from each evaluator:

- `evaluate()` (static): `level_width = meta.total_width`.
- `evaluate_infinite()`: `level_width = 0.0`.

**Infinite-Run is provably untouched:** `reached_goal` is always `False` there,
so `GOAL_MULT · level_width · False == 200 · False == 0` regardless of width.
The committed golden and 3-seed generalist reproduce byte-for-byte.

### 3. Normalization sync (`episodes.py` — `compute_level_par`)

Keep "normalized ≈ 1.0 means fully solved" by syncing the par goal term to the
fitness goal term:

```
# before:  + 200.0 * has_goal
# after:   + config.GOAL_MULT * total_width * has_goal
```

So `par = total_width·(1 + GOAL_MULT·has_goal) + 100·keys + 50·collectibles`.
A solved agent's raw fitness ≈ par → norm ≈ 1.0; a full-traversal-but-no-goal
run lands at ≈ `1/(1+GOAL_MULT)`. Levels stay mutually comparable.

### 4. Min aggregation (`episodes.py`, `trainer.py`, `train_levels.py`)

`aggregate_fitness` gains a mode:

```python
def aggregate_fitness(scores, lam, mode="mean_std") -> float:
    # "mean_std": mean - lam*std   (unchanged default)
    # "min":      min(scores)      (lam ignored)
    # empty -> ValueError; single score -> returns it exactly under BOTH modes
```

`evaluate_episodes`'s picklable args become `(idx, genome, episodes, lam, mode)`;
`train()` gains `aggregate: str = "mean_std"` (default preserves today's
behavior) and records it in the `run.json` finalize meta. `train_levels.py`
passes `aggregate="min"`; `train_infinite.py` is unchanged (keeps `mean_std`).
`TrainScene`'s separate eval path is unaffected.

**Why min:** score = the worst level's normalized score. A lucky easy clear
doesn't raise it (the hard level is still the floor), so the GA is forced to
keep improving the weakest level. It cannot be gamed — lowering a strength
doesn't change the min; the only way up is to raise the floor, i.e. complete the
levels one weakest-first. The worst level's `progress_x` still supplies a dense
gradient, so the objective isn't flat early.

### Why single-episode runs stay byte-identical

For a one-element score list, `min([x]) == x == mean([x]) − lam·0`. So both the
default `mean_std` path and the new `min` path return the raw fitness exactly for
single-episode evaluation. Combined with the Infinite-Run goal-term invariant
above, `train(infinite_seed=…)` and `train_infinite.py` are unchanged.

## Testing

`fitness.py` tests (`tests/test_ai_smoke.py`):
- Update the formula/death tests to the width-scaled goal term.
- All-zero `FitnessInputs` (incl. `level_width=0`) → `0.0` (unchanged).
- **Completion dominance:** `fitness(reached_goal=True, progress_x=W, level_width=W)
  − fitness(reached_goal=False, progress_x=W, level_width=W) == GOAL_MULT·W`.
- **Infinite invariant:** with `reached_goal=False`, fitness is independent of
  `level_width` (two different widths give equal fitness).

`compute_level_par` tests (`tests/test_ai_multiepisode.py`):
- tutorial_hill par updates to `total_width·(1+GOAL_MULT)` (goal, no keys/collectibles).
- flat-only level → par == width; empty level → 1.0 (both unchanged).

`aggregate_fitness` tests (`tests/test_ai_multiepisode.py`):
- `min` mode: `aggregate_fitness([0.2, 1.0, 0.5], lam=1.0, mode="min") == 0.2`;
  single score under `min` returns itself.
- Existing `mean_std` tests keep passing via the default `mode` argument.

Trainer tests (`tests/test_ai_multiepisode.py`):
- `evaluate_episodes` with `mode="min"` returns the min of per-episode normalized
  scores; update the two existing direct-call tests to the 5-tuple args.
- `train(aggregate="min", episodes=[…])` smoke + run-to-run determinism.
- `train_levels` subprocess test asserts `run.json["aggregate"] == "min"`.

Backward-compat (must stay green unchanged):
- Infinite-Run fitness unchanged → `train_infinite` determinism / pool-equality
  tests, and the golden/3-seed reproduction, all hold.
- Static-level determinism tests assert finiteness/run-to-run equality (not exact
  values), so they stay green; only the direct fitness/par formula tests update.

Full suite (currently 388) green after the formula-test updates.

## Risks

- **The committed `lvls5` generalist is now stale** — it was trained on the old
  objective. Earning completions requires a *retrain* under the new fitness + min
  (`python train_levels.py`), which is the post-merge payoff run, not part of this
  spec's code.
- **`min` is harsh early.** If training stalls on a brutal level, the documented
  next lever is a soft-min (low percentile) or a mean/min blend.
- **`GOAL_MULT = 2.0` is a tunable starting guess**; it lives in `config.py` for
  easy iteration. No production caller depends on absolute static-level fitness
  values (selection is comparative), so changing it is safe.
