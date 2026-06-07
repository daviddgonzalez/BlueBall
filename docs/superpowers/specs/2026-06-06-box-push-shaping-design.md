# Box-Push Shaping + Box-Lava Specialist — Design Spec

**Date:** 2026-06-06
**Branch:** `feature/box-push-shaping`
**Status:** Design approved; ready for implementation plan.

## Problem

The maze reverse spawn-curriculum (PR #2, merged `56a4329`) **stalled at stage 0**
and never cracked maze. Diagnosis: maze's `box_lava_gap` chunk is a **768px lava
pit** (`Lava` sensor x:[3328,4096]) crossed only via a `PushableBox` (size 64,
at x=3294) shoved in as a stepping stone — two ~347px hops, each within the
~413px single-jump reach. Two findings:

1. **The curriculum's `near_goal` stage spawned at x=3312 — 18px *right* of the
   box (3294)** — so the agent was dropped past the box, facing bare lava it
   can't jump, with the box behind it and unusable. The box must be approached
   **from the left**; the reverse-curriculum premise ("spawn near goal, recede")
   breaks on a manipulation puzzle.
2. **The agent gets no reward gradient toward the box-push.** Pushing the box is
   semi-automatic (rolling right into a dynamic body pushes it), but until the
   agent actually lands on the box and continues, `progress_x` doesn't
   distinguish a helpful push from a dive into the lava. So 200 focused
   generations produced nothing.

Verified physically: with the box settled mid-lava its top is at y=603 — 13px
**above** the lava surface (y=616) — and a ball rests on it (`dead=False`). The
intended solution works; the GA just can't find it.

Decision (user): **de-risk with a box-lava *specialist* first** — prove the
reactive controller can push the box and box-hop the lava from a spawn just left
of the box — before redesigning the full-maze curriculum. This spec covers only
that specialist + the reusable box-push shaping.

## Goals

- Add **box-push reward shaping**: a fitness term rewarding rightward
  displacement of a `PushableBox`, giving the GA a gradient up the box-push
  maneuver. Tunable; the reusable piece for the eventual full-maze solution.
- Train a **box-lava specialist**: a genome that, spawned just left of the box
  (both key-gates already behind it, so keys granted), reaches the goal — i.e.
  pushes the box in and crosses the lava.
- Reuse the existing curriculum machinery (a single-stage curriculum is a
  fixed-spawn specialist); keep everything **opt-in and isolated**.
- Keep Infinite-Run, static `train_levels`, the normal curriculum path, and all
  409 current tests **byte-identical / green**.

## Non-goals

- The full-maze curriculum redesign (fixing `near_goal` placement, chaining the
  box-lava stage to the earlier maze) — the next cycle, once the specialist
  proves the maneuver is learnable.
- The generalist (one agent across all levels).
- Box-push shaping in the static `evaluate` / `evaluate_infinite` paths — only
  `evaluate_curriculum` is shaped here (that's where the specialist trains);
  `trainer.py` is **not modified**.
- Changing GA operators, encoding, the goal/death/key/collectible fitness terms,
  or the level itself.

## Design

`trainer.py` is **not touched**. Changes live in `fitness.py`, `config.py`,
`curriculum.py`, and `train_maze_curriculum.py`.

### 1. Box-push shaping (`fitness.py`, `config.py`)

`FitnessInputs` gains a **defaulted** field so every existing construction is
unchanged:

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float
    collectibles: int
    reached_goal: bool
    died: bool
    steps_taken: int
    keys_collected: int
    level_width: float
    box_progress: float = 0.0   # NEW: net rightward PushableBox displacement (px)
```

`fitness()` gains one term:

```
fitness = … (unchanged terms) … + config.BOX_PUSH_MULT * box_progress
```

`config.BOX_PUSH_MULT = 1.0` — a tunable starting guess (lives in config for
iteration). **Not gameable:** a box-pusher that never crosses scores
≈`BOX_PUSH_MULT·box_progress − 200(death)`; a crosser also earns `progress_x`
past the lava plus the `GOAL_MULT·level_width` goal term (~8400), so crossing
always wins. `BOX_PUSH_MULT = 0` reduces exactly to progress-only.

**Byte-identical guarantee:** `box_progress` defaults to 0.0, so `_episode_fitness`
(trainer.py, used by `evaluate`/`evaluate_infinite`) and every existing test
produce identical results. Only callers that explicitly pass `box_progress` (the
curriculum evaluator, below) change.

### 2. Box tracking in `evaluate_curriculum` (`curriculum.py`)

`evaluate_curriculum` already builds its fitness locally (it uses
`fitness()`/`FitnessInputs` directly). It additionally tracks the box:

- After `load_level`, find the box: `box = next((e for e in world.entities if
  type(e).__name__ == "PushableBox"), None)`; `box_start_x = box.body.position.x`
  (or `None`).
- In the substep loop, if `box` is present, track
  `box_max_x = max(box_max_x, box.body.position.x)` (mirrors the player's `max_x`
  high-water mark, robust to knockback).
- At the end: `box_progress = max(0.0, box_max_x - box_start_x)` if a box exists,
  else `0.0`. Pass it into `FitnessInputs(... box_progress=box_progress)`.

Levels with no box → `box_progress = 0.0` → no behavior change. Determinism is
preserved (physics is deterministic).

### 3. Focused box-lava stage (`curriculum.py`)

```python
BOX_LAVA_SPAWN_MARGIN = 12.0   # px left of the box's left face

def build_box_lava_curriculum(level) -> list[CurriculumStage]:
    """Single-stage curriculum for the box-lava section: spawn just left of the
    PushableBox (so rolling right pushes it), with every key granted (both gates
    are behind this spawn). Used to train a box-lava specialist."""
```

It loads the level, finds the `PushableBox` (`box_x`, `box.size`) and the `Key`
entities, and returns exactly one `CurriculumStage`:

- `spawn_xy = (box_x - box.size/2 - BOX_LAVA_SPAWN_MARGIN, true_spawn_y)` — on
  the approach ledge, immediately left of the box;
- `granted_keys = granted_keys_before(keys, spawn_x)` (both keys — gates behind),
  reusing the Task-1 helper;
- `label = "box_lava"`.

`train_curriculum` gains one optional parameter:

```python
def train_curriculum(*, level_path, pop_size, generations, …,
                     stages: list[CurriculumStage] | None = None):
    stages = stages if stages is not None else build_spawn_curriculum(level_path)
```

Default `None` preserves today's behavior exactly. With a one-element `stages`
list, the adaptive loop trains at that fixed spawn and never recedes (it's
already the last stage) — a specialist. `cracked` is `True` iff the elite reaches
the goal from it.

### 4. CLI (`train_maze_curriculum.py`)

A `--box-lava` flag:

- builds `build_box_lava_curriculum(level)` and passes it as `train_curriculum(
  stages=…)`;
- writes to a distinct run dir `mazeboxlavacurr_w<seed>_<ts>` (via
  `run_dir_name(level_name="mazeboxlava", curriculum=True)`, whose key is
  `f"{level_name}curr"` — reusing the existing `curriculum=True` key path, no
  persistence change);
- the verdict re-evaluates the final genome **from the box-lava spawn**
  (`stages[-1]`, the single `box_lava` stage), printing `reached_goal`.

Without `--box-lava` the CLI is unchanged (full reverse curriculum).

### 5. Determinism & back-compat

- Deterministic given `(ga_seed, world_seed)`: box tracking and the custom-stage
  loop add no nondeterminism.
- `box_progress` default 0.0 + `stages` default `None` → Infinite-Run golden,
  `train_levels`, the normal `train_curriculum` path, and all current tests are
  untouched. `trainer.py` is not modified.

## Testing

`fitness.py` (`tests/test_ai_smoke.py`):
- A positive `box_progress` adds exactly `config.BOX_PUSH_MULT * box_progress`;
  all-zero inputs (incl. `box_progress=0`) still return 0.0; the existing formula
  tests stay green (they omit `box_progress` → default 0.0).

`curriculum.py` (`tests/test_ai_curriculum.py`):
- `build_box_lava_curriculum(maze)`: returns exactly **one** stage, label
  `"box_lava"`, spawn x just left of the box (`< box_x` and on the approach
  ledge), `granted_keys` == both maze keys; frame-1 spawn safety (not dead after
  one substep).
- `evaluate_curriculum` box tracking: on maze, an agent that moves right yields
  `box_progress > 0` and a fitness higher (by the box term) than the same run
  scored with `BOX_PUSH_MULT=0`; on a box-less level, `box_progress == 0` (no
  change). (Use a determinism-friendly construction, e.g. compare against a
  recomputed expected.)
- `train_curriculum(stages=build_box_lava_curriculum(maze), …)`: trains at the
  single fixed stage, deterministic run-to-run (identical `best_genome` +
  trajectory); the recorded `curriculum.stages == ["box_lava"]`.
- `train_curriculum` with default `stages=None` is unchanged (existing tests).

CLI:
- `train_maze_curriculum.py --box-lava --pop 4 --gens 2 --workers 1` exits 0,
  writes `mazeboxlavacurr_w1_*` with `final_best.npy` + `run.json` whose
  `curriculum.stages == ["box_lava"]`; stdout has `reached_goal=`.

Full suite (currently 409) green after the additive changes.

## Risks

- **Specialist may still fail.** Even with the box reachable and shaped, the
  box-hop needs jump timing the reactive net may not learn. If so, levers
  (documented, not built): raise `BOX_PUSH_MULT`, nudge the spawn closer to the
  box, more generations, or accept the box-hop needs a richer controller
  (generalist cycle). The verdict + `box_progress` in the run make a partial
  result (pushes box but doesn't cross) legible.
- **Box ends up mis-positioned.** Rewarding pure rightward displacement could
  push the box too far; tolerance is wide (the agent has a double-jump, ~800px
  reach), and `progress_x`/goal dominate, so a usable configuration is selected.
  If overshoot proves common, switch `box_progress` to a distance-to-target
  potential — deferred.
- **`BOX_PUSH_MULT = 1.0` is a starting guess.** Lives in config; selection is
  comparative, so the exact value isn't load-bearing.
