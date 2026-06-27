# Start-First (Forward) Spawn Curriculum — Design

**Date:** 2026-06-27
**Status:** Approved (brainstorm), pending implementation plan
**Relates to:** `docs/superpowers/specs/2026-06-06-maze-curriculum-design.md` (the
reverse curriculum this mirrors), `docs/superpowers/runbooks/2026-06-24-purposeful-jump-ab.md`
(the A/B that diagnosed the bug).

## Problem

The maze specialist kept reflexively jumping into the deadly `spike_wall` right
after spawn (x≈105–130 from spawn x=80) and dying ~50–135px in. The A/B run
(2026-06-25) traced this not to perception, capacity, or physics, but to
**curriculum direction**:

- The existing reverse spawn-curriculum keeps the goal fixed and recedes the
  *spawn* start-ward as the population masters each stage. The true start is
  therefore trained **last** — and because the population gets stuck at an early
  stage, the spawn never recedes far enough to reach the start at all. So the
  agent is **never trained on the opening hazard it dies to.**
- The reverse curriculum is built for **goal-gated** levels (difficulty near the
  goal). The maze is **start-gated** (difficulty at the start), so the reverse
  curriculum is exactly backwards.

Manually training from the true start (`train levels --levels maze`, static)
took maze progress from 135px → 2973px (~70%, a 22× improvement). This spec
makes that direction a first-class, level-declared curriculum instead of a
manual workaround — and stages it so it also helps push past the 70% plateau.

## Goal

A **start-first (forward) spawn curriculum**: for levels that declare themselves
start-gated, training spawns at the true start every generation (so the opening
hazard is always trained) and advances a moving **finish-line checkpoint**
goal-ward as each segment is mastered.

Non-goals (explicitly deferred): tuning checkpoint counts/positions beyond a
sensible maze default; wiring the generalist's maze episode to this; the actual
gens/seeds push from 70% → goal. This slice delivers the **mechanism**.

## Design

### Core idea — mirror the reverse curriculum

The forward curriculum is the mirror image of the reverse one:

```
Reverse (goal-gated):   START ........ [spawn→] ...... GOAL   (spawn recedes left)
Forward (start-gated):  START [spikes] ......[→checkpoint]... GOAL   (finish line extends right)
```

- **Reverse:** goal fixed, spawn moves start-ward.
- **Forward:** spawn fixed at the true start, finish-line checkpoint moves
  goal-ward.

For a start-gated level this is correct by construction: stage 0 is "spawn at
start, survive the spikes, reach checkpoint 0"; later stages extend the finish
line; the final stage's line *is* the real goal.

Everything stays inside the existing trio — `build_spawn_curriculum`,
`evaluate_curriculum`, `train_curriculum` — plus the level loader. No new
training command, no new CLI path. `train maze` keeps working; it simply trains
in the correct direction because the level declares itself start-gated.

### 1. Detection — `start_gated` flag (explicit)

A level declares its direction in JSON, threaded through `LevelMeta` exactly like
`starting_abilities` / `curriculum_spawns`:

- `"start_gated": true` → forward curriculum.
- absent / false → existing reverse curriculum (every current level unchanged).

Optional companion field `"curriculum_checkpoints": [x0, x1, ...]` — ascending
finish-line x positions. If omitted, checkpoints are derived generically (key
x's ascending, then the goal). For the maze we declare explicit checkpoints with
**checkpoint 0 placed just past the `spike_wall`**, so "beat the opening" is its
own first stage — the spot the reverse curriculum never reached.

`build_spawn_curriculum` branches on `meta.start_gated`:

- false → existing reverse stages (output **byte-identical** to today).
- true → forward stages: every stage spawns at the true start, `granted_keys=0`
  (the agent collects keys naturally by traversing forward), with ascending
  `checkpoint_x`; the final stage has `checkpoint_x=None`.

### 2. What makes it a curriculum (not just static)

`CurriculumStage` gains an optional field `checkpoint_x: float | None = None`.

In `evaluate_curriculum`, when `checkpoint_x` is set, the episode **terminates
early** the moment `player.body.position.x >= checkpoint_x`, and that crossing
counts as `reached` (the same per-generation success signal `train_curriculum`
already consumes to advance a stage).

The early-termination cap is the curriculum lever. Without it, fitness is just
`progress_x` and the GA maximizes raw distance regardless of stage — i.e.
identical to static-from-start, with checkpoints as mere bookkeeping. The cap
makes all genomes that reach checkpoint i tie on progress, so early generations
are selected for **reliably and efficiently clearing the current segment**
before the population is asked to go further — instead of a lucky long-runner
dominating before anyone robustly beats the spikes.

The **final** forward stage has `checkpoint_x=None`, so it is the real task
(spawn at start → real goal). `cmd_train_maze`'s true-start verdict re-eval
(`stages[-1]`) therefore stays honest with no special-casing.

### Plumbing

- `evaluate_curriculum`'s args tuple grows to carry `checkpoint_x` (currently a
  7-tuple → 8-tuple). Update its two producers: `train_curriculum`'s
  `args_iter`, and `cmd_train_maze`'s verdict call (passes `None`). Update any
  tests that construct the tuple directly.
- `reached` returned by `evaluate_curriculum` = real `reached_goal` OR (checkpoint
  set AND crossed). On the final stage (`checkpoint_x is None`) this is exactly
  today's `reached_goal`.

### Known wart (inherited, not introduced)

`best_genome` is the argmax of in-loop fitness across stages with different
scaffolding — the reverse curriculum already documents this as "only meaningful
when re-evaluated from the true start after training," which `cmd_train_maze`
does. The forward curriculum has the same property; we keep parity and keep the
caveat rather than expand scope here.

## Testing (TDD)

- `build_spawn_curriculum` on a start-gated level → all stages spawn at the true
  start; `checkpoint_x` ascending; final stage `checkpoint_x is None`;
  `granted_keys == 0` throughout.
- `build_spawn_curriculum` on a normal level → reverse output **byte-identical**
  to before (regression guard against the branch leaking).
- Explicit `curriculum_checkpoints` honored; omitted → derived from keys + goal.
- `evaluate_curriculum` terminates and reports `reached=True` exactly at a
  checkpoint crossing; with `checkpoint_x=None` behaves as today.
- `maze.json` loads with `start_gated` + checkpoints; a tiny `train_curriculum`
  smoke run advances through forward stages without error.
- Full existing suite stays green (reverse-curriculum and loader tests).

## Files

- `src/blueball/levels/loader.py` — `LevelMeta.start_gated`, read from JSON.
- `src/blueball/levels/maze.json` — `start_gated: true` + `curriculum_checkpoints`.
- `src/blueball/ai/curriculum.py` — `CurriculumStage.checkpoint_x`; forward branch
  in `build_spawn_curriculum`; checkpoint handling in `evaluate_curriculum`;
  thread `checkpoint_x` through the args tuple in `train_curriculum`.
- `src/blueball/cli.py` — `cmd_train_maze` verdict call passes `checkpoint_x=None`.
- Tests under `tests/` mirroring the cases above.
