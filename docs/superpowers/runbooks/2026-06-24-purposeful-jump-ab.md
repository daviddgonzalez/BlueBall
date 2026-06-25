# Purposeful-Jump Overhaul — A/B Runbook

**Date:** 2026-06-24
**Spec:** `docs/superpowers/specs/2026-06-24-purposeful-jump-overhaul-design.md`
**Plan:** `docs/superpowers/plans/2026-06-24-purposeful-jump-overhaul.md`
**Branch:** `worktree-purposeful-jump-overhaul`

## Purpose

Decide whether the purposeful-jump overhaul (jump-state perception + efficiency
reward + gap-gated jump bonus) actually reduces bounce-spam and improves
completion vs. the pre-overhaul baseline — measured, not eyeballed.

## What changed (treatment)

- Observation grew 35→37 inputs: `air_jumps_remaining`, `can_jump_now`.
- Fitness adds, in goal-terminal modes only (`level_width > 0`):
  `SPEED_W * progress_x / max(steps, 1)`; and in all modes:
  `JUMP_GAP_BONUS * purposeful_jumps`.
- New tunable knobs (in `src/blueball/config.py`): `SPEED_W = 50.0`,
  `JUMP_GAP_BONUS = 5.0`, `GAP_PROBE_AHEAD_PX = 48.0`, `GAP_PROBE_DEPTH_PX = 120.0`.
- Legacy 510-element genomes still load (zero-pad migration), so old race
  ghosts keep working.

Genome size changed 510→534, so treatment genomes are NOT comparable as files
to baseline genomes — compare *behavior/metrics*, not weights.

## Environment

This work lives in a git worktree whose `blueball` is not the editable install.
Always run with the env prefix so the worktree's `src/` is used:

```bash
PREFIX='SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONPATH="$PWD/src" /home/ddgg0/projects/BlueBall/.venv/bin/python'
```

Run all commands below from the worktree root.

## Procedure

### 1. Capture the CONTROL (pre-overhaul) baseline

"Control" = the code BEFORE this branch's behavior changes. Capture it from a
checkout at the pre-overhaul base commit `ed86ce3` (or `master` before merge) —
NOT from this branch, because the observation/fitness here are the treatment.

From a separate checkout/worktree at `ed86ce3`:

```bash
# maze curriculum specialist (the stuck one)
python main.py train maze --level maze --pop 80 --gens 200 --ga-seed 0 --world-seed 1
# completion gym
python main.py train gym  --pop 80 --gens 200 --max-steps 6000 --ga-seed 0 --world-seed 1 --abilities double_jump
```

Record for each: generations-to-first-goal (maze) / segments cleared (gym),
final completion, and — by watching the best genome — its bounce metrics.

Watch a saved genome (HUD now shows `jumps/100px` and `air%`):

```bash
python main.py watch-best <run_dir>/final_best.npy --level maze
python main.py watch-best <run_dir>/final_best.npy --gym 4242
```

> NOTE: the pre-overhaul HUD does NOT show jumps/100px or air% (those ship with
> this branch). To get baseline bounce numbers, either eyeball the baseline
> genome's behavior, OR load the (zero-pad-migrated) baseline genome under THIS
> branch's `watch-best` so the new metrics HUD renders it. The migrated genome
> behaves identically to the original, so the metrics are faithful to baseline
> behavior. This is the recommended way to get apples-to-apples bounce numbers.

### 2. Run the TREATMENT (this branch)

Identical flags, on `worktree-purposeful-jump-overhaul`:

```bash
python main.py train maze --level maze --pop 80 --gens 200 --ga-seed 0 --world-seed 1
python main.py train gym  --pop 80 --gens 200 --max-steps 6000 --ga-seed 0 --world-seed 1 --abilities double_jump
```

Record the same numbers, plus watch the treatment best genome's bounce metrics.

### 3. Compare

| metric                     | control | treatment |
|----------------------------|---------|-----------|
| maze gens-to-first-goal    |         |           |
| maze final reached_goal    |         |           |
| maze best fitness          |         |           |
| gym segments cleared       |         |           |
| best jumps/100px (maze)    |         |           |
| best air% (maze)           |         |           |
| best jumps/100px (gym)     |         |           |
| best air% (gym)            |         |           |

## Success criteria (from the spec)

The overhaul is a win if BOTH hold:

1. **Completion improves**: higher completion rate and/or fewer
   generations-to-first-completion on the maze curriculum (primary) and/or gym
   (secondary).
2. **Bounce drops**: measurably lower `jumps/100px` and `air%`, with the bounce
   visibly gone when watching the best genome.

If completion improves but bounce metrics don't move (or vice versa), treat it
as partial — investigate which term is (not) pulling its weight before merging.

## Tuning loop

Knobs, in rough order of leverage:

- `SPEED_W` (efficiency strength) — if bounce persists, raise; if the agent
  rushes/suicides or the goal prize gets swamped, lower.
- `JUMP_GAP_BONUS` (purposeful-jump reward) — raise to encourage committing to
  real gaps; set to `0.0` to test efficiency-alone (the spec's fallback).
- `GAP_PROBE_AHEAD_PX` / `GAP_PROBE_DEPTH_PX` (what counts as "a ledge ahead") —
  widen/deepen if real gap-jumps aren't being credited; tighten if flat-ground
  jumps are sneaking credit. Calibrate against the measured jump reach
  (single-jump gap-reach ~420px; see the jump-reachability reference).

Per the project's commit-cadence convention, do NOT auto-commit during tuning —
adjust, re-run, and commit only on an explicit "commit".

## If it plateaus

If the treatment can't beat the baseline after reasonable tuning, the NEAT
escape hatch (specced in `2026-05-23-blue-ball-design.md`) is the next step. It
remains a clean drop-in behind the `Agent` interface — and now sits on a richer
reward/perception baseline to swap behind.

## Pipeline smoke (already verified)

A tiny end-to-end run confirmed the new pipeline trains without error and writes
534-element genomes:

```bash
python main.py train maze --pop 8 --gens 2 --world-seed 1   # -> reached_goal verdict, fitness printed
python main.py train gym  --pop 8 --gens 2 --max-steps 1500 --abilities double_jump --world-seed 1  # -> best/mean printed
```
