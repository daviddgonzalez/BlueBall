# Purposeful-Jump Overhaul — Design

**Date:** 2026-06-24
**Status:** Approved (design); implementation pending
**Author:** David Gonzalez (with Claude)

## Problem

Trained specialists are stuck. Watching them, they appear to **bounce/jump
constantly instead of learning *why* to jump**. The maze curriculum and the
completion gym both fail to reliably reach goals.

### Root-cause analysis (evidence-based)

The bounce is a **limit cycle**, not under-training:
`grounded=1` → net emits JUMP → airborne → `grounded=0` → net does something
else → lands → `grounded=1` → JUMP again. It is a stable fixed point of the
policy × physics loop.

Two structural reasons it persists:

1. **No fitness differential against bouncing.** `ai/fitness.py` is dominated
   by raw `progress_x`, with a near-zero `-0.01 * steps_taken` time cost.
   `AIR_CONTROL = 0.0` (`config.py:39`) means there is *zero* horizontal torque
   while airborne — every airborne frame is dead acceleration time — yet the
   current fitness cannot see that cost. Bouncing forward and rolling forward
   score about the same, so there is no gradient pressure to stop bouncing.
2. **No jump-state perception.** The 35-input observation
   (`ai/observation.py`) gives the net `grounded` but *not* how many air-jumps
   remain, nor whether a jump would actually fire this frame. Even under
   anti-bounce pressure, the net lacks the inputs to respond intelligently.

### Why not the NEAT "escape hatch"

The escape hatch (NEAT via `neat-python`, specced in
`2026-05-23-blue-ball-design.md`) is for one failure: FTNN lacks the
capacity/topology to represent the policy. The evidence says capacity is **not**
the wall:

- 32-seed Infinite traversal generalizes (~7% drop vs overfit) — "genetics are
  fine."
- The net solves levels solo when the task is in-distribution.
- box-lava cracked via a *starting-abilities* fix, not added capacity.

NEAT would spend complexity budget on a problem we don't have. We **defer NEAT**
and keep it a clean drop-in (per the original spec's `Agent`-interface
alignment) in case this fix plateaus.

## Goals

- Make bouncing genuinely cost fitness, via a robust, un-farmable differential.
- Reward *purposeful* jumps (over ledges) without the farming trap that sank the
  box-push `box_progress` shaping.
- Give the net the jump-state inputs needed to act on the new pressure.
- Validate with a controlled A/B that includes **numbers**, not just eyeballing.

## Non-goals

- NEAT escape hatch (deferred; remains architecturally ready).
- Reworking traversal/generalization (probes show it already generalizes).
- "Jump toward goal/key" reward — rejected as farmable (any forward motion is
  roughly "toward" a +x goal, so it would reward bouncing-forward).
- Efficiency reward for goalless Infinite Run (degenerate there; see below).

## Design

### 1. Reward changes — `ai/fitness.py`

Two additive terms on top of the existing formula. The existing terms
(`progress_x`, keys, collectibles, goal, death, segment bonus, climb) are
unchanged.

**1a. Efficiency term (goal modes only).**

```
+ SPEED_W * (progress_x / max(steps_taken, 1))
```

Robust by construction: episodes break early only on death or goal
(`trainer.py:113`), so in goal modes `steps_taken` is real information. A fast
finisher has high `progress_x` and few steps → high ratio; a dawdler/bouncer
burns all `max_steps` → low ratio; a suicide-rusher is crushed by the existing
`-200` death term, and a stop-early agent has tiny `progress_x`. There is no
knife-edge per-step penalty, so there is no suicidal-rushing failure mode.

**Scope:** disabled for goalless Infinite Run, where every healthy agent uses
all `max_steps`, making `progress_x / steps` a monotonic rescale of
`progress_x` (no new differential). The stuck agents are all goal-based, so the
lever lands where the problem is.

**1b. Gap-gated takeoff bonus.**

```
+ JUMP_GAP_BONUS * purposeful_jumps
```

`purposeful_jumps` counts a jump only when, at the moment it fires *from the
ground*, a forward-down ground probe detects a ledge/drop ahead. Un-farmable on
flat ground (probe hits ground → no bonus). This is "jump before a ledge," made
safe.

New field on `FitnessInputs`: `purposeful_jumps: int = 0`.

### 2. Perception changes — `ai/observation.py`, `ai/ftnn.py`

Add **2 jump-state inputs** (currently the net is blind to both):

- `air_jumps_remaining` (normalized to [0, 1]) — whether a double-jump is still
  available.
- `can_jump_now` (1.0 if a jump would actually fire this frame: grounded,
  coyote window, buffered, or air-jump available; else 0.0) — the single most
  informative jump-state signal.

This bumps `INPUT_SIZE` 35 → 37 and `GENOME_SIZE` accordingly, invalidating
saved genomes (sanctioned by the encoding doc; retrain from scratch). Update the
layout doc/offsets in `observation.py` and the `FTNN_INPUTS` import in
`ftnn.py`. The existing runtime shape guard must still pass.

### 3. Gap detection + cheap metrics — `entities/player.py`, `ai/trainer.py`

At the jump-fire site in `Player.update`, when the jump fires and the player was
grounded at takeoff, cast a forward-down probe from a point ahead of the ball
(in the direction of travel). If ground drops away ahead (no ground within a
short check depth) → increment a per-player `purposeful_jumps` counter.

Also tally cheaply, per episode, for A/B metrics:

- `jumps_fired` — total jumps that fired.
- `airborne_steps` — frames spent not grounded.

The evaluators in `trainer.py` read these off the player and pass
`purposeful_jumps` into `FitnessInputs`; `jumps_fired` / `airborne_steps` feed
reported metrics (jumps-per-100px, airtime-%).

### 4. Config — `config.py`

New constants: `SPEED_W`, `JUMP_GAP_BONUS`, and the gap-probe geometry
(`GAP_PROBE_AHEAD_PX`, `GAP_PROBE_DEPTH_PX`). Start from principled estimates;
tune in the A/B.

## Validation (A/B)

- **Control:** current master (35 inputs, old fitness).
- **Treatment:** this design (37 inputs, new fitness).
- Same GA seeds, world seeds, population, generations.
- **Primary target:** maze curriculum specialist (the stuck one).
- **Secondary:** completion gym.
- **Success criteria:**
  - Higher completion rate and/or fewer generations-to-first-completion, **and**
  - measurably lower jumps-per-100px and airtime-%, with the bounce visibly
    gone when watching the best genome.

## Tuning

`SPEED_W` and `JUMP_GAP_BONUS` are the two new knobs; the gap-probe geometry is a
third. All start from principled estimates and get tuned in the A/B. Per the
project's commit-cadence convention, do **not** auto-commit during iterative
tuning — wait for an explicit "commit."

## Risks

- **Over-shaping (box-push lesson).** Mitigated: efficiency is outcome-tied (not
  a behavior proxy) and the gap bonus is un-farmable on flat ground. If the gap
  bonus still distorts, it can be set to 0 and rely on efficiency alone.
- **`SPEED_W` mis-calibration** swamping the goal prize or progress. Mitigated by
  starting small and tuning; the ratio form avoids suicidal-rushing.
- **Genome invalidation** — expected; full retrain. No saved-genome migration.
- **If this plateaus**, the NEAT escape hatch remains the next step, now with a
  cleaner reward/perception baseline to swap behind.
