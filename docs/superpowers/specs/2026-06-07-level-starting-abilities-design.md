# Level-Declared Starting Abilities (Training/Level Ability Parity) — Design Spec

**Date:** 2026-06-07
**Branch:** `feature/box-push-shaping` (worktree `worktree-feature+box-push-shaping`)
**Status:** Design approved; ready for implementation plan.

## Problem

The box-lava specialist (`train_maze_curriculum.py --box-lava`, 80×200, world=1)
**plateaued at fitness 1020.5 by generation ~55 and never reached the goal**
(`cracked: false`, 0/200 gens elite-reached-goal). Decomposing the final
genome's run from the box-lava spawn `(3250, 540)`:

- `progress_x` = 707.4 (rolled to x=3957), `box_progress` = 518.0 (shoved the box
  3294 → 3812, deep into the lava), **died** (−200) at x=3957/y=600 — i.e. it
  pushed the box in, rolled in behind it, and **fell into the lava 235px short of
  the goal** (x=4192).

Root-cause investigation (all verified in code, 2026-06-07):

1. **The training player has no abilities — including no double jump.** Every
   training path constructs the player with an empty abilities set:
   `make_curriculum_player` (`curriculum.py:162`), `trainer.evaluate`
   (`trainer.py:93,130`), and `scenes/train.py:133`. `_max_air_jumps()` returns
   `0` without `Ability.DOUBLE_JUMP`, so the agent has a **single jump**.
2. **The maze level assumes the player already has double jump.**
   `Ability.DOUBLE_JUMP` is unlocked in **`tutorial_hill.json`** (an
   `ability_pickup` chunk) and persisted via `save.py` (`unlocked_abilities`).
   `maze.json` has **no** ability pickup — a real player arrives at maze with
   double jump carried over. The real game grants it in `scenes/play.py:76`
   (`abilities=unlocked`). **The trainer is solving a strictly harder game than
   real players** — a training/level mismatch.
3. **The network *could* condition on double jump but never has.** The FTNN input
   vector (`ai/observation.py`, INPUT_SIZE=35) already includes an abilities bit
   (index 26 = `DOUBLE_JUMP`) and `grounded` (index 19) — but in training that
   bit is always 0, so the net has zero double-jump experience. (It does **not**
   observe `_air_jumps_remaining`, so second-jump timing would be learned
   implicitly; that is acceptable and out of scope.)

### Why this is the blocker (measured jump envelope)

A scripted-controller measurement on flat ground (takeoff at the 315 px/s speed
cap):

- **Single jump: max range ≈ 415 px.**
- **Double jump: max range ≈ 580 px.**

The `box_lava_gap` chunk (`pit_tiles=24` → 768 px pit, box 64 px) needs the box
as a mid-pit stepping stone, splitting the pit into two gaps:

| box position | gap1 | gap2 | single (415 px) | double (580 px) |
|---|---|---|---|---|
| centered (3712) | 352 | 352 | marginal | **easy** ✓ |
| where the GA actually left it (3812) | **452** | 252 | **impossible** ❌ | **fine** ✓ |

With single jump, the section is impossible at the box position the GA naturally
produces (gap1=452 > 415). With **double jump it is solvable even there**. The
chunk's own `random_params` caps the designed-solvable pit at **5–7 tiles**;
maze uses 24 — consistent with a level authored for a double-jumping player.

The maze reverse spawn-curriculum previously "stalled at stage 0," and **stage 0
(near-goal) *is* the box-lava section** — so the missing ability likely blocked
the whole maze curriculum, not just this specialist.

## Goals

- Give the training player **the abilities a real player has when arriving at
  that level**, declared by the level itself. For maze, that is double jump.
- Apply to **all maze training** (curriculum evaluator, `trainer.evaluate`, and
  the live `TrainScene`) — the principled parity fix, not a box-lava-only hack.
- Keep the encoding stable: the abilities bit is already an FTNN input, so
  `GENOME_SIZE` is **unchanged** and existing saved genomes stay compatible.
- Honor the declared abilities in the **real game** too (`scenes/play.py`), so
  loading a level directly (fresh save) is fair and the semantics are consistent
  everywhere.
- Preserve back-compat: levels that declare no starting abilities behave exactly
  as today; all 420 current tests stay green.

## Non-goals

- The center cradle (box snaps to mid-pit) and the on-box-jump fitness reward —
  **held in reserve**. We grant double jump and retrain first; only if the
  specialist still fails to reach the goal do we add those. (They make the
  maneuver *easier to learn*, not *possible* — double jump is the *possible*.)
- Shrinking the pit / widening the box.
- Any change to GA operators, encoding, fitness terms, the box-push shaping
  (already shipped), or a campaign/level-ordering system.
- Exposing `_air_jumps_remaining` in the observation.

## Design (Approach A — level-declared starting abilities)

The abstraction: a level declares *what abilities the player arrives with*;
training honors it directly (there is no save during training), and the real
game unions it with save-unlocked abilities.

**1. `LevelMeta.starting_abilities`** (`src/blueball/levels/loader.py`)
- Add field `starting_abilities: frozenset[Ability] = frozenset()` to the frozen
  `LevelMeta` dataclass.
- In `load_level`, parse `data.get("starting_abilities", [])` into
  `frozenset(Ability(s) for s in ...)` (same string→enum mapping the
  `ability_pickup` chunk uses: `Ability("double_jump")`). Default empty.

**2. Declare maze's abilities** (`src/blueball/levels/maze.json`)
- Add top-level `"starting_abilities": ["double_jump"]`.

**3. Grant in all maze training** (the chosen scope)
- `make_curriculum_player(world, genome, spawn_xy, granted_keys, abilities=frozenset())`
  — new defaulted param; passes `abilities` to `Player(...)`.
- `evaluate_curriculum` passes `meta.starting_abilities` (it already has `meta`).
- `trainer.evaluate` (the static-level path, `trainer.py:93`) passes
  `set(meta.starting_abilities)`. **`evaluate_infinite` (`trainer.py:130`) is
  left unchanged** — Infinite Run streams terrain with no level file and no
  declared abilities. The static `evaluate` is the *authoritative* fitness path
  for level training (a `TrainScene` on a level dispatches to it).
- `scenes/train.py` passes `set(self.level_meta.starting_abilities)` to its
  *cosmetic display* players (`_start_generation`), so what's shown matches what
  `trainer.evaluate` trains; the Infinite-Run branch builds a `LevelMeta` with
  the default empty set — unchanged.

**4. Real game parity** (`src/blueball/scenes/play.py`)
- Construct the player with `abilities = set(unlocked) | set(meta.starting_abilities)`
  so a level's declared starting abilities hold even on a fresh save / direct
  load. (Save-driven unlocks still accumulate as before.)

**Data flow:** `level JSON → load_level → LevelMeta.starting_abilities →
Player(abilities=…)`. Single source of truth (the level file); every consumer
reads it from `meta`.

## Components & boundaries

- **`loader.py`** — owns parsing/validation of `starting_abilities`. Input: level
  JSON. Output: `LevelMeta.starting_abilities`. Depends on `abilities.Ability`.
- **`make_curriculum_player`** — owns curriculum player construction. New
  `abilities` param; otherwise unchanged.
- **`trainer.evaluate` / `TrainScene` / `PlayScene`** — each reads
  `meta.starting_abilities` and passes it to `Player`. No new logic beyond the
  grant (PlayScene unions with save unlocks).

## Error handling

- Unknown ability string in `starting_abilities` → `Ability(s)` raises
  `ValueError` at load time (fail fast, same as `ability_pickup`).
- Missing/empty `starting_abilities` → default `frozenset()` (back-compat).

## Testing

- `LevelMeta.starting_abilities` defaults to `frozenset()` for a level without
  the field (e.g. `tutorial_hill`, `speed_run`).
- `load_level("maze")` → `meta.starting_abilities == frozenset({Ability.DOUBLE_JUMP})`.
- A curriculum player built on maze (via `make_curriculum_player` with
  `meta.starting_abilities`) has `Ability.DOUBLE_JUMP` and `_max_air_jumps() == 1`.
- `make_curriculum_player` default (no abilities passed) yields an empty-ability
  player — existing curriculum tests stay green.
- An unknown ability string raises `ValueError`.
- PlayScene unions save-unlocked with `meta.starting_abilities` (maze player has
  double jump even with an empty save).
- Full suite stays green (420 + new tests). Genome size unchanged.

## Empirical follow-up (post-merge, not part of the plan's tests)

Retrain the box-lava specialist (`--box-lava`, now double-jump-equipped) at
80×200, world=1, and check the verdict: success is `reached_goal=True`
(`cracked: true`). If it still plateaus, escalate to the reserved cradle +
on-box-jump reward.

## Self-review

- **Spec coverage:** Goal (ability parity) → loader + maze.json + 4 training/play
  grant sites. Back-compat → default-empty frozenset + the `make_curriculum_player`
  default. Encoding stability → abilities bit already in INPUT_SIZE (no
  GENOME_SIZE change). Real-game parity → play.py union.
- **Placeholders:** none.
- **Ambiguity:** "all maze training" is enumerated explicitly (curriculum,
  trainer, TrainScene); PlayScene is real-game parity, not training.
- **Scope:** single focused change; cradle/reward explicitly deferred.
