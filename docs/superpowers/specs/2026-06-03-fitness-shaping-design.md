# Fitness Function Shaping — Design Spec

**Date:** 2026-06-03
**Branch:** `feature/fitness-shaping`
**Status:** Design approved; ready for implementation plan.

## Problem

The GA trainer is hardened and merged, but its fitness function
(`src/blueball/ai/fitness.py`) is still the generic v1 written for static
levels with a goal:

```
progress_x + 50·collectibles + 200·reached_goal − 0.01·steps − 100·died
```

Before spending compute on a real training run, the objective the agent
optimizes needs to be deliberate. Two concrete problems:

1. **Primary term is final displacement, not furthest reached.** `progress_x`
   is `player.x − spawn_x` at episode end. An agent that sprints far then gets
   knocked back or dies falling is scored on where it *ended*, not its best.
   The in-game Infinite Run score is `10 · furthest_x_reached`, so training and
   the score the player actually sees measure different things.

2. **No credit for keys.** On gated levels (e.g. `maze.json`) the goal sits
   behind locked doors. Without rewarding key pickup, the agent gets *zero*
   gradient until it blindly completes the entire key→door→goal sequence — a
   sparse-reward wall a GA struggles to climb. Keys are the natural
   intermediate sub-goal.

## Goals

- Make fitness reward **furthest progress**, aligning Infinite Run training
  with the actual game score and making static-level progress robust to
  knockback/death-fallback.
- Reward **keys collected** as an intermediate sub-goal so the agent can learn
  the gated levels.
- Keep a **modest, tiebreaker-scale death penalty** — discourages pointless
  early death without ever letting a short safe run beat a much longer one.
- One shared `fitness()` serving both training modes; terms that don't apply to
  a given run stay 0 (Infinite Run has no keys/goal chunks, so those terms are
  naturally 0 there).

## Non-goals

- Per-mode or per-level fitness variants — one function, driven by which signals
  are non-zero.
- Velocity/style/exploration shaping, curriculum, multi-seed averaging — out of
  scope; revisit only if training plateaus.
- Changing the GA operators, network, or trainer control flow.

## Design

### Fitness formula

```
fitness =
      max_progress_x            # furthest x reached − spawn  (PRIMARY, dense gradient)
    + 100 · keys_collected      # NEW — popcount(keys_held); intermediate reward toward gated goals
    +  50 · collectibles        # existing pickup reward
    + 200 · reached_goal        # existing — completing a static level
    −   0.01 · steps_taken      # small anti-stall (dawdling costs a little)
    − 200 · died                # modest death nudge (was −100); tiebreaker-scale
```

Magnitudes (all tunable, this is the function most likely to be iterated):
distance dominates (can reach thousands of px); a key (+100) and a collectible
(+50) are meaningful sub-goals; reaching the goal (+200) is the biggest single
event on a static level; death (−200) is a few-hundred-px nudge that never
overrides a large distance lead but breaks ties toward survivors; the
−0.01/step cost mainly demotes agents that sit still.

### Interface change

`FitnessInputs` gains one field:

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float     # NOW: furthest x reached − spawn_x (was final x − spawn_x)
    collectibles: int
    reached_goal: bool
    died: bool
    steps_taken: int
    keys_collected: int   # NEW — bin(player.keys_held).count("1")
```

`progress_x`'s *meaning* changes (furthest, not final); the field name stays.

### Evaluator changes (`src/blueball/ai/trainer.py`)

Both `evaluate` (static) and `evaluate_infinite` (Infinite Run):

- Track the furthest x the player reaches across the step loop:
  `max_x = max(max_x, player.body.position.x)` each iteration (seed with the
  spawn x so a player that never moves yields `progress_x = 0`).
- Pass `progress_x = max_x − spawn_x`.
- Pass `keys_collected = bin(player.keys_held).count("1")`.

`evaluate_infinite` keeps `reached_goal=False`; on Infinite Run `keys_held`
stays 0, so the keys term is 0 there.

### Why one shared function is correct

A training run targets one level (`level_path`) or Infinite Run
(`infinite_seed`). The fitness terms that don't apply to that run are simply 0:
Infinite Run → no keys, no goal, no collectibles; a flat tutorial level → keys 0.
No branching needed; the formula self-selects.

## Testing

In `tests/test_ai_smoke.py`:

- Update `test_fitness_shape_matches_spec_formula` to the new formula (incl.
  `keys_collected`).
- Update `test_fitness_penalizes_death_and_charges_step_cost` for the −200
  death penalty.
- Add `test_fitness_rewards_keys` — `keys_collected=2` adds exactly 200.
- Add an evaluator test that `progress_x` reflects the **max** x reached, not
  the final x: drive a player forward then back (or to death after a peak) and
  assert fitness uses the peak. Use a small deterministic setup.
- Existing determinism / Pool-equality tests must stay green (they assert
  finiteness and run-to-run equality, not exact fitness values).

Full suite (currently 360) must stay green after test updates.

## Risks

- Changing the death constant and adding a term **changes static-level fitness
  numbers**; the direct-formula tests are updated in lockstep. No production
  caller depends on absolute fitness values (selection is comparative).
- `progress_x` semantics change (final → furthest) alters `evaluate` results
  for static levels too; this is intended and strictly a better progress signal.
