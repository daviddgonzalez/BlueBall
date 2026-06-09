# Completion Gym — Design Spec

- **Date:** 2026-06-08
- **Status:** Approved design, pre-implementation
- **Author:** David Gonzalez (with Claude)

## 1. Motivation

The AI/GA stack trains well on **Infinite Run** (endless procedural chunk
streaming) and on hand-built campaign levels, but diagnostics from the prior
training thread established a specific gap:

- **Traversal generalizes.** Multi-seed Infinite Run shows the network learns to
  *run and jump* in a way that transfers (32 training seeds → ~7% overfit drop).
- **Completion mechanics are in no training distribution.** Keys, doors, goals,
  and box/lava puzzles never appear in Infinite Run (the `ChunkSampler` only
  emits `sampler_include=True` chunks; goals/keys/doors would break endless
  streaming). So the one trained generalist genome has never had to *collect a
  key, open a door, push a box across lava, and reach a goal* — which is exactly
  what campaign levels demand.

The **Completion Gym** closes that gap: a training mode with the spirit of
Infinite Run (endless, fresh, procedural, deterministic-by-seed) but built from
the *completion* chunks, so the network learns the key→door→box→goal loop by
repetition across many randomized chains.

## 2. Goals / Non-goals

**Goals**
- Train a single generalist genome that handles completion mechanics, to fix the
  campaign-level completion failures.
- Reuse the existing entities, chunks, FTNN, GA, observation encoding, and
  persistence with **no changes to collision handling or entity classes**.
- Guarantee every generated segment is solvable *by construction* (no runtime
  solvability checker).
- Stay deterministic and seed-reproducible like the rest of the trainer.

**Non-goals**
- Freeform procedural level *geometry* generation (the "Approach B" solver path).
  Out of scope; may be revisited later by adding freeform templates.
- New hand-authored campaign levels.
- Any change to the live game / `PlayScene` (training-only feature).

## 3. Key design decisions (settled in brainstorming)

| Decision | Choice |
|----------|--------|
| Episode shape | **Hybrid endless chain**: finite, solvable goal-*segments* materialized back-to-back; reaching one segment's goal streams the next. |
| Difficulty | **Ramp within the chain**: each chain starts with easy segments and gets harder with depth. No external curriculum stages to stall on; the agent always sees easy goals first, so reward signal never dies. |
| Segment generation | **Approach A — template library**: segments are composed from existing, already-solvable completion chunks. Solvable by construction. |
| Completion counting | **x-boundary crossing** (see §7), not a dedicated gym-goal collision handler. |
| Goal reward | Preserved and **repeatable**: a flat `GYM_SEGMENT_BONUS` banked per cleared segment (default calibrated to the campaign goal-bonus magnitude, `≈ GOAL_MULT × typical_segment_width`). |
| Granted abilities | Configurable; **default `{DOUBLE_JUMP}`**. Templates declare the minimum abilities they need; the sampler only emits templates solvable under the granted set. |

## 4. Two grounding facts from the codebase

These shaped the design and must be respected by the implementation:

1. **`reached_goal` is terminal today.** `on_goal` (collision.py) sets
   `player.reached_goal = True` and calls `world.complete_level()`; both
   `evaluate` and `evaluate_infinite` (trainer.py) `break` when
   `player.reached_goal` is true. The gym must **not** terminate on a goal — see
   §7.

2. **Chunk `build()` signatures are inconsistent.** `GoalChunk.build` and
   `BoxLavaGap.build` accept `base_y`; `KeyChunk.build` and `DoorChunk.build`
   take *no* `base_y` and pin to the `GROUND_Y` module constant. Therefore the
   gym **cannot** reuse `TerrainStream.materialize_chunk` (which always passes
   `base_y=`). The gym stays on a **flat `GROUND_Y` baseline** and each template
   calls each chunk's `build()` with that chunk's own correct signature.

## 5. Components

**New files**
- `src/blueball/levels/segments.py` — segment **templates** + `SegmentSampler`.
- `src/blueball/levels/segment_stream.py` — `SegmentStream`.
- `train_completion_gym.py` — CLI (mirrors `train_infinite.py`).

**Reused unchanged**
- Entities/chunks: `Key`, `Door`, `Goal`, `PushableBox`, `Lava`, `Flat`, and the
  `key` / `door` / `goal` / `box_lava_gap` / `pushable_box` chunks.
- `ai/ftnn.py`, `ai/ga.py`, `ai/genome.py`, `ai/observation.py`,
  `ai/persistence.py`, `world.py`, `collision.py`, all entity classes.

**Touched minimally**
- `ai/episodes.py` — add `kind="gym"` handling to `EpisodeSpec` dispatch.
- `ai/trainer.py` — add `evaluate_gym(...)`; route `kind=="gym"` in
  `evaluate_episodes`.
- `ai/fitness.py` — add a `segments_cleared` term to `FitnessInputs` /
  `fitness()`, defaulting to 0 so all existing callers are numerically
  unchanged.
- `config.py` — add `GYM_*` tunables.

## 6. Segment templates & solvability

A **segment** is a self-contained, solvable, flat unit that ends in a `goal`,
built from existing chunks with short `Flat` footing between them. Solvability is
guaranteed *by construction*:
- Each constituent chunk is already hand-verified solvable.
- Ordering enforces dependencies: a `key` is always placed before the `door`
  that requires it; a locked door is solid geometry (plus the ceiling wall the
  `door` chunk adds) so the player **cannot** pass without first collecting the
  key. Crossing the segment therefore *implies* solving it.

**Template interface** (in `segments.py`): each template is an object/function
that, given `(world, x_offset)`, builds its chunks left-to-right on the
`GROUND_Y` baseline and returns:
- `width` — total segment width in px,
- the set of added shapes/bodies/entities/constraints (for culling; gathered via
  the same pre/post space-diff trick `TerrainStream.materialize_chunk` uses),
- `tier` (difficulty) and `min_abilities` (frozenset of `Ability`).

**Tiers** (difficulty knob, drives the ramp):
- **Tier 0** — `flat → goal`. Pure "run to the goal."
- **Tier 1** — `key(id=0) → flat → door(id=0) → goal`. Collect-then-unlock.
- **Tier 2** — `box_lava_gap → goal`, plus a pushable-box step variant. Box
  manipulation.
- **Tier 3** — combos, e.g. `key→door → box_lava_gap → goal`.

Box-lava pits use ~20-24 tiles (matching campaign `maze.json`'s `pit_tiles=24`) so a granted double-jump cannot vault them without the box — verified by `test_boxlava_pit_requires_the_box_not_vaultable`.

Each template declares `min_abilities`. The sampler filters to templates whose
`min_abilities ⊆ granted_abilities`. Default granted set is `{DOUBLE_JUMP}`
(configurable via CLI) — this makes the single-vs-double-jump assumption
explicit, the exact class of bug that broke box-lava training before.

Key IDs within a single segment start at 0 and increment per key used; because
`keys_held` is cleared at every segment boundary (§7), IDs are free to be reused
by the next segment.

## 7. SegmentStream + the two runtime details

`SegmentStream` (sibling of `TerrainStream`) materializes segments ahead of the
ball and culls those behind it, exposing the **ordered list of segment-end x
positions** (boundaries).

- **Construction** lays a guaranteed `Flat` at `x=0` (footing for spawn) then
  builds an initial handful of segments.
- **`maintain(player_x)`** builds ahead to keep `load_ahead` px materialized and
  culls segments fully behind `player_x - load_behind`, removing their shapes,
  bodies, entities, and constraints (same teardown loop as `TerrainStream`).
- The shared "diff the space before/after a build, record what was added, remove
  it on cull" logic will be **extracted into a small shared helper** reused by
  both streamers so they can't drift. (Fallback if we prefer not to touch the
  Infinite Run path at all: replicate the ~15 lines in `SegmentStream`.)

**Detail 1 — count-and-continue (no termination on goal).** `evaluate_gym` does
**not** read `reached_goal` / `level_complete` for termination. Goal entities are
still spawned, purely so the network *perceives* them (rays read a goal as +1
reward — important for campaign transfer). Completion is counted by **x-boundary
crossing**: when the player's running `max_x` passes a segment's end-x, that
segment is counted as cleared and banks `GYM_SEGMENT_BONUS`. Because a locked
door cannot be passed without its key (and the box-lava gap cannot be crossed
without the box step), crossing the boundary is equivalent to solving the
segment. The goal reward is thus preserved and made *repeatable* — it is just
delivered through the `segments_cleared` counter instead of the terminal
`reached_goal` flag.

**Detail 2 — key scoping across the chain.** `keys_held` is an 8-bit field; a
stale set bit would let a later reused-id door open "for free," breaking
solvability. So the gym **clears `keys_held` to 0 at each segment boundary**. To
avoid losing key reward when clearing, `evaluate_gym` tracks **cumulative** keys:
each tick it reads `popcount(keys_held)`, accumulates any *increase* over the
previous tick, and resets its per-tick baseline to 0 when it clears the field at
a boundary. Fitness is fed this cumulative count, not the live popcount.

## 8. Fitness

Extend `FitnessInputs` with `segments_cleared: int = 0` and add one term to
`fitness()`:

```
fitness = progress_x
        + 100.0 * keys_collected            # cumulative in gym; popcount elsewhere
        +  50.0 * collectibles
        + GOAL_MULT * level_width * reached_goal      # unchanged; 0 in gym (level_width=0)
        + GYM_SEGMENT_BONUS * segments_cleared         # NEW; 0 for non-gym callers
        -   0.01 * steps_taken
        - 200.0 * died
```

- For non-gym callers `segments_cleared` defaults to 0, so their fitness is
  numerically unchanged.
- In the gym, `reached_goal` is never used as a terminal/credit signal and
  `level_width` is 0, so the campaign goal term vanishes; the
  `GYM_SEGMENT_BONUS * segments_cleared` term carries the goal reward.
- `GYM_SEGMENT_BONUS` is a flat per-segment constant (keeps `segments_cleared`
  a clean int), defaulting to roughly `GOAL_MULT × typical_segment_width` so
  reward-per-completion is in the same range as the campaign goal bonus, aiding
  transfer. Tunable via `config`. (If per-segment-width weighting proves
  important during tuning, we can later swap the int count for an
  evaluator-accumulated float bonus — out of scope for v1.)

## 9. Episode integration

`EpisodeSpec` already carries `kind`, `seed`, `world_seed`, `max_steps`, `norm`.
Add `kind="gym"`, where `seed` is the **gym chain seed**. `evaluate_episodes`
gains a branch:

```python
elif ep.kind == "gym":
    _, raw = evaluate_gym((idx, genome, ep.seed, ep.world_seed, ep.max_steps, granted_abilities))
```

`evaluate_gym` mirrors `evaluate_infinite`: build `World`, register collisions,
create `SegmentStream(world, seed, granted_abilities)`, spawn the player with
`abilities=granted_abilities`, loop `maintain → substep` up to `max_steps`,
tracking `max_x`, `segments_cleared` (boundary crossings), and cumulative keys;
never break on goal; break on death. Return gym fitness.

## 10. Generalization, CLI, persistence, determinism

- **Multi-seed by default.** Train each genome across many gym seeds (the
  multi-seed Infinite result showed traversal generalizes at ~32 seeds). Each
  seed is a different chain, so the net learns mechanics, not one layout. Use the
  existing multi-episode aggregation (`mean - lam*std`).
- **CLI** `train_completion_gym.py`: `--pop`, `--gens`, `--num-seeds` /
  `--seeds`, `--workers`, `--abilities`, `--max-steps`. Saves to
  `genomes/gym_w<seed>_<timestamp>/` via existing `TrainingRunWriter`.
- **Determinism.** `SegmentSampler` seeded with `random.Random(seed)` (like
  `ChunkSampler`); physics via `world.substep()`. Same seeds → byte-identical
  `best_genome`.
- **`max_steps`** default is **higher than Infinite Run's**, since a chain with
  puzzles covers less x per step (collecting keys, pushing boxes, waiting).

## 11. Testing

- **Templates:** each template builds without error and is solvable — assert a
  scripted/known-good action sequence (or a geometric reachability check) reaches
  each goal; assert `min_abilities` is honored.
- **`SegmentSampler`:** deterministic for a fixed seed; tier rises with depth;
  filters by granted abilities; no back-to-back duplicate template.
- **`SegmentStream`:** materializes ahead, culls behind (no shape/body/entity
  leak — assert space counts return to baseline after a cull), boundary list is
  correct and ordered.
- **`evaluate_gym`:** counts a clear exactly when `max_x` crosses a boundary;
  cumulative-key tracking survives the per-boundary `keys_held` clear; never
  terminates early on a goal; terminates on death and at `max_steps`.
- **Integration:** a tiny gym run (small pop/gens, fixed seeds) is reproducible
  across two runs (identical `best_genome`).

## 12. Open knobs to tune later

Ramp rate, tier composition / weights, `GYM_SEGMENT_BONUS_PER`, granted ability
set, seeds-per-genome, `max_steps`, `load_ahead`/`load_behind`, anti-repeat
sigma.

## 13. File-by-file change list

| File | Change |
|------|--------|
| `src/blueball/levels/segments.py` | **New** — templates + `SegmentSampler`. |
| `src/blueball/levels/segment_stream.py` | **New** — `SegmentStream`. |
| `train_completion_gym.py` | **New** — CLI. |
| `src/blueball/ai/trainer.py` | Add `evaluate_gym`; route `kind=="gym"` in `evaluate_episodes`. |
| `src/blueball/ai/episodes.py` | `kind="gym"` dispatch; (carry `granted_abilities` into the gym episode path). |
| `src/blueball/ai/fitness.py` | Add `segments_cleared` to `FitnessInputs` + one fitness term. |
| `src/blueball/config.py` | Add `GYM_*` tunables (ramp, segment bonus, defaults). |
| `src/blueball/levels/streaming.py` | (Optional) extract the diff-and-cull helper for reuse. |
| `tests/` | New tests per §11. |
