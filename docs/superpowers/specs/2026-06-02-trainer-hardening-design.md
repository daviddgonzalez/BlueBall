# Trainer Hardening — Design Spec

**Date:** 2026-06-02
**Branch:** `feature/ai-scaffolding`
**Status:** Design approved; ready for implementation plan.

## Problem

The GA trainer now works end-to-end (Observation adapter, headless Infinite
Run eval, reference seeds, genome persistence — all landed on
`feature/ai-scaffolding`). Before committing compute to a real, long training
run, the trainer has three weaknesses that would make those runs untrustworthy
or unwatchable:

1. **Float drift breaks cross-machine determinism.** `evaluate`/
   `evaluate_infinite` advance physics with `world.step(config.PHYS_DT)` once
   per iteration. `PHYS_DT = 1/120` is not exactly representable in IEEE 754,
   so `World`'s accumulator carries a residual that, over thousands of
   iterations, can cross `PHYS_DT` and fire a phantom extra substep — diverging
   trajectories across numpy/Python builds and CPU architectures. This bites
   far harder over a long Infinite Run than over a short static level.

2. **The parallel path is untested.** `train_infinite.py` defaults to
   `multiprocessing.Pool(...).imap` for evaluation, but nothing pins that the
   workers pickle correctly, that out-of-order results reorder, or that the
   Pool path produces the same `best_genome` as serial `map`. A silent flake
   here corrupts a multi-hour run.

3. **The visual trainer grades the wrong thing.** `TrainScene`
   (`train_main.py`) trains on a *static* level via `load_level`, while real
   training is on Infinite Run. Worse, it only puts `n_visible` (16) players in
   the World and assigns the rest `-inf` fitness, so the effective population is
   16, not 80 — selection pressure is badly diluted, and you cannot watch the
   population learn the actual reference course.

## Goals

- Bit-identical evaluation results across machines for a fixed seed triple
  (`ga_seed`, `world_seed`, `infinite_seed`), regardless of run length.
- The `multiprocessing.Pool` evaluation path is verified to match the serial
  path and to be safe to rely on for long runs.
- The visual `TrainScene` trains on Infinite Run with the full population
  evaluated (no `-inf` dilution), letting the user watch representative agents
  run the real reference course while authoritative fitness is computed off the
  render loop.

## Non-goals (deferred to later AI work)

- Loading a trained genome into the real game / "watch the best agent play"
  mode.
- Fitness-function shaping (death penalty, time, collectibles, exploration).
- Observation-encoding richness (semantic ray channel vs one-hot hit types,
  normalization tuning, ability/key bit widths).
- `ReplayAgent` + recording for Race mode.
- NEAT escape hatch if FTNN plateaus.

These are real follow-ups but out of scope here; this spec is strictly about
making the *trainer* trustworthy.

---

## WS1 — Deterministic fixed substep

### Current behavior

`World.step(frame_dt)` (`world.py`) accumulates `frame_dt` and runs zero or
more fixed `PHYS_DT` substeps:

```python
self._accumulator += frame_dt
while self._accumulator >= config.PHYS_DT and substeps < MAX_ACCUMULATED_STEPS:
    self.space.step(config.PHYS_DT)
    # ... entity updates ...
    self._accumulator -= config.PHYS_DT
    substeps += 1
```

The trainer calls `world.step(config.PHYS_DT)` once per iteration. Today that
fires exactly one substep, but the accumulator's float residual drifts and can
eventually fire two (or zero) in a single call.

### Change

Add a method that runs **exactly one** fixed substep with no accumulator:

```python
def substep(self) -> None:
    """Advance physics by exactly one fixed PHYS_DT substep, bypassing the
    real-time accumulator. Deterministic across hosts — N calls == N substeps,
    with no float residual. Used by the headless trainer; the live game uses
    step(frame_dt) for real-time pacing."""
    self.space.step(config.PHYS_DT)
    for entity in self.entities:
        update = getattr(entity, "update", None)
        if update is not None:
            update(config.PHYS_DT)
```

(The exact entity-update loop must mirror what `step()` does today — the plan
will factor the shared body so `step()` and `substep()` cannot diverge.)

`trainer.evaluate` and `trainer.evaluate_infinite` replace
`world.step(config.PHYS_DT)` with `world.substep()`.

`PlayScene` and `TrainScene`'s live display are unchanged — they keep
`step(frame_dt)`.

### Why trajectories don't change

Passing exactly `PHYS_DT` to `step()` already produces one substep per call, so
`substep()` yields the same physics sequence — it just removes the residual
that could later desync. Existing finiteness/determinism smoke tests stay
green; only exact cross-host reproducibility improves.

### Tests

- `substep()` runs exactly one `space.step` (e.g. assert a known body moves by
  the one-substep delta; or spy the substep count).
- `evaluate_infinite` is bit-identical across two calls at a **large**
  `max_steps` (≥ 2000), where drift would have surfaced.

### Interface summary

- New: `World.substep() -> None`.
- Changed: `trainer.evaluate`, `trainer.evaluate_infinite` loop bodies.

---

## WS2 — Multiprocessing.Pool integration test

No production change; this workstream is test coverage that WS1 makes
achievable (exact equality under Pool).

### Tests

- **Determinism under Pool:** `train(infinite_seed=S, ga_seed=G, world_seed=W,
  map_fn=Pool(2).imap)` produces a `best_genome` `array_equal` to the same call
  with serial `map`. Small pop/gens/steps for CI speed.
- **Pickling + reordering:** evaluating a handful of genomes via
  `Pool.imap` returns one `(idx, fitness)` per genome and, after the trainer's
  `sort(key=idx)`, matches the serial result element-for-element.
- The test must clean up the Pool (`close()`/`join()`), and skip gracefully if
  the platform can't fork/spawn (guard with a try/skip), so CI on constrained
  runners doesn't hang.

### Risk note

BLAS/threading could in principle perturb numpy results across processes, but
the FTNN is tiny pure-numpy matmuls; combined with WS1 there is no remaining
nondeterminism source. If the equality test ever flakes, that is the signal to
investigate (documented in the test).

---

## WS3 — TrainScene Infinite Run parity + async viz-only eval

### Shape

`TrainScene` splits into two independent paths:

- **Display path (cosmetic):** one `World` + `TerrainStream(infinite_seed)`,
  `n_visible` `FTNNAgent` players stepped live each frame for the user to
  watch. Terrain is maintained off the leading visible player; the camera
  follows the pack. This path determines nothing about selection.
- **Truth path (async, authoritative):** the full population is evaluated
  headlessly off the render loop via `pool.map_async(evaluate_infinite, ...)`.
  Its results drive fitness, elitism, and breeding.

Because all players share `PLAYER_GROUP` (no inter-player collision) and
terrain geometry is identical for a fixed seed, a visible player's live run
reproduces its headless fitness — the display is a faithful window onto the
same evaluation, not an approximation.

### Constructor

```python
TrainScene(
    screen,
    *,
    level_path=None,
    infinite_seed=None,         # exactly one of level_path / infinite_seed
    pop_size=config.TRAIN_POP_SIZE,
    n_visible=16,
    generations=config.TRAIN_GENERATIONS,
    max_steps=config.MAX_STEPS,
    ga_seed=config.GA_SEED,
    world_seed=config.DEFAULT_SEED,
    save_dir=None,              # optional persistence
    pool=None,                  # injectable; defaults to a real Pool
)
```

`train_main.py` constructs `TrainScene(screen, infinite_seed=config.INFINITE_RUN_SEED)`.

Static-level training (`level_path`) remains supported for parity with
`train()`, using `evaluate` instead of `evaluate_infinite` on the truth path
and `load_level` instead of `TerrainStream` on the display path.

### Per-generation control flow

1. **Gen start:** build the display World + TerrainStream (or load_level);
   instantiate `n_visible` players from the current population's first
   `n_visible` genomes; launch `result = pool.map_async(eval_fn, args)` for the
   full population.
2. **Each frame:** `terrain.maintain(leading_visible_x)`; `world.step(frame_dt)`;
   render; draw HUD (gen #, last completed best/mean, "evaluating…" while
   `not result.ready()`).
3. **Eval ready (`result.ready()`):** collect `(idx, fitness)`, sort by idx,
   update running best, append history, persist via `save_dir` if set, `breed()`
   the next population, increment gen, go to step 1. **Generation flips when the
   eval completes** (the display may be cut mid-animation — acceptable per
   design).
4. **Done (gen == generations):** stop launching evals; idle/exit.

### Injectable pool (testability)

`pool` is any object exposing `map_async(fn, iterable) -> AsyncResult`-like
with `.ready()` and `.get()`. Default: a real `multiprocessing.Pool`. Tests
inject a **synchronous stub** whose `map_async` runs immediately and whose
`ready()` returns `True`, so a TrainScene test can step a few frames and assert
a generation advanced, the full population was evaluated, and breeding occurred
— without spawning processes or depending on wall-clock timing.

### Lifecycle & errors

- Real Pool opened in `__init__` (worker count configurable), closed on scene
  exit; `terminate()` if an eval is still pending when the user leaves.
- Worker exceptions propagate on `.get()` and surface (don't silently hang the
  scene).

### Invisible-agent problem

Resolved: the entire population is evaluated on the truth path, so there is no
`-inf` dilution and no effective-pop shrinkage. `n_visible` only controls how
many agents are drawn.

### Tests

- TrainScene with `infinite_seed` constructs and steps without raising under
  `SDL_VIDEODRIVER=dummy` (real headless pygame fixture, synchronous pool stub).
- With the stub, after enough frames the scene advances a generation, breeds a
  new population of `pop_size`, and the truth path evaluated all `pop_size`
  genomes (not just `n_visible`).
- Construction rejects neither/both of `level_path`/`infinite_seed`.

### Interface summary

- Changed: `TrainScene.__init__` signature + internals (display/truth split).
- Changed: `train_main.py` entry to use `infinite_seed`.
- New: synchronous pool stub in tests (test-only).

---

## Testing strategy (all workstreams)

Run under `SDL_VIDEODRIVER=dummy` with the repo venv. New/updated tests:

- `tests/test_world.py` (or existing): `substep()` exactness.
- `tests/test_ai_smoke.py`: long-`max_steps` `evaluate_infinite` determinism;
  Pool determinism vs serial; Pool pickling/reordering.
- `tests/test_train_scene.py` (or in `test_ai_smoke.py`): Infinite Run
  construction, generation advance via synchronous pool stub, full-pop eval,
  constructor validation.

The full suite (currently 337 green) must stay green.

## Risks / open questions

- **Pool inside a pygame app:** opening a `multiprocessing.Pool` from a running
  pygame process is heavier than the headless script; fork safety is fine
  because `ai/` is pygame-free (workers never import SDL). The injectable-pool
  seam keeps tests off real processes.
- **Display vs eval pacing:** the headless eval may finish well before the live
  animation would naturally end; flipping on eval-ready (per design) keeps the
  loop simple but means the on-screen run is often cut short. Acceptable; HUD
  communicates the authoritative numbers.
- **Entity-update parity between `step()` and `substep()`:** the plan must
  factor the shared per-substep body so the two cannot drift.
