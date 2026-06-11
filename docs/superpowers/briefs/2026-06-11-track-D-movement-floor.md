# Track D — movement-floor run (session brief)

**Read first:** `docs/superpowers/specs/2026-06-11-generalist-parallel-tracks-design.md`. This track banks a strong **mover genome** (Track B's warm-start) and measures the movement floor.

## One-line goal
Run a long/large multi-seed Infinite training — **with double-jump granted** — to produce a genome that genuinely learned to move, and hand it to Track B as `--init`.

## Setup
- Branch off `feature/completion-gym` (finished gym; ≡ `master` once fast-forwarded). using-git-worktrees / `EnterWorktree`; suggested branch `feature/movement-floor`.
- Interpreter `.venv/bin/python`. Long runs: use `run_in_background` and poll.

## Why scale up
The prior 32-seed Infinite run plateaued **undertrained** (gens=80, best/mean still climbing; held-out ~1570 px = a floor, not a ceiling). Scale seeds (≈32–64) and generations (≈300–500) and let it converge.

## ⚠️ Double-jump is part of the movement skill (not optional)
`evaluate_infinite` currently grants **no** abilities (no level meta). You must grant double-jump in the infinite run — *learning to move* includes learning to double-jump; a mover that only runs and single-jumps has an impoverished repertoire and would warm-start B with a dead abilities-input bit.

**Necessary but not sufficient:** the infinite terrain must actually *demand* double-jump (wide `gap` / `box_lava_gap` chunks the difficulty ramp escalates into), or the agent will ignore the ability. **Confirm the sampler's harder tiers genuinely require double-jump.** If a single jump still clears them, first tune via existing `ChunkSampler` params (`ramp_per_chunk`, `sigma`, `target_chunks`). **Do NOT edit `sampler.py`** (gym-owned) — if chunk *geometry* must change, file a request to the gym owner instead.

## Files you OWN
An additive `--abilities` (or equivalent) flag threading a granted-ability set into `evaluate_infinite` (`ai/trainer.py`). Keep your edit in `evaluate_infinite` — Track B edits `train(...)` in the same file, so distinct functions = no collision. Plumb the flag through the `train infinite` CLI subcommand.

## Files you must NOT touch
`sampler.py`/`segments.py`/`segment_stream.py` (gym — frozen), the net files (Track C), `episodes.py` and `train(...)`'s warm-start path (Track B).

## Acceptance criteria
- A banked `final_best.npy` (trained **with double-jump granted**) whose **held-out-seed** mean distance beats the current ~1570 px floor.
- **Evidence the mover actually *uses* double-jump** — clears a `gap`/`box_lava_gap` a single jump can't (watch-best / Track A is the natural check).
- Handed to Track B as `--init`; record the held-out number as the movement baseline.

## Process
Small additive flag (TDD), then a background compute run. Can start immediately.
