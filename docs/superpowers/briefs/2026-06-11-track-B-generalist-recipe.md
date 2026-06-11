# Track B — integration recipe (KEYSTONE) (session brief)

**Read first:** `docs/superpowers/specs/2026-06-11-generalist-parallel-tracks-design.md`. This is the keystone track — both hypotheses (H1 capacity, H2 distribution) are tested *through* this harness.

## One-line goal
A **combined multi-episode generalist trainer** that mixes infinite + the 5 static levels + the (finished) completion gym into one selection objective — the harness that actually produces "clears all 5 levels + moves well on Infinite Run."

## Setup
- Branch off `feature/completion-gym` (finished gym; ≡ `master` once fast-forwarded). using-git-worktrees / `EnterWorktree`; suggested branch `feature/generalist-recipe`.
- Interpreter `.venv/bin/python`. The full suite is green at 501 tests on this base.

## What to build
`python main.py train generalist` — one objective mixing **infinite + 5 static levels + gym** episodes:
- **Double-jump granted** on static + gym episodes (via `LevelMeta.starting_abilities` / gym `abilities`) — see the spec's global invariant.
- Per-episode **normalization** with the existing `compute_level_par` so large levels don't dominate.
- **Worst-case `min` aggregation** as the default objective (`aggregate="min"`), `mean_std` selectable.
- Optional **warm-start genome** `--init <genome.npy>` (from Track D) seeding the initial population.
- `run.json` records the full mixed episode set + per-episode-kind scores.

## Files you OWN
- `ai/episodes.py` — new `mixed_episodes(infinite_seeds, level_names, gym_seeds, world_seed, max_steps, abilities)` constructor (compose the existing `infinite_episodes` / `static_episodes` / `gym_episodes`).
- `ai/trainer.py` — thread an optional warm-start genome into `train(...)`'s initial population (a **new/distinct** code path; Track D edits `evaluate_infinite` in the same file — keep your change in `train(...)` to avoid collision).
- `src/blueball/cli.py` — `train generalist` subcommand (additive; the existing `train` group already has `infinite`/`levels`/`maze`/`gym`).
- `config.py` — mix counts / weights (distinct keys; Track C adds a net-version key — don't collide).

## Files you must NOT touch
- `levels/segments.py`, `sampler.py`, `segment_stream.py` (gym — frozen).
- The net files `ftnn.py`/`observation.py`/`agent.py` (Track C). Use whatever net the base provides.

## Reuse
`evaluate_episodes`, `gym_episodes`/`static_episodes`/`infinite_episodes`, `aggregate_fitness` (`"min"`/`"mean_std"`), persistence (`run_dir_name` — add a `gen*` variant), the 7-template gym pool (Goal, KeyDoorGoal, BoxLava, KeyDoorBoxLava, BoxStep, BoxLeap, BoostGap).

## Process
**Write a plan first** (writing-plans skill) → execute via subagent-driven-development. This track has real design surface (mix ratios, warm-start plumbing, run.json schema) — plan before coding.

## Sequencing note (gym is DONE)
No build-now-then-rebase split anymore. Build the harness **and** run the definitive all-5-levels training as soon as it exists. The full box-difficulty curriculum (which makes the hard box-hop *learnable*) is already in the gym pool.

## Acceptance criteria
- `train generalist` runs end-to-end (small pop/gens) → writes a genome + `run.json` with the mixed set + per-kind scores.
- Warm-start verified: seeding from a known mover reproduces ≥ that mover's infinite score at gen 0.
- Backward-compat: existing `train infinite` / `train levels` / `train gym` byte-identical.
- **Stretch (the real goal):** a full run beats the 0/5 baseline — completes the easy static levels (tutorial_hill, speed_run) while holding a non-trivial infinite distance.

## Open knobs (tune, don't block)
Mix ratios (how many infinite vs. static vs. gym episodes per eval; whether to ramp the gym share over generations) — start even-ish, iterate.
