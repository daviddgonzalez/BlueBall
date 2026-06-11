# Track A — watch-best playback tool (session brief)

**Read first:** `docs/superpowers/specs/2026-06-11-generalist-parallel-tracks-design.md` (full context, coordination contract, why this matters). This brief is the kickoff for the session that owns Track A.

## One-line goal
Build a tool to **visually watch a saved `.npy` genome play** any level/seed — the missing visibility that taxes every other track.

## Setup
- Branch off `feature/completion-gym` (the finished gym; ≡ `master` once it's fast-forwarded). Use the using-git-worktrees skill / `EnterWorktree`; suggested branch `feature/watch-best`.
- Interpreter: `.venv/bin/python` (`python` is not on PATH). Run tests with `.venv/bin/python -m pytest -q`.

## What to build
`python main.py watch-best <run-dir|genome.npy> [--level NAME | --infinite SEED | --gym SEED]`:
- Loads the genome (`np.load`); a run-dir argument resolves to its `final_best.npy`.
- Constructs a scene with `FTNNAgent(genome)` as the player and runs the pygame loop with the `Renderer`.
- HUD overlay: live fitness, x-progress, deaths, `reached_goal`.
- **Grants the level's `LevelMeta.starting_abilities`** (so a maze/gym genome replays with double-jump faithfully — see the spec's double-jump invariant).
- Default target if none given: the static levels; `--infinite`/`--gym` select streamed modes.

## Files you OWN
- New `src/blueball/scenes/playback.py` (the playback scene; model it on `scenes/play.py`'s loop).
- Additive `watch-best` subcommand in `src/blueball/cli.py` (copy the pattern of `cmd_play`).

## Files you must NOT touch
- `levels/segments.py`, `sampler.py`, `segment_stream.py` (gym — frozen).
- The net files (`ftnn.py`, `observation.py`, `agent.py`) belong to Track C — **reuse `FTNNAgent`, don't modify it**.
- `episodes.py`, and the `train(...)`/`evaluate_*` functions (Track B/D).

## Reuse
`FTNNAgent` (`agent.py`), `Renderer` (`render/renderer.py`), the `PlayScene` loop pattern (`scenes/play.py`), the level loader, `ChunkSampler` (infinite), the gym `SegmentStream` (for `--gym`).

## Process
Execute directly with TDD (test-driven-development skill) — this is well-specified, no brainstorm needed.

## Acceptance criteria
- Loads `genomes/<run>/final_best.npy` and plays **tutorial_hill**, **maze**, and an **infinite seed** in a window.
- HUD fitness tracks the headless eval of the same genome (faithful within animation-cutoff tolerance).
- A **non-interactive smoke path** (no window — e.g. headless construct + N steps) exercises genome-load + scene-construct so it's CI-testable.

## Why this is high-leverage
Every result so far is a fitness number with no way to *see* behaviour. Track D's acceptance ("mover demonstrably *uses* double-jump") and B/C debugging all lean on this. Land it early.
