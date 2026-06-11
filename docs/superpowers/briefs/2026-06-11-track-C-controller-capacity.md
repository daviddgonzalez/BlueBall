# Track C — controller capacity / H1 (session brief)

**Read first:** `docs/superpowers/specs/2026-06-11-generalist-parallel-tracks-design.md`. This track tests **H1: is the stateless 510-param FTNN too small to hold 5 disparate levels at once?**

## One-line goal
Build an **alternate controller** (bigger / recurrent / level-conditioned), quarantined behind a net-version flag, and run a head-to-head vs. the 510-param net to settle the capacity question.

## Setup
- Branch off `feature/completion-gym` (finished gym; ≡ `master` once fast-forwarded). using-git-worktrees / `EnterWorktree`; suggested branch `feature/controller-capacity`.
- Interpreter `.venv/bin/python`.

## ⚠️ Double-jump is mandatory (don't repeat the plateau bug)
Every net you train **must be granted double-jump**. The box-lava plateau (fitness 1020.5, never reached goal) → cracked (10163, gen 18) was caused *purely* by training single-jump players on a double-jump level. Training your alternate net without abilities would reproduce that plateau and yield a **false capacity verdict**. The base carries the `starting_abilities` machinery and the `train maze` (curriculum) / `train gym` paths, which grant double-jump — use them.

## Process
**Run your own brainstorm first** (brainstorming skill) — the net choice is a real design fork. Candidate directions (pick in the brainstorm):
- **Bigger hidden layer** — more capacity, same stateless shape. Simplest, lowest risk. *Recommended starting point.*
- **Recurrent state** via the existing `Agent.reset(world)` hook — memory across frames.
- **Level-conditioned observation** — give the net a level-identity signal (changes `INPUT_SIZE`/`GENOME_SIZE`).

Then writing-plans → implement.

## Hard constraint — quarantine the net change
Any `GENOME_SIZE` change **must be gated behind a net-version config flag**, default = the current 510-param net, so existing `.npy` files, Track D's mover, and Track B's genomes keep loading. Define how a `.npy` declares its net version so playback (Track A) and the recipe (Track B) load the right topology (this is an open question in the spec — you own resolving it).

## Files you OWN
`ai/ftnn.py`, `ai/observation.py`, `agent.py` (`FTNNAgent`), `config.py` (net-version flag — a **distinct key** from Track B's mix-weight keys).

## Files you must NOT touch
Everything else — the net files are yours alone and no other track touches them, so keep your blast radius there. Don't edit `trainer.py`/`episodes.py` (B/D) or the gym files.

## Validation — two stages
1. **Early signal (independent of B):** train the alternate net on `train maze` (curriculum) / `train gym` (both grant double-jump); confirm it learns ≥ the 510-param net on a hard single objective.
2. **Definitive verdict (through B):** run Track B's combined recipe with the new net vs. the 510-param net on the worst-case (`min`) objective.

## Acceptance criteria
- Alternate net trains through the existing trainer paths behind the version flag; **default 510-param behaviour unchanged** (anchor with an existing-genome test).
- A documented head-to-head: does bigger/recurrent/conditioned beat the 510-param net on the `min` objective? **Write up the verdict (adopt / reject).**
