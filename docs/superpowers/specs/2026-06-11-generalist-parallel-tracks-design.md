# Generalist Acceleration — Parallel Tracks (Coordination Spec)

**Date:** 2026-06-11
**Goal:** Reach, ASAP, a single trained genome that **clears all 5 static levels** *and* **moves competently on Infinite Run** (i.e. it genuinely learned to move, not memorize). This spec decomposes that goal into independent work streams that run concurrently in separate worktrees without colliding, and defines the coordination contract that makes the parallelism safe.

This is a **coordination spec**, not a feature spec. Each track gets its own plan (and, where flagged, its own brainstorm) in its own session. The deliverable here is the decomposition, the base-branch/file-ownership contract, and per-track acceptance criteria.

## Background (why these tracks)

From the multi-episode training thread (see memory `project_multi_episode_training`):

- The 510-param **stateless FTNN** (35 inputs → 12 hidden tanh → 6 actions) **can** learn general movement — a 32-seed Infinite run generalized to held-out seeds with only a ~7% drop (1697 → 1570 px). Traversal genetics are fine; the 32-seed run was *undertrained* (gens=80, best/mean still climbing).
- **Completion mechanics** (keys → doors → goal, box-lava crossing) were in **no** training distribution, so on the 5 fixed levels they could only be memorized. The generalist completed **0/5** (≈0.10–0.22 normalized), stalling on maze at x≈1390 (~31% in) — upstream traversal, before even reaching the box-lava.
- Two untested hypotheses for the 0/5 generalist remain:
  - **H2 (distribution):** it was never trained on a completion distribution. Fix the *recipe*, keep the 510-param net.
  - **H1 (capacity):** one stateless 510-param net cannot hold 5 disparate levels at once. Need a bigger / recurrent / level-conditioned controller.
  - **Decision:** pursue **both, in parallel.**
- The **completion gym** (the procedural, solvable-by-construction completion distribution) is **finished** (built by a separate session on `feature/completion-gym`, completed 2026-06-11; full suite 501 tests green). It provides `train gym`, `SegmentSampler`, `SegmentStream`, `evaluate_gym`, `gym_episodes`, `GYM_SEGMENT_BONUS`, and a **7-template segment library**: `GoalSegment`, `KeyDoorGoalSegment`, `BoxLavaSegment`, `KeyDoorBoxLavaSegment`, the box-difficulty curriculum (`BoxStepSegment`, `BoxLeapSegment`), and `BoostGapSegment`. The completion distribution Track B needs is ready *now*.
- **Visibility is broken:** there is no tool to watch a saved `.npy` genome play. Every result so far is a fitness number with no way to *see* the behaviour. This taxes every track.

## Branch topology (verified 2026-06-11)

- **`master`** — baseline. No gym, no double-jump.
- **`feature/box-push-shaping`** — PR#3, 15 commits ahead of master, 432 tests. Has the **double-jump / `LevelMeta.starting_abilities` foundation** + box-push shaping + maze curriculum. **No gym.** Complete, unmerged.
- **`feature/completion-gym`** (current branch) — the **only** branch with the **gym** *and* double-jump, and a **strict superset of `master`** (it contains all of master's visual overhaul). **Finished and green** (501 tests; box curriculum complete; the formerly-expected-red `test_boxlava_random_varies_pit_width` was removed as planned). **`completion-gym` → `master` is a clean fast-forward** — master ⊆ completion-gym, zero conflicts.

## Coordination strategy — fast-forward the finished gym to `master`, then fan out (chosen)

The gym is finished and green, so the foundation problem is now solved cleanly. **`completion-gym` → `master` is a fast-forward** (master ⊆ completion-gym), so landing it carries the visual overhaul + double-jump + the complete gym onto `master` with **zero conflicts**.

**Recommended Step 0 — fast-forward `completion-gym` into `master`.** One clean, reversible operation; afterwards `master` is the complete, stable foundation every track wants. (Merging the *stale* `box-push-shaping` is moot — it predates the visual overhaul and has no gym.)

The decisive constraint is unchanged — **every training path grants double-jump.** The box-lava plateau (fitness 1020.5, never reached goal) vs. the cracked run (10163, gen 18) was *caused* by spawning single-jump players on a level that assumes double-jump: a net trained without abilities yields a **false capacity verdict** (C), and a mover that never double-jumps hasn't learned to move (D). `master` (post-FF) carries the `starting_abilities` machinery, so all four tracks inherit it.

**Global invariant:** every training path in this effort grants double-jump. It is unlocked early (tutorial) and carried by the save, so the generalist always has it; no training distribution should ever spawn a single-jump-only player.

**All four tracks branch off `master` (post-FF).** They are byte-identical to branching off the `completion-gym` *tip*, so if you haven't run the FF yet, branch off `completion-gym` tip and the briefs still work verbatim. Each track is its own branch; the file-ownership contract below keeps them conflict-free from one another. Because the gym is done, there is **no "rebase when the gym lands"** — it has landed; tracks integrate back to `master` independently as they finish.

## Dependency graph

```
A (watch-best)  ──┐ independent, start now, fast        → unblocks visibility for ALL
D (movement run)──┤ independent, background compute     → banks warm-start genome ─┐
                  │                                                                 ▼
B (recipe) ───────┴ start now (gym is DONE) ──────── KEYSTONE: the combined harness
                                                          ▲
C (controller) ── code now ──────────────────────────────┘ early-validate on maze/gym (double-jump); definitive A/B test through B
```

All four branch off `master` (post-FF; ≡ `completion-gym` tip). **B is the keystone:** both H1 and H2 are tested *through* B's combined-episode harness (H2 = run B with the 510-param net; H1 = run B with C's alternate net). C can also self-validate early on the existing `train maze` / `train gym` paths (which grant double-jump) before B's harness exists. A and D are support.

## File-ownership contract (conflict avoidance)

| File | Owner | Notes |
|---|---|---|
| `scenes/playback.py` (new) | A | new file, no contention |
| `cli.py` | A, B (additive blocks only) | each adds a distinct subcommand; coordinate only at final integration |
| `ai/episodes.py` | B | new `mixed_episodes` constructor |
| `ai/trainer.py` | B, D | B: warm-start seeding of initial population; D: opt-in ability grant in `evaluate_infinite` — distinct functions, additive |
| `ai/fitness.py` | — (frozen) | no track modifies it; the gym (now in the base) owns the `segments_cleared` term |
| `ai/ftnn.py`, `ai/observation.py`, `agent.py` | C | the net — nobody else touches it |
| `config.py` | B, C (distinct keys) | B adds mix weights; C adds a net-version flag; different keys, additive |
| `levels/segments.py`, `sampler.py`, `segment_stream.py` | gym (done — frozen) | **no track touches these** |

The gym is already in the base, so every track only *adds* to it. The only files shared *between tracks* are `cli.py` (A, B add distinct subcommands), `config.py` (B, C add distinct keys), and `trainer.py` (B, D edit distinct functions) — all **additive, non-overlapping blocks**, so cross-track conflict is near-zero.

---

## Track A — watch-best playback tool

**Base:** `master` (post-FF; ≡ `completion-gym` tip). **Conflict surface:** `cli.py` only (additive). **Effort:** low. **Can execute directly** (well-specified; no brainstorm needed).

**What:** `python main.py watch-best <run-dir|genome.npy> [--level NAME | --infinite SEED | --gym SEED]` opens a pygame window, loads the genome into `FTNNAgent`, plays it visually with a HUD (live fitness, x, deaths, reached_goal). Defaults: a run-dir resolves to its `final_best.npy`. Grants the level's `LevelMeta.starting_abilities` (so maze replays with double-jump faithfully).

**Owns:** new `src/blueball/scenes/playback.py`; additive `watch-best` subcommand in `cli.py`.
**Reuses:** `FTNNAgent` (`agent.py`), `Renderer`, the `PlayScene` loop pattern, level loader, `ChunkSampler`.

**Acceptance:**
- Loads `genomes/<run>/final_best.npy` and plays tutorial_hill, maze, and an infinite seed in a window.
- HUD fitness tracks the headless eval of the same genome (faithful, within animation-cutoff tolerance).
- A non-interactive smoke path (no window) exercises the genome-load + scene-construct so it is testable in CI.

---

## Track B — integration recipe (KEYSTONE)

**Base:** `master` (post-FF; ≡ `completion-gym` tip). **Conflict surface:** `episodes.py`/`trainer.py`/`cli.py`/`config.py` — all additive; only `trainer.py`/`cli.py`/`config.py` are shared with other tracks (distinct functions/keys). **Effort:** medium. **Keystone — write a plan first (writing-plans → subagent-driven).**

**What:** a combined multi-episode generalist trainer exposed as `python main.py train generalist`. One selection objective mixes **infinite + the 5 static levels + gym** episodes:
- double-jump granted on the static + gym episodes (via `starting_abilities` / gym `abilities`);
- per-episode normalization (existing `compute_level_par`) so big levels don't dominate;
- worst-case aggregation (`aggregate="min"`) as the default objective, with `mean_std` selectable;
- optional **warm-start genome** (`--init <genome.npy>`, from Track D) seeding the initial population;
- `run.json` records the full mixed episode set and per-episode-kind scores.

**Owns:** `ai/episodes.py` (new `mixed_episodes(infinite_seeds, level_names, gym_seeds, world_seed, max_steps, abilities)` constructor), `ai/trainer.py` (thread an optional warm-start genome into `train(...)`'s initial population), `cli.py` (`train generalist` subcommand), `config.py` (mix counts / weights).
**Reuses:** `evaluate_episodes`, `gym_episodes` / `static_episodes` / `infinite_episodes`, `aggregate_fitness`, persistence (`run_dir_name` gains a `gen*` variant).

**Sequencing — the gym is finished, so B goes end-to-end now.** B consumes the gym's stable interface (`gym_episodes`, `evaluate_gym` / `evaluate_episodes(kind="gym")`, `SegmentSampler`, `SegmentStream`, `GYM_SEGMENT_BONUS`) and the full **7-template** pool (Goal, KeyDoorGoal, BoxLava, KeyDoorBoxLava, BoxStep, BoxLeap, BoostGap) — including the box-difficulty curriculum that makes the hard box-hop *learnable* (box-lava only cracked with staged shaping). The earlier build-now-then-rebase split is gone: B builds the harness **and** can run the definitive all-5-levels training as soon as the harness exists. No inherited red test (it was removed); no pending gym-core signature changes (the gym is done).

**Acceptance:**
- `train generalist` runs end-to-end (small pop/gens) and writes a genome + `run.json` with the mixed set and per-kind scores.
- Warm-start path verified: seeding from a known mover genome reproduces ≥ that mover's infinite score on gen 0.
- Single-mode backward-compat preserved (existing `train infinite` / `train levels` / `train gym` byte-identical).
- **Stretch (the real goal):** a full run beats the 0/5 baseline — completes the easy static levels (tutorial_hill, speed_run) while holding a non-trivial infinite distance.

---

## Track C — controller capacity (H1)

**Base:** `master` (post-FF; ≡ `completion-gym` tip). **Conflict surface:** the net files only — zero overlap. **Effort:** medium–high. **Needs its own brainstorm** (the net choice is a real fork).

**Double-jump is mandatory for C.** Every net C trains must be granted double-jump post-tutorial. The box-lava plateau (1020.5, never reached goal) vs. the cracked run (10163, gen 18) was caused purely by training single-jump players on a double-jump level — training C's net without abilities would reproduce that plateau and yield a **false capacity verdict**. The base (`master` post-FF) carries the `starting_abilities` machinery and the `train maze` (curriculum) / `train gym` paths, which grant double-jump, so C can self-validate early before B's harness exists.

**What:** an alternate controller, **quarantined behind a net-version config flag** so any `GENOME_SIZE` change does not break existing `.npy`, Track D's mover, or Track B's genomes (default stays the 510-param net until C produces a verdict). Candidate directions (to be chosen in C's brainstorm):
- bigger hidden layer (more capacity, same stateless shape — simplest, smallest risk);
- recurrent state via the existing `Agent.reset(world)` hook (memory across frames);
- level-conditioned observation input (give the net a level-identity signal — changes `INPUT_SIZE`/`GENOME_SIZE`).

**Owns:** `ai/ftnn.py`, `ai/observation.py`, `agent.py` (`FTNNAgent`), `config.py` (net-version flag, distinct key from B's).
**Validation, two stages:** (1) *early signal, independent of B* — train the alternate net on the existing `train maze` (curriculum) / `train gym` paths (both grant double-jump) to confirm it learns at least as well as the 510-param net on a hard single objective; (2) *definitive cross-level verdict, through B* — run B's combined recipe with the new net vs. the 510-param net on the worst-case (`min`) objective. C's *code* lands independently; only the stage-2 verdict waits for B's harness.

**Acceptance:**
- The alternate net trains through the existing trainer paths behind the version flag; default 510-param behaviour is unchanged (anchored by an existing-genome test).
- A documented head-to-head on B's recipe: does bigger/recurrent/conditioned beat the 510-param net on the worst-case (`min`) objective? Verdict (adopt / reject) written up.

---

## Track D — movement-floor run (compute)

**Base:** `master` (post-FF; ≡ `completion-gym` tip). **Conflict surface:** a single additive flag in `trainer.py` (distinct function from B's warm-start change). **Effort:** low (mostly compute). **Background job; can start immediately.**

**What:** a long/large multi-seed Infinite run (the 32-seed run plateaued *undertrained* at gen 80). Scale up seeds (e.g. 32–64) and generations (e.g. 300–500) to bank a strong mover genome — used as Track B's warm-start and as the measured movement floor.

**Double-jump is part of the movement skill (not optional).** `evaluate_infinite` currently grants no abilities (no level meta). D grants double-jump in the infinite run because *learning to move* includes learning to double-jump — a mover that only runs and single-jumps has an impoverished repertoire and would warm-start B with a dead abilities-input bit. **Necessary but not sufficient:** the infinite terrain must actually *demand* double-jump (wide `gap` / `box_lava_gap` chunks the difficulty ramp escalates into), or the agent will ignore the ability it was handed. D's brief must confirm the sampler's harder tiers genuinely require double-jump; if a single jump still clears them, raise the gap widths / ramp so they don't. The code change itself is a small additive flag threading the granted-ability set into `evaluate_infinite`.

**Owns:** an additive `--abilities` (or equivalent) flag threading a granted-ability set into `evaluate_infinite` (existing `train infinite` otherwise). **D does not edit `sampler.py`** (gym-owned) — if the harder tiers don't require double-jump, D first tunes via existing `ChunkSampler` params (`ramp_per_chunk`, `sigma`, `target_chunks`); only if chunk *geometry* needs to change does D file a request to the gym session rather than touching their file.

**Acceptance:**
- A banked `final_best.npy` (trained with double-jump granted) whose **held-out-seed** mean distance beats the current ~1570 px floor.
- **Evidence the mover actually *uses* double-jump** — it clears a `gap`/`box_lava_gap` that a single jump can't — confirming the skill was learned, not merely granted (watch-best / Track A is the natural check).
- Handed to Track B as `--init`, and the held-out number recorded as the movement baseline.

---

## Open questions / deferred

- **Net choice for C** — resolved in C's own brainstorm. Start with the bigger-hidden-layer option (lowest risk) unless level-conditioning is judged necessary.
- **Mix ratios for B** — how many infinite vs. static vs. gym episodes per evaluation, and whether to ramp the gym share over generations. Tunable; start with an even-ish split and iterate.
- **Genome compatibility across net versions** — C must define how `.npy` files declare their net version so playback (A) and the recipe (B) load the right topology.

## Staffing

4 worktrees, all branched off `master` (post-FF; ≡ `completion-gym` tip). This session produces this spec plus a self-contained kickoff brief per track (`docs/superpowers/briefs/2026-06-11-track-{A,B,C,D}.md`); each track then runs in its own session/worktree. Tracks A and D are specified enough to execute directly (D adds its small double-jump flag, then runs as a background job); Track B (keystone) writes a plan first (writing-plans → subagent-driven); Track C runs its own brainstorm first (net choice). Because the gym is finished, each track integrates back to `master` independently as it completes — no shared rebase barrier.
