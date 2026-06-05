# Multi-Episode Training (multi-seed + static-level generalist) — Design Spec

**Date:** 2026-06-04
**Branch:** `feature/multi-episode-training`
**Status:** Design approved; ready for implementation plan.

## Problem

The first real training run produced the golden Infinite Run agent
(`genomes/inf1234_w1_20260604-183539/`, fitness 6928). It was trained — and
selected — on a **single** terrain seed (`infinite_seed=1234`, `world_seed=1`).
Selecting on one episode rewards memorizing that one terrain, not general
competence. Two concrete gaps:

1. **Overfitting to one seed.** Every genome in every generation faced the exact
   same Infinite Run terrain. A genome that exploits a quirk of seed 1234 scores
   identically to one that generalizes. There is no pressure toward robustness.

2. **The static levels are never trained.** The five hand-built levels
   (`tutorial_hill`, `maze`, `lava_rising`, `vertical_climb`, `speed_run`) have
   goals, keys, and collectibles — exactly the fitness terms (`+200·goal`,
   `+100·keys`, `+50·collectibles`) that Infinite Run never exercises. No script
   trains on them: `train()` accepts a `level_path` but nothing drives it, and
   there is no way to train one agent across several levels at once.

## Goals

- Score each genome across **multiple episodes** (terrain seeds and/or static
  levels) and select on an aggregate, so the GA rewards generalization.
- Aggregate as **mean − λ·std** of the per-episode fitnesses — rewards
  consistency, not just a good average.
- Use a **fixed episode set** for the whole run: clean, comparable fitness
  curves and byte-reproducible runs.
- Add a **static-level trainer CLI** that trains one generalist across a chosen
  set of levels, with **per-level normalization** so a big level doesn't
  dominate selection.
- Extend the **infinite trainer CLI** with opt-in multi-seed, default behavior
  unchanged.
- Keep one GA loop. Single-episode training stays byte-identical to today.

## Non-goals

- Resampling the episode set each generation (considered; rejected for v1 in
  favor of comparable curves and simple reproducibility).
- Mixing infinite-run and static-level episodes in a single aggregate — the
  scales differ and the two trainers stay separate runs.
- A "watch-best" playback tool, fitness-formula changes, GA-operator changes, or
  network/encoding changes — all out of scope.
- Curriculum / difficulty scheduling.

## Design

### Architecture

A shared **multi-episode evaluation core**. One genome is scored over a *list*
of episodes; the per-episode fitnesses are normalized and aggregated into the
single number the GA selects on. The GA loop, breeding, elitism, history, and
persistence are unchanged — only the per-genome evaluation unit changes. Two
thin CLIs build different episode lists on top of the same core.

Single-episode training is the **N=1 degenerate case** and reproduces today's
behavior exactly, which is what keeps every existing test and `train_infinite.py`
green.

### New module: `src/blueball/ai/episodes.py`

Three small, isolated, testable units.

**`EpisodeSpec`** — a frozen, picklable dataclass (must survive
`multiprocessing.Pool` pickling):

```python
@dataclass(frozen=True)
class EpisodeSpec:
    kind: str            # "infinite" | "static"
    seed: int            # sampler_seed for infinite; ignored for static
    level_path: str | None  # for static (str path so it pickles cleanly)
    world_seed: int
    max_steps: int
    norm: float = 1.0    # divisor applied to this episode's raw fitness
```

**`aggregate_fitness(scores: Sequence[float], lam: float) -> float`** — pure:

```
mean(scores) − lam · pstd(scores)
```

- A single score → population std is 0 → returns that score **exactly**. This is
  the property that makes N=1 byte-identical to today.
- Empty `scores` → `ValueError`.
- Uses population std (ddof=0) so a 1-element list is well-defined.

**`compute_level_par(level_path: str) -> float`** — the per-level normalizer.
Builds the level once in a throwaway headless `World`, counts its contents from
`world.entities` (by class name, the same way the observation layer classifies
entities), and returns a reference "fully-solved" score using the **same weights
as the fitness function**:

```
par = total_width
    + 200 · (1 if any Goal else 0)
    + 100 · count(Key)
    +  50 · count(Collectible)
```

- `total_width` comes from the `LevelMeta` returned by `load_level`.
- Guard: if `par <= 0` (degenerate/empty level) → return `1.0` so the caller's
  `norm` never divides by zero.
- Called **once per level** at run start, never inside the evaluation loop.

### Evaluator: `evaluate_episodes` in `trainer.py`

A top-level (picklable) function that replaces the per-genome eval unit:

```python
def evaluate_episodes(args) -> tuple[int, float]:
    idx, genome, episodes, lam = args
    scores = []
    for ep in episodes:
        if ep.kind == "infinite":
            _, raw = evaluate_infinite((idx, genome, ep.seed, ep.world_seed, ep.max_steps))
        else:
            _, raw = evaluate((idx, genome, ep.world_seed, ep.level_path, ep.max_steps))
        scores.append(raw / ep.norm)
    return idx, aggregate_fitness(scores, lam)
```

It reuses the **existing** `evaluate` / `evaluate_infinite` bodies verbatim —
no duplication of the physics/eval loop. Empty `episodes` → `ValueError`.

### `train()` changes (backward compatible)

```python
train(..., episodes: Sequence[EpisodeSpec] | None = None,
           lam: float = config.GA_FITNESS_STD_PENALTY)
```

- If `episodes` is provided, it is the source of truth and `evaluate_episodes`
  is the eval unit (the `lam` rides along in the args tuple).
- If `episodes is None`, `train()` builds a **1-element list** from the existing
  `level_path=` / `infinite_seed=` arguments (with `norm=1.0`). Existing call
  sites, `train_infinite.py`, and all current tests therefore run an unchanged
  code path, and `aggregate_fitness([x], lam) == x`.

Everything downstream of evaluation (sort by index, history dict, breeding with
`config.GA_*`, `on_generation`, persistence) is untouched.

### CLI: `train_infinite.py` (extended)

- New flags: `--seeds 1234,777,9` (explicit list) **or** `--num-seeds N`
  (generate N distinct sampler seeds deterministically from the base
  `--infinite-seed` via a `np.random.default_rng(infinite_seed)` draw). If both
  are passed, `--seeds` (explicit) wins; if neither, the single base seed is used.
- **Default is the single base seed** → byte-identical to today; multi-seed is
  opt-in.
- Builds N infinite `EpisodeSpec`s (`norm=1.0`, shared `world_seed`/`max_steps`)
  and passes `episodes=[...]` to `train()`.

### CLI: `train_levels.py` (new)

- Sibling of `train_infinite.py`; same `--pop/--gens/--max-steps/--ga-seed/
  --world-seed/--workers` flags.
- `--levels tutorial_hill,maze,...` — default all five; a subset is allowed; a
  single name is single-level training. Names resolve against
  `src/blueball/levels/<name>.json`; an unknown name → a clear error listing the
  valid names.
- For each level: `compute_level_par()` once, build a static `EpisodeSpec` with
  that `norm`. Pass `episodes=[...]` to `train()`.

### Why infinite stays raw (norm=1.0)

All infinite seeds share the same fitness scale (distance-dominated), so
normalization isn't needed to keep them comparable, and keeping raw fitness
means new infinite runs report on the **familiar thousands scale** — directly
comparable to the golden's 6928. Consequence: λ acts on raw fitness for infinite
runs and on normalized ~0..1 fitness for static runs. That's acceptable because
they are separate runs and `lam` is per-run configurable.

### Persistence (`src/blueball/ai/persistence.py`)

- Run-folder naming extended for multi-episode runs:
  - single-seed infinite keeps today's `inf1234_w1_<ts>`,
  - multi-seed infinite → `inf1234x3_w1_<ts>` (base seed × N),
  - static levels → `lvls5_w1_<ts>` (level count).
- `run.json` gains an `episodes` array (each episode's
  `kind`/`seed`-or-`level`/`world_seed`/`max_steps`/`norm`) and the `lam` value,
  alongside the existing fields. The golden's existing `run.json` is a strict
  subset of the new format — nothing is removed or renamed.

### Config (`src/blueball/config.py`)

- Add `GA_FITNESS_STD_PENALTY = 1.0` — the default λ.

## Testing

New file `tests/test_ai_multiepisode.py` (keeps the already-large
`tests/test_ai_smoke.py` focused):

- `aggregate_fitness`: N=1 returns the exact value (std 0); a known 2-value case
  matches `mean − λ·std`; empty list raises `ValueError`.
- `compute_level_par`: finite and positive for `tutorial_hill`, and equals
  `total_width + 200·goal + 100·keys + 50·collectibles` for a level whose
  contents are known; a contrived zero-content case returns `1.0`.
- `evaluate_episodes`: returns `(idx, finite float)`; a **single-episode result
  equals the raw `evaluate` / `evaluate_infinite` fitness** for the same args
  (backward-compat equivalence).
- Determinism: two `train(episodes=[two infinite seeds], ga_seed=0)` runs
  produce a **byte-identical** `best_genome`.
- Pool equivalence: `evaluate_episodes` via `Pool.imap` matches serial `map`
  (mirrors the existing pinned pool-equality test).
- Smoke: a small multi-episode `train` (pop 8, 3 gens, 2 episodes, tiny
  `max_steps`) → no crash, `history` length 3 with finite `best`/`mean`/`min`,
  `best_genome.shape == (510,)` float32.
- The existing single-seed / single-level `train` tests stay green **unchanged**.

Full suite (currently 360) must stay green after additions.

## Risks

- **Episode-set overfitting persists at smaller scale.** A fixed set means
  agents can still overfit to those N episodes, just N instead of 1. Mitigated by
  choosing N ≥ 3 and revisited only if generalization is still poor (resampling
  per generation is the documented next lever).
- **`compute_level_par` couples to entity class names** (`Goal`/`Key`/
  `Collectible`). If those are renamed, par silently changes; the par test pins
  the expected value for a known level to catch this.
- **Static-level training is unproven.** Hard levels (e.g. `maze`, which needs
  memory the feed-forward net lacks) may not train well; this design only
  provides the harness, not a guarantee of good agents. Single-level runs let us
  isolate which levels are learnable.
- **No production caller depends on absolute fitness values** (selection is
  comparative), so adding normalization to the static path is safe; the infinite
  path keeps raw fitness specifically to stay comparable to the golden.
