# Multi-Episode Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score each genome across a fixed set of episodes (multiple Infinite-Run seeds and/or static levels), aggregate as mean − λ·std with per-level normalization, and expose it through a multi-seed `train_infinite.py` and a new `train_levels.py`.

**Architecture:** A shared multi-episode evaluation core in `src/blueball/ai/`. `EpisodeSpec` describes one episode; `evaluate_episodes` runs a genome over a list of them by reusing the existing `evaluate`/`evaluate_infinite` bodies, then aggregates. `train()` gains an `episodes=` parameter; single-episode is the N=1 degenerate case, so all existing call sites and tests run an unchanged code path. Two thin CLIs build different episode lists.

**Tech Stack:** Python 3, numpy, pymunk (physics, existing), pytest. Spec: `docs/superpowers/specs/2026-06-04-multi-episode-training-design.md`.

**Branch:** `feature/multi-episode-training`

**Conventions:**
- Run tests with `pytest -q` (pytest is on PATH; venv at `.venv/`).
- Level paths in tests: `Path(blueball.__file__).parent / "levels" / "<name>.json"`.
- `GENOME_SIZE == 510`, genomes are `float32` shape `(510,)`.
- All new tests go in `tests/test_ai_multiepisode.py` except the `run_dir_name` tests, which extend `tests/test_genome_persistence.py`.

---

### Task 0: `EpisodeSpec`, `aggregate_fitness`, and the λ config default

**Goal:** Create the pure core of the feature — the episode descriptor, the mean−λ·std aggregator, and the configurable λ default — with no dependency on `World`.

**Files:**
- Create: `src/blueball/ai/episodes.py`
- Modify: `src/blueball/config.py` (append one constant in the AI/GA block, after line 116)
- Create: `tests/test_ai_multiepisode.py`

**Acceptance Criteria:**
- [ ] `EpisodeSpec` is a frozen, picklable dataclass with fields `kind, seed, level_path, world_seed, max_steps, norm=1.0`.
- [ ] `aggregate_fitness([x], lam)` returns `x` exactly (single score → std 0).
- [ ] `aggregate_fitness([10.0, 20.0], 0.5) == 12.5` (mean 15 − 0.5·population-std 5).
- [ ] `aggregate_fitness([], lam)` raises `ValueError`.
- [ ] `config.GA_FITNESS_STD_PENALTY == 1.0`.

**Verify:** `pytest -q tests/test_ai_multiepisode.py -v` → 6 tests pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — create `tests/test_ai_multiepisode.py`:

```python
"""Tests for multi-episode training: aggregation, normalization, episode
construction, and the multi-episode trainer path."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import blueball


def test_aggregate_single_score_is_identity():
    # One episode -> population std is 0 -> returns the score exactly. This is
    # what makes single-episode training byte-identical to the old path.
    from blueball.ai.episodes import aggregate_fitness
    assert aggregate_fitness([42.5], lam=1.0) == 42.5


def test_aggregate_matches_mean_minus_lambda_std():
    from blueball.ai.episodes import aggregate_fitness
    # mean([10,20]) = 15; population std = 5; 15 - 0.5*5 = 12.5
    assert aggregate_fitness([10.0, 20.0], lam=0.5) == pytest.approx(12.5)


def test_aggregate_empty_raises():
    from blueball.ai.episodes import aggregate_fitness
    with pytest.raises(ValueError):
        aggregate_fitness([], lam=1.0)


def test_episodespec_is_frozen():
    from blueball.ai.episodes import EpisodeSpec
    ep = EpisodeSpec(kind="infinite", seed=1234, level_path=None,
                     world_seed=1, max_steps=100)
    with pytest.raises(Exception):
        ep.seed = 5  # frozen dataclass -> FrozenInstanceError


def test_episodespec_is_picklable():
    from blueball.ai.episodes import EpisodeSpec
    ep = EpisodeSpec(kind="static", seed=0, level_path="x.json",
                     world_seed=1, max_steps=100, norm=123.0)
    assert pickle.loads(pickle.dumps(ep)) == ep


def test_config_has_std_penalty_default():
    from blueball import config
    assert config.GA_FITNESS_STD_PENALTY == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blueball.ai.episodes'` (and the config test fails on missing attribute).

- [ ] **Step 3: Create `src/blueball/ai/episodes.py`**

```python
"""Multi-episode training: episode specs, fitness aggregation, per-level
normalization, and episode-list constructors.

A genome is scored across a *list* of EpisodeSpecs; each per-episode raw
fitness is divided by that episode's `norm` and the results are aggregated as
mean - lam*std. A single episode aggregates to itself (population std 0), so
single-episode training reproduces the pre-multi-episode behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeSpec:
    """One evaluation episode. Picklable so it survives multiprocessing.Pool."""

    kind: str                 # "infinite" | "static"
    seed: int                 # sampler_seed for infinite; ignored for static
    level_path: str | None    # for static (str so it pickles cleanly)
    world_seed: int
    max_steps: int
    norm: float = 1.0         # divisor applied to this episode's raw fitness


def aggregate_fitness(scores: Sequence[float], lam: float) -> float:
    """Combine per-episode fitnesses into one selection score: mean - lam*std.

    Uses population std (ddof=0), so a single score returns itself exactly.
    Empty input is a programming error.
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("aggregate_fitness requires at least one score")
    return float(arr.mean() - lam * arr.std())
```

- [ ] **Step 4: Add the config constant** — in `src/blueball/config.py`, immediately after the `GA_ELITISM = 1` line (line 116) in the `# AI / GA training` block:

```python
GA_FITNESS_STD_PENALTY = 1.0  # lambda: per-episode std penalty (mean - lam*std)
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add src/blueball/ai/episodes.py src/blueball/config.py tests/test_ai_multiepisode.py
git commit -m "feat(ai): EpisodeSpec + aggregate_fitness (mean - lambda*std) + config lambda"
```

---

### Task 1: `compute_level_par` — per-level normalizer

**Goal:** Add the per-level "fully-solved" reference score used to normalize static levels so a big level doesn't dominate multi-level selection.

**Files:**
- Modify: `src/blueball/ai/episodes.py` (add imports + `compute_level_par`)
- Modify: `tests/test_ai_multiepisode.py` (append 3 tests)

**Acceptance Criteria:**
- [ ] `compute_level_par(path)` for `tutorial_hill.json` equals `total_width + 200` (it has a goal, no keys, no collectibles).
- [ ] A flat-only in-memory level → par equals its `total_width` (no bonus terms).
- [ ] An empty-chunks level (par would be 0) → returns `1.0` (divide-by-zero guard).
- [ ] Accepts a path, `Path`, or in-memory dict (same as `load_level`).

**Verify:** `pytest -q tests/test_ai_multiepisode.py -v` → 9 tests pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_multiepisode.py`:

```python
def _levels_dir() -> Path:
    return Path(blueball.__file__).parent / "levels"


def _load_meta(level):
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    return load_level(level, world)


def test_level_par_tutorial_hill_is_width_plus_goal():
    from blueball.ai.episodes import compute_level_par
    path = _levels_dir() / "tutorial_hill.json"
    # tutorial_hill has a goal, no keys, no collectibles -> par = width + 200
    meta = _load_meta(path)
    assert compute_level_par(path) == pytest.approx(meta.total_width + 200.0)


def test_level_par_flat_only_has_no_bonus():
    from blueball.ai.episodes import compute_level_par
    level = {
        "name": "Flat", "background": "#000000", "ground": "#000000",
        "spawn": [80, 540],
        "chunks": [{"type": "flat", "width_tiles": 5}],
    }
    par = compute_level_par(level)
    meta = _load_meta(level)
    assert par > 0.0
    assert par == pytest.approx(meta.total_width)  # no goal/keys/collectibles


def test_level_par_empty_returns_one():
    from blueball.ai.episodes import compute_level_par
    level = {
        "name": "Empty", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "chunks": [],
    }
    assert compute_level_par(level) == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -k level_par -v`
Expected: FAIL — `ImportError: cannot import name 'compute_level_par'`.

- [ ] **Step 3: Add imports + implementation** — in `src/blueball/ai/episodes.py`, extend the import block at the top to add:

```python
from pathlib import Path
from typing import Sequence, Union

from ..collision import register as register_collisions
from ..levels.loader import load_level
from ..world import World
```

(Keep the existing `from dataclasses import dataclass` and `import numpy as np`. Merge the `typing` import so it reads `from typing import Sequence, Union`.)

Then append the function and its entity-name constants:

```python
_GOAL_NAME = "Goal"
_KEY_NAME = "Key"
_COLLECTIBLE_NAME = "Collectible"


def compute_level_par(level: Union[str, Path, dict]) -> float:
    """Reference 'fully-solved' fitness for a static level, used to normalize it
    so big levels don't dominate multi-level selection. Built once per level
    (never inside the eval loop). Same weights as ai/fitness.py:

        par = total_width + 200*has_goal + 100*keys + 50*collectibles

    Counts entities by class name (Goal/Key/Collectible), the way the
    observation layer classifies them. Guards par > 0 so callers never divide
    by zero.
    """
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(level, world)
    names = [type(e).__name__ for e in world.entities]
    par = (
        float(meta.total_width)
        + 200.0 * (1.0 if _GOAL_NAME in names else 0.0)
        + 100.0 * names.count(_KEY_NAME)
        + 50.0 * names.count(_COLLECTIBLE_NAME)
    )
    return par if par > 0.0 else 1.0
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/episodes.py tests/test_ai_multiepisode.py
git commit -m "feat(ai): compute_level_par per-level normalizer"
```

---

### Task 2: Episode-list constructors (seeds, infinite, static, level resolution)

**Goal:** Add the small, testable builders the CLIs use to assemble episode lists: deterministic multi-seed generation, infinite/static `EpisodeSpec` lists, and level-name → path resolution.

**Files:**
- Modify: `src/blueball/ai/episodes.py` (append constructors)
- Modify: `tests/test_ai_multiepisode.py` (append 5 tests)

**Acceptance Criteria:**
- [ ] `generate_seeds(1234, 1) == [1234]`.
- [ ] `generate_seeds(1234, 4)` is deterministic, has the base seed first, and yields 4 distinct ints.
- [ ] `infinite_episodes([1,2], world_seed, max_steps)` → 2 infinite specs, `norm==1.0`, `level_path is None`.
- [ ] `resolve_level_paths(["does_not_exist"])` raises `ValueError` mentioning "Available".
- [ ] `static_episodes([tutorial_hill_path], ...)` → one static spec whose `norm == compute_level_par(path)`.

**Verify:** `pytest -q tests/test_ai_multiepisode.py -v` → 14 tests pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_multiepisode.py`:

```python
def test_generate_seeds_single():
    from blueball.ai.episodes import generate_seeds
    assert generate_seeds(1234, 1) == [1234]


def test_generate_seeds_distinct_deterministic_and_includes_base():
    from blueball.ai.episodes import generate_seeds
    a = generate_seeds(1234, 4)
    b = generate_seeds(1234, 4)
    assert a == b                # deterministic
    assert a[0] == 1234          # base seed first
    assert len(set(a)) == 4      # distinct


def test_infinite_episodes_build():
    from blueball.ai.episodes import infinite_episodes
    eps = infinite_episodes([1, 2], world_seed=1, max_steps=100)
    assert [e.kind for e in eps] == ["infinite", "infinite"]
    assert [e.seed for e in eps] == [1, 2]
    assert all(e.norm == 1.0 and e.level_path is None for e in eps)


def test_resolve_level_paths_unknown_raises():
    from blueball.ai.episodes import resolve_level_paths
    with pytest.raises(ValueError) as exc:
        resolve_level_paths(["does_not_exist"])
    assert "Available" in str(exc.value)


def test_resolve_and_static_episodes_tutorial_hill():
    from blueball.ai.episodes import (compute_level_par, resolve_level_paths,
                                      static_episodes)
    paths = resolve_level_paths(["tutorial_hill"])
    assert paths[0].endswith("tutorial_hill.json")
    eps = static_episodes(paths, world_seed=1, max_steps=100)
    assert eps[0].kind == "static"
    assert eps[0].norm == pytest.approx(compute_level_par(paths[0]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -k "seeds or episodes or resolve" -v`
Expected: FAIL — `ImportError: cannot import name 'generate_seeds'`.

- [ ] **Step 3: Append the constructors** to `src/blueball/ai/episodes.py`:

```python
LEVELS_DIR = Path(__file__).resolve().parent.parent / "levels"


def generate_seeds(base: int, n: int) -> list[int]:
    """N distinct sampler seeds derived deterministically from `base`. The base
    seed is always first, so a multi-seed run still includes the reference
    course. Used by the infinite trainer's --num-seeds."""
    if n <= 1:
        return [int(base)]
    rng = np.random.default_rng(int(base))
    seeds = [int(base)]
    while len(seeds) < n:
        s = int(rng.integers(0, 2**31 - 1))
        if s not in seeds:
            seeds.append(s)
    return seeds


def infinite_episodes(seeds, world_seed, max_steps) -> list[EpisodeSpec]:
    """One infinite-run EpisodeSpec per sampler seed (norm=1.0: all infinite
    seeds share the same distance-dominated scale)."""
    return [
        EpisodeSpec(kind="infinite", seed=int(s), level_path=None,
                    world_seed=int(world_seed), max_steps=int(max_steps))
        for s in seeds
    ]


def available_levels() -> list[str]:
    """Sorted level names discoverable under the levels package directory."""
    return sorted(p.stem for p in LEVELS_DIR.glob("*.json"))


def resolve_level_paths(names) -> list[str]:
    """Map level names to JSON path strings, erroring on an unknown name."""
    available = available_levels()
    paths = []
    for name in names:
        if name not in available:
            raise ValueError(
                f"Unknown level {name!r}. Available: {', '.join(available)}"
            )
        paths.append(str(LEVELS_DIR / f"{name}.json"))
    return paths


def static_episodes(level_paths, world_seed, max_steps) -> list[EpisodeSpec]:
    """One static EpisodeSpec per level, each normalized by its level par."""
    return [
        EpisodeSpec(kind="static", seed=0, level_path=str(p),
                    world_seed=int(world_seed), max_steps=int(max_steps),
                    norm=compute_level_par(p))
        for p in level_paths
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/episodes.py tests/test_ai_multiepisode.py
git commit -m "feat(ai): episode-list constructors (seeds, infinite, static, level resolution)"
```

---

### Task 3: `evaluate_episodes` + multi-episode `train()` + run.json fields

**Goal:** Wire the core into the trainer: a picklable `evaluate_episodes` worker, a backward-compatible `episodes=`/`lam=` on `train()`, and `episodes`/`lam` recorded in `run.json`.

**Files:**
- Modify: `src/blueball/ai/trainer.py` (imports; add `evaluate_episodes`; restructure `train()` eval setup; extend finalize meta)
- Modify: `tests/test_ai_multiepisode.py` (append 5 tests)

**Acceptance Criteria:**
- [ ] `evaluate_episodes((idx, genome, (one_infinite_spec,), lam))` returns `(idx, finite float)` equal to the raw `evaluate_infinite` fitness for the same args (single-episode equivalence).
- [ ] `evaluate_episodes` with an empty episode tuple raises `ValueError`.
- [ ] `train(episodes=[...], ga_seed=0)` run twice → byte-identical `best_genome`.
- [ ] Multi-episode `train` smoke (pop 8, 3 gens, 2 episodes) → `history` length 3, finite stats, `best_genome.shape == (510,)` float32.
- [ ] `train` with `map_fn=Pool.imap` matches serial `map` `best_genome`.
- [ ] Existing `tests/test_ai_smoke.py` and `tests/test_genome_persistence.py` stay green (single-episode path unchanged).

**Verify:** `pytest -q tests/test_ai_multiepisode.py tests/test_ai_smoke.py tests/test_genome_persistence.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_multiepisode.py`:

```python
def test_evaluate_episodes_single_equals_raw_evaluate_infinite():
    from blueball.ai.episodes import EpisodeSpec
    from blueball.ai.genome import random_genome
    from blueball.ai.trainer import evaluate_episodes, evaluate_infinite
    g = random_genome(np.random.default_rng(0))
    _, raw = evaluate_infinite((0, g, 1234, 1, 120))
    ep = EpisodeSpec(kind="infinite", seed=1234, level_path=None,
                     world_seed=1, max_steps=120)
    _, agg = evaluate_episodes((0, g, (ep,), 1.0))
    assert agg == pytest.approx(raw)


def test_evaluate_episodes_empty_raises():
    from blueball.ai.trainer import evaluate_episodes
    g = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError):
        evaluate_episodes((0, g, (), 1.0))


def test_train_multi_episode_is_deterministic():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=120)
    a = train(pop_size=6, generations=3, episodes=eps, ga_seed=0)
    b = train(pop_size=6, generations=3, episodes=eps, ga_seed=0)
    assert np.array_equal(a.best_genome, b.best_genome)


def test_train_multi_episode_smoke():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=80)
    result = train(pop_size=8, generations=3, episodes=eps, ga_seed=0)
    assert len(result.history) == 3
    for h in result.history:
        assert np.isfinite(h["best"]) and np.isfinite(h["mean"]) and np.isfinite(h["min"])
    assert result.best_genome.shape == (510,)
    assert result.best_genome.dtype == np.float32


def test_train_multi_episode_pool_matches_serial():
    import multiprocessing as mp
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=60)
    serial = train(pop_size=6, generations=2, episodes=eps, ga_seed=0, map_fn=map)
    with mp.Pool(2) as pool:
        par = train(pop_size=6, generations=2, episodes=eps, ga_seed=0,
                    map_fn=pool.imap)
    assert np.array_equal(serial.best_genome, par.best_genome)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -k "evaluate_episodes or multi_episode" -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_episodes'`.

- [ ] **Step 3: Add imports** — in `src/blueball/ai/trainer.py`, extend the imports:

Change `from dataclasses import dataclass` to:
```python
from dataclasses import asdict, dataclass
```
Change `from typing import Callable, Iterable` to:
```python
from typing import Callable, Iterable, Sequence
```
Add this line alongside the other `from .` imports (e.g. after `from .genome import random_genome`):
```python
from .episodes import EpisodeSpec, aggregate_fitness
```
(`evaluate_episodes` is defined below in this same file — do not import it.)

- [ ] **Step 4: Add `evaluate_episodes`** — in `src/blueball/ai/trainer.py`, immediately after `evaluate_infinite` (after line 140):

```python
def evaluate_episodes(args: tuple) -> tuple[int, float]:
    """Score one genome across a list of EpisodeSpecs and aggregate as
    mean - lam*std. Picklable in/out for multiprocessing.Pool. Args is
    (idx, genome, episodes, lam). Reuses evaluate / evaluate_infinite per
    episode; a single episode aggregates to its own raw fitness exactly."""
    idx, genome, episodes, lam = args
    if not episodes:
        raise ValueError("evaluate_episodes requires at least one episode")
    scores = []
    for ep in episodes:
        if ep.kind == "infinite":
            _, raw = evaluate_infinite(
                (idx, genome, ep.seed, ep.world_seed, ep.max_steps))
        else:
            _, raw = evaluate(
                (idx, genome, ep.world_seed, ep.level_path, ep.max_steps))
        scores.append(raw / ep.norm)
    return idx, aggregate_fitness(scores, lam)
```

- [ ] **Step 5: Restructure `train()`** — change the signature to add `episodes` and `lam`. Replace the signature block (lines 143-155) so it reads:

```python
def train(
    *,
    pop_size: int,
    generations: int,
    level_path: Path | None = None,
    infinite_seed: int | None = None,
    episodes: Sequence[EpisodeSpec] | None = None,
    lam: float = config.GA_FITNESS_STD_PENALTY,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable[[Callable, Iterable], Iterable] = map,
    on_generation: Callable[[int, np.ndarray, list[np.ndarray]], None] | None = None,
    save_dir: Path | str | None = None,
) -> TrainingResult:
```

Then replace the validation + eval-fn setup block (lines 173-189, from the `if pop_size < 1:` checks through the `else:` make_args branch) with:

```python
    if pop_size < 1:
        raise ValueError(f"train requires pop_size >= 1, got {pop_size}")
    if generations < 1:
        raise ValueError(f"train requires generations >= 1, got {generations}")

    if episodes is None:
        if (level_path is None) == (infinite_seed is None):
            raise ValueError(
                "train requires exactly one of level_path, infinite_seed, or episodes"
            )
        if infinite_seed is not None:
            episodes = [EpisodeSpec(kind="infinite", seed=int(infinite_seed),
                                    level_path=None, world_seed=world_seed,
                                    max_steps=max_steps)]
        else:
            episodes = [EpisodeSpec(kind="static", seed=0,
                                    level_path=str(level_path),
                                    world_seed=world_seed, max_steps=max_steps)]
    else:
        episodes = list(episodes)
        if not episodes:
            raise ValueError("train requires a non-empty episodes list")

    episodes = tuple(episodes)
    eval_fn = evaluate_episodes

    def make_args(i):
        return (i, population[i], episodes, lam)
```

(`population` is bound later, exactly as in the existing closure — `make_args` is only called after the population is built.)

- [ ] **Step 6: Record episodes + lam in run.json** — in the `writer.finalize(...)` meta dict (lines 237-248), add two keys after the `"level_path": ...` line:

```python
            "episodes": [asdict(ep) for ep in episodes],
            "lam": lam,
```

- [ ] **Step 7: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py tests/test_ai_smoke.py tests/test_genome_persistence.py -v`
Expected: PASS (new multi-episode tests + all existing AI/persistence tests green).

- [ ] **Step 8: Commit**

```bash
git add src/blueball/ai/trainer.py tests/test_ai_multiepisode.py
git commit -m "feat(ai): evaluate_episodes + multi-episode train() + run.json episodes/lam"
```

---

### Task 4: `run_dir_name` multi-seed / multi-level variants

**Goal:** Extend the run-folder naming to encode multi-seed infinite runs (`inf1234x3`) and multi-level runs (`lvls5`), keeping single-seed/single-level names unchanged.

**Files:**
- Modify: `src/blueball/ai/persistence.py` (`run_dir_name`)
- Modify: `tests/test_genome_persistence.py` (append 3 tests)

**Acceptance Criteria:**
- [ ] `run_dir_name(infinite_seed=1234, world_seed=1, timestamp="T")` → `"inf1234_w1_T"` (unchanged).
- [ ] `run_dir_name(infinite_seed=1234, world_seed=1, timestamp="T", num_seeds=3)` → `"inf1234x3_w1_T"`.
- [ ] `run_dir_name(world_seed=1, timestamp="T", num_levels=5)` → `"lvls5_w1_T"`.
- [ ] Existing `run_dir_name` tests stay green.

**Verify:** `pytest -q tests/test_genome_persistence.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_genome_persistence.py`:

```python
def test_run_dir_name_multi_seed_infinite():
    from blueball.ai.persistence import run_dir_name
    name = run_dir_name(infinite_seed=1234, world_seed=1, timestamp="T", num_seeds=3)
    assert name == "inf1234x3_w1_T"


def test_run_dir_name_multi_level():
    from blueball.ai.persistence import run_dir_name
    name = run_dir_name(world_seed=1, timestamp="T", num_levels=5)
    assert name == "lvls5_w1_T"


def test_run_dir_name_single_seed_unchanged():
    from blueball.ai.persistence import run_dir_name
    assert run_dir_name(infinite_seed=1234, world_seed=1, timestamp="T") == "inf1234_w1_T"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_genome_persistence.py -k "multi_seed or multi_level" -v`
Expected: FAIL — `TypeError: run_dir_name() got an unexpected keyword argument 'num_seeds'`.

- [ ] **Step 3: Replace `run_dir_name`** in `src/blueball/ai/persistence.py` (lines 27-36) with:

```python
def run_dir_name(
    *,
    world_seed: int,
    timestamp: str,
    infinite_seed: int | None = None,
    level_name: str | None = None,
    num_seeds: int = 1,
    num_levels: int | None = None,
) -> str:
    """Build the per-run folder name from seeds/levels + a timestamp string.

    inf1234_w1_<ts>      single-seed infinite (unchanged)
    inf1234x3_w1_<ts>    multi-seed infinite (base seed x N)
    lvls5_w1_<ts>        multi-level static run (level count)
    tutorial_hill_w7_T   single static level by name
    """
    if num_levels is not None:
        key = f"lvls{num_levels}"
    elif infinite_seed is not None:
        key = f"inf{infinite_seed}" if num_seeds <= 1 else f"inf{infinite_seed}x{num_seeds}"
    else:
        key = level_name or "level"
    return f"{key}_w{world_seed}_{timestamp}"
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q tests/test_genome_persistence.py -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/persistence.py tests/test_genome_persistence.py
git commit -m "feat(ai): run_dir_name multi-seed and multi-level variants"
```

---

### Task 5: `train_infinite.py` multi-seed CLI

**Goal:** Extend the headless infinite trainer with opt-in multi-seed (`--seeds` / `--num-seeds`), routing through the multi-episode `train()`. Default behavior (single seed) is unchanged except for the additive `episodes`/`lam` in run.json.

**Files:**
- Modify: `train_infinite.py`
- Modify: `tests/test_ai_multiepisode.py` (append 1 subprocess test)

**Acceptance Criteria:**
- [ ] `--seeds 1,2,3` (explicit) overrides `--num-seeds`; neither flag → single base `--infinite-seed`.
- [ ] Running with `--num-seeds 2` writes one `genomes/inf<seed>x2_w<ws>_<ts>/` folder containing `final_best.npy` and a `run.json` with 2 `episodes` and `lam == 1.0`.
- [ ] Exit code 0 on success.

**Verify:** `pytest -q tests/test_ai_multiepisode.py -k train_infinite_cli -v` → passes

**Steps:**

- [ ] **Step 1: Write the failing test** — append to `tests/test_ai_multiepisode.py`:

```python
def test_train_infinite_cli_writes_run(tmp_path):
    import json
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_infinite.py"
    r = subprocess.run(
        [sys.executable, str(script), "--pop", "4", "--gens", "2",
         "--max-steps", "60", "--num-seeds", "2", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("inf1234x2_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert len(meta["episodes"]) == 2
    assert meta["lam"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -k train_infinite_cli -v`
Expected: FAIL — the script still passes `infinite_seed=` (single) and ignores `--num-seeds`, so no `inf1234x2_w1_*` folder is produced.

- [ ] **Step 3: Rewrite `train_infinite.py`** — replace the whole file with:

```python
"""Headless reference training on Infinite Run.

Runs the GA on the pinned reference seed (config.INFINITE_RUN_SEED) by default,
or across multiple seeds for generalization, and persists the result into a
timestamped run folder under genomes/. No pygame / no display required.

    python train_infinite.py                      # single reference seed
    python train_infinite.py --num-seeds 3        # base seed + 2 derived seeds
    python train_infinite.py --seeds 1234,777,9   # explicit seed set
    python train_infinite.py --gens 50            # override generations

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import generate_seeds, infinite_episodes
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--infinite-seed", type=int, default=config.INFINITE_RUN_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="train across N seeds derived from --infinite-seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="explicit comma-separated sampler seeds (overrides --num-seeds)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = generate_seeds(args.infinite_seed, args.num_seeds)

    episodes = infinite_episodes(seeds, world_seed=args.world_seed,
                                 max_steps=args.max_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        infinite_seed=args.infinite_seed, world_seed=args.world_seed,
        timestamp=timestamp, num_seeds=len(seeds),
    )

    print(
        f"Training {args.pop}x{args.gens} on Infinite Run seeds={seeds} "
        f"world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train(
            pop_size=args.pop,
            generations=args.gens,
            episodes=episodes,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py -k train_infinite_cli -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_infinite.py tests/test_ai_multiepisode.py
git commit -m "feat(train): train_infinite.py multi-seed via --seeds / --num-seeds"
```

---

### Task 6: `train_levels.py` static-level CLI

**Goal:** New headless trainer that trains one generalist across a set of static levels (default: all), per-level-normalized, persisting into a `lvls<N>_…` run folder.

**Files:**
- Create: `train_levels.py`
- Modify: `tests/test_ai_multiepisode.py` (append 2 subprocess tests)

**Acceptance Criteria:**
- [ ] `--levels tutorial_hill` trains a single level; default (no flag) trains all discoverable levels.
- [ ] A successful run writes one `genomes/lvls<N>_w<ws>_<ts>/` folder with `final_best.npy` and a `run.json` whose `episodes` are `kind=="static"` with `norm > 1.0`.
- [ ] An unknown `--levels` name exits non-zero with a message containing "Available".

**Verify:** `pytest -q tests/test_ai_multiepisode.py -k train_levels_cli -v` → passes

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_multiepisode.py`:

```python
def test_train_levels_cli_writes_run(tmp_path):
    import json
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_levels.py"
    r = subprocess.run(
        [sys.executable, str(script), "--levels", "tutorial_hill",
         "--pop", "4", "--gens", "2", "--max-steps", "120", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("lvls1_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert len(meta["episodes"]) == 1
    assert meta["episodes"][0]["kind"] == "static"
    assert meta["episodes"][0]["norm"] > 1.0


def test_train_levels_cli_unknown_level_errors(tmp_path):
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_levels.py"
    r = subprocess.run(
        [sys.executable, str(script), "--levels", "nope",
         "--pop", "2", "--gens", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0
    assert "Available" in (r.stderr + r.stdout)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/test_ai_multiepisode.py -k train_levels_cli -v`
Expected: FAIL — `train_levels.py` does not exist, subprocess returns non-zero with `can't open file`.

- [ ] **Step 3: Create `train_levels.py`**:

```python
"""Headless training on the hand-built static levels.

Trains one generalist agent across a set of levels (default: all of them),
scoring each genome on every level and selecting on the per-level-normalized
mean - lam*std. Each level's fitness is divided by its 'par' so a big level
does not dominate selection. No pygame / no display required.

    python train_levels.py                              # all levels
    python train_levels.py --levels maze                # single level
    python train_levels.py --levels tutorial_hill,speed_run --gens 50

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import (available_levels, resolve_level_paths,
                                  static_episodes)
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--levels", type=str, default=None,
                        help="comma-separated level names (default: all)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    names = args.levels.split(",") if args.levels else available_levels()
    try:
        level_paths = resolve_level_paths(names)
    except ValueError as e:
        raise SystemExit(str(e))

    episodes = static_episodes(level_paths, world_seed=args.world_seed,
                               max_steps=args.max_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        world_seed=args.world_seed, timestamp=timestamp,
        num_levels=len(level_paths),
    )

    print(
        f"Training {args.pop}x{args.gens} on levels={names} "
        f"world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train(
            pop_size=args.pop,
            generations=args.gens,
            episodes=episodes,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.3f} mean={final['mean']:.3f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q tests/test_ai_multiepisode.py -k train_levels_cli -v`
Expected: PASS (both subprocess tests).

- [ ] **Step 5: Full-suite regression check**

Run: `pytest -q`
Expected: the full suite is green (360 prior tests + the new multi-episode tests).

- [ ] **Step 6: Commit**

```bash
git add train_levels.py tests/test_ai_multiepisode.py
git commit -m "feat(train): train_levels.py static-level generalist trainer"
```

---

## Post-implementation

- Update `genomes/README.md` only if the run-folder naming section needs the new `inf<seed>x<N>` / `lvls<N>` variants documented (small doc follow-up; optional).
- The actual training runs (producing new golden genomes) are a separate, user-driven step — this plan ships the harness, not new trained agents.
