# AI / GA Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Blue Ball AI / Genetic Algorithm (GA) scaffolding so a fixed-topology neural network (FTNN) population can be evolved end-to-end on `tutorial_hill.json`, prove it works with a no-crash smoke test, and make the training process visible in a new `TrainScene` with a free pan/zoom camera.

**Architecture:** New isolated `ai/` package holds the FTNN (`ftnn.py`), genome representation (`genome.py`), GA operators (`ga.py`), an `Observation` → input-vector adapter (`observation.py`), a fitness function (`fitness.py`), and a generation-loop trainer (`trainer.py`). `FTNNAgent` is a new subclass of `Agent` added to `agent.py`; the existing `Observation` dataclass is **not** modified (the parallel level-design branch owns it). `Player` and `collision.py` gain a shared pymunk `ShapeFilter` group so multiple agents share one `World` non-interactively, plus a per-player `reached_goal` flag so the trainer can score each agent individually. The trainer's parallelism strategy defaults to serial `map`; `multiprocessing.Pool` is opt-in for real training runs. `TrainScene` runs the trainer in-process and renders the population live; a new `FreeCamera` lets the developer pan with arrow keys and zoom with `+`/`-`.

**Tech Stack:** Python 3.11+, NumPy (already a dependency), PyGame-ce, Pymunk, pytest. No new third-party dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-26-ai-scaffolding-design.md`.

**Worktree convention:** This plan should be executed in `.worktrees/ai-scaffolding` on a `feature/ai-scaffolding` branch forked from `master`. The `using-git-worktrees` skill handles setup at execution time. `.worktrees/` is already gitignored.

---

## File structure

Final layout after this plan lands. New files marked `+`, modified files marked `~`.

```
src/blueball/
├── ai/                                       (+ new package)
│   ├── __init__.py                           (+ )
│   ├── ftnn.py                               (+ FTNN class + topology constants)
│   ├── genome.py                             (+ GENOME_SIZE + random_genome())
│   ├── ga.py                                 (+ mutate, crossover, tournament_select, breed)
│   ├── observation.py                        (+ observation_to_inputs() + RAY_COUNT)
│   ├── fitness.py                            (+ FitnessInputs + fitness())
│   └── trainer.py                            (+ evaluate() + train() + TrainingResult)
├── agent.py                                  (~ add FTNNAgent subclass; Observation untouched)
├── camera.py                                 (~ Camera.scale field + FreeCamera subclass)
├── collision.py                              (~ PLAYER_GROUP constant + on_goal reached_goal write)
├── config.py                                 (~ append TRAIN_* and GA_* and MAX_STEPS defaults)
├── entities/player.py                        (~ ShapeFilter group + reached_goal flag)
└── scenes/train.py                           (+ new TrainScene)

train_main.py                                 (+ new entry script for TrainScene)

tests/
├── test_ai_smoke.py                          (+ all AI scaffolding tests)
├── test_camera.py                            (~ scale + FreeCamera cases)
├── test_collision.py                         (~ goal-handler reached_goal case)
└── test_player.py                            (~ PLAYER_GROUP + reached_goal + two-player cases)
```

`renderer.py`, `levels/`, `chunks/`, level JSON, the existing `Observation` dataclass, and `main.py` are **not** touched.

---

## Task 0: FTNN topology + genome representation

**Goal:** Establish the network topology constants (`FTNN_INPUTS`, `FTNN_HIDDEN`, `FTNN_OUTPUTS`, `GENOME_SIZE`), the `FTNN` class with a pure-numpy forward pass, and the `random_genome()` factory. All downstream tasks depend on these constants.

**Files:**
- Create: `src/blueball/ai/__init__.py`
- Create: `src/blueball/ai/ftnn.py`
- Create: `src/blueball/ai/genome.py`
- Create: `tests/test_ai_smoke.py`

**Acceptance Criteria:**
- [ ] `FTNN_INPUTS == 14`, `FTNN_HIDDEN == 12`, `FTNN_OUTPUTS == 6`, `GENOME_SIZE == 258`.
- [ ] `FTNN(random_genome(rng)).forward(np.zeros(14, dtype=np.float32))` returns a `np.ndarray` of shape `(6,)` and dtype `float32`.
- [ ] `FTNN(np.zeros(258, dtype=np.float32)).forward(np.zeros(14, dtype=np.float32))` returns the all-zero output vector (sanity check: zero genome, zero input, tanh(0)=0 ⇒ zero out).
- [ ] `FTNN(genome_with_wrong_shape)` raises `ValueError` with a message that mentions the expected size.
- [ ] `random_genome(rng).shape == (258,)` and `random_genome(rng).dtype == np.float32`.
- [ ] `random_genome(rng_a) == random_genome(rng_b)` element-wise when `rng_a` and `rng_b` are constructed with the same seed (deterministic RNG plumbing).

**Verify:** `pytest -q tests/test_ai_smoke.py -v` → 6 tests pass

**Steps:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_ai_smoke.py`:

```python
"""Smoke tests for the AI / GA scaffolding.

Tests are appended across Tasks 0–6 (FTNN/genome, GA ops, observation
adapter, fitness, FTNNAgent, trainer). All AI-scaffolding test cases
live in this one file so the suite reads top-to-bottom.
"""

from __future__ import annotations

import numpy as np
import pytest


# ----- Task 0: FTNN topology + genome -----

def test_ftnn_topology_constants():
    from blueball.ai.ftnn import FTNN_INPUTS, FTNN_HIDDEN, FTNN_OUTPUTS, GENOME_SIZE
    assert FTNN_INPUTS == 14
    assert FTNN_HIDDEN == 12
    assert FTNN_OUTPUTS == 6
    # 14*12 + 12 + 12*6 + 6 = 258
    assert GENOME_SIZE == 258


def test_ftnn_forward_pass_shape_and_dtype():
    from blueball.ai.ftnn import FTNN, FTNN_INPUTS, FTNN_OUTPUTS, GENOME_SIZE
    genome = np.zeros(GENOME_SIZE, dtype=np.float32)
    net = FTNN(genome)
    y = net.forward(np.zeros(FTNN_INPUTS, dtype=np.float32))
    assert y.shape == (FTNN_OUTPUTS,)
    assert y.dtype == np.float32


def test_ftnn_zero_genome_zero_input_yields_zero_output():
    from blueball.ai.ftnn import FTNN, FTNN_INPUTS, GENOME_SIZE
    net = FTNN(np.zeros(GENOME_SIZE, dtype=np.float32))
    y = net.forward(np.zeros(FTNN_INPUTS, dtype=np.float32))
    assert np.all(y == 0.0)


def test_ftnn_rejects_wrong_genome_shape():
    from blueball.ai.ftnn import FTNN
    with pytest.raises(ValueError, match="258"):
        FTNN(np.zeros(100, dtype=np.float32))


def test_random_genome_shape_and_dtype():
    from blueball.ai.genome import random_genome, GENOME_SIZE
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert g.shape == (GENOME_SIZE,)
    assert g.dtype == np.float32


def test_random_genome_is_deterministic_under_same_seed():
    from blueball.ai.genome import random_genome
    a = random_genome(np.random.default_rng(42))
    b = random_genome(np.random.default_rng(42))
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: `ModuleNotFoundError: No module named 'blueball.ai'`

- [ ] **Step 3: Create `src/blueball/ai/__init__.py`**

Empty file (the package needs to exist; nothing is re-exported at the package level — callers import the specific module).

```python
"""AI / GA scaffolding for Blue Ball.

This package is isolated: nothing outside `ai/` imports anything from `ai/`
except the agent (`FTNNAgent` is the bridge in `blueball.agent`) and the
training scene / entry script.
"""
```

- [ ] **Step 4: Create `src/blueball/ai/ftnn.py`**

```python
"""Fixed-Topology Neural Network (FTNN) used by AI agents.

A two-layer fully-connected network: 14 inputs → 12 tanh hidden → 6 outputs.
The 6 outputs correspond one-to-one with the `Action` enum values; the
trainer's `FTNNAgent.act()` picks `argmax` to choose an action.

The whole network is parameterized by a single flat float32 genome array.
Layout:
    [W1 (14*12=168) | b1 (12) | W2 (12*6=72) | b2 (6)]
    GENOME_SIZE = 168 + 12 + 72 + 6 = 258
"""

from __future__ import annotations

import numpy as np

FTNN_INPUTS = 14
FTNN_HIDDEN = 12
FTNN_OUTPUTS = 6   # one per Action

_W1_SIZE = FTNN_INPUTS * FTNN_HIDDEN
_B1_SIZE = FTNN_HIDDEN
_W2_SIZE = FTNN_HIDDEN * FTNN_OUTPUTS
_B2_SIZE = FTNN_OUTPUTS

GENOME_SIZE = _W1_SIZE + _B1_SIZE + _W2_SIZE + _B2_SIZE


class FTNN:
    """A 14 → 12 tanh → 6 fully-connected network. Pure numpy."""

    def __init__(self, genome: np.ndarray) -> None:
        if genome.shape != (GENOME_SIZE,):
            raise ValueError(
                f"FTNN requires a genome of shape ({GENOME_SIZE},), got {genome.shape}"
            )
        if genome.dtype != np.float32:
            genome = genome.astype(np.float32)

        i = 0
        self._W1 = genome[i:i + _W1_SIZE].reshape(FTNN_INPUTS, FTNN_HIDDEN)
        i += _W1_SIZE
        self._b1 = genome[i:i + _B1_SIZE]
        i += _B1_SIZE
        self._W2 = genome[i:i + _W2_SIZE].reshape(FTNN_HIDDEN, FTNN_OUTPUTS)
        i += _W2_SIZE
        self._b2 = genome[i:i + _B2_SIZE]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one observation through the network. Returns shape (FTNN_OUTPUTS,)."""
        h = np.tanh(x @ self._W1 + self._b1)
        return (h @ self._W2 + self._b2).astype(np.float32, copy=False)
```

- [ ] **Step 5: Create `src/blueball/ai/genome.py`**

```python
"""Genome construction helpers for the FTNN."""

from __future__ import annotations

import numpy as np

from .ftnn import GENOME_SIZE


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """Sample a fresh genome from N(0, 1). Returns float32 ndarray (GENOME_SIZE,)."""
    return rng.standard_normal(GENOME_SIZE, dtype=np.float32)
```

- [ ] **Step 6: Run tests, confirm pass**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/__init__.py src/blueball/ai/ftnn.py src/blueball/ai/genome.py tests/test_ai_smoke.py
git commit -m "feat: FTNN topology + genome factory"
```

---

## Task 1: GA operators (mutate / crossover / tournament_select / breed)

**Goal:** Pure-function GA operators that take genome arrays and a numpy `Generator`, and return new genome arrays. No global state, no module-level RNG. Selection is tournament-style with configurable `k`. Breed handles a full next-generation step including elitism.

**Files:**
- Create: `src/blueball/ai/ga.py`
- Modify: `tests/test_ai_smoke.py` (append new tests)

**Acceptance Criteria:**
- [ ] `mutate(g, rng, rate=0.1, sigma=0.1)` returns a new array of the same shape and dtype; at `rate=0.0` returns an element-wise-equal copy; at `rate=1.0, sigma=1.0` changes at least 95% of weights.
- [ ] `mutate` does not modify its input array in place.
- [ ] `crossover(a, b, rng)` returns a new array of the same shape with each gene equal to either `a[i]` or `b[i]`. Given parent of all-zeros and parent of all-ones, the result has both 0s and 1s (deterministic seed; assert the fraction of 1s is in `(0.3, 0.7)` for a length-258 genome).
- [ ] `tournament_select(fitnesses, rng, k=4)` returns two distinct indices both drawn from a sample of size `k`. With `k == len(fitnesses)`, the returned pair are the two highest fitness indices.
- [ ] `breed(population, fitnesses, rng, elitism=1)` returns a new list of length `len(population)`. The top `elitism` genomes from `population` (ranked by `fitnesses`) appear unchanged in the returned list.

**Verify:** `pytest -q tests/test_ai_smoke.py -v` → all pass (6 from Task 0 + 6 new)

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 1: GA operators -----

def test_mutate_at_rate_zero_returns_equal_but_different_object():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.arange(258, dtype=np.float32)
    out = mutate(g, rng, rate=0.0, sigma=1.0)
    assert out is not g
    assert np.array_equal(out, g)


def test_mutate_at_rate_one_changes_most_weights():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.zeros(258, dtype=np.float32)
    out = mutate(g, rng, rate=1.0, sigma=1.0)
    changed = np.count_nonzero(out != g)
    assert changed / 258 > 0.95


def test_mutate_does_not_modify_input():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.zeros(258, dtype=np.float32)
    snapshot = g.copy()
    mutate(g, rng, rate=1.0, sigma=1.0)
    assert np.array_equal(g, snapshot)


def test_crossover_inherits_from_both_parents():
    from blueball.ai.ga import crossover
    rng = np.random.default_rng(0)
    a = np.zeros(258, dtype=np.float32)
    b = np.ones(258, dtype=np.float32)
    child = crossover(a, b, rng)
    frac_b = np.count_nonzero(child == 1.0) / 258
    assert 0.3 < frac_b < 0.7
    # Every gene came from either parent
    assert np.all((child == 0.0) | (child == 1.0))


def test_tournament_select_returns_top_two_when_k_is_full():
    from blueball.ai.ga import tournament_select
    rng = np.random.default_rng(0)
    fitnesses = np.array([1.0, 5.0, 3.0, 9.0, 7.0])
    i1, i2 = tournament_select(fitnesses, rng, k=5)
    assert {i1, i2} == {3, 4}     # indices of 9.0 and 7.0


def test_breed_preserves_population_size_and_elitism():
    from blueball.ai.ga import breed
    rng = np.random.default_rng(0)
    pop = [np.full(258, float(i), dtype=np.float32) for i in range(8)]
    fitnesses = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 7.0, 4.0])
    nxt = breed(pop, fitnesses, rng, elitism=1)
    assert len(nxt) == 8
    # The best (fitness 9.0 -> pop[3], all 3.0) must survive unchanged.
    elite = pop[3]
    assert any(np.array_equal(g, elite) for g in nxt)
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 6 from Task 0 pass; 6 new fail (`ModuleNotFoundError: No module named 'blueball.ai.ga'`).

- [ ] **Step 3: Create `src/blueball/ai/ga.py`**

```python
"""Genetic Algorithm operators over flat genome arrays.

All operators are pure functions of their arguments: they take genome
arrays and a numpy Generator, return new arrays. No module-level state.
"""

from __future__ import annotations

import numpy as np


def mutate(
    genome: np.ndarray,
    rng: np.random.Generator,
    *,
    rate: float = 0.1,
    sigma: float = 0.1,
) -> np.ndarray:
    """Return a mutated copy of `genome`. Each gene is perturbed by
    `rng.normal(0, sigma)` with probability `rate`. Input is never modified.
    """
    out = genome.copy()
    if rate <= 0.0:
        return out
    mask = rng.random(out.shape[0]) < rate
    noise = rng.normal(0.0, sigma, size=out.shape[0]).astype(np.float32)
    out[mask] += noise[mask]
    return out


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-gene uniform crossover. Each gene independently from A or B with 50/50."""
    if parent_a.shape != parent_b.shape:
        raise ValueError(
            f"crossover parents must match shape; got {parent_a.shape} vs {parent_b.shape}"
        )
    mask = rng.random(parent_a.shape[0]) < 0.5
    return np.where(mask, parent_a, parent_b).astype(np.float32, copy=False)


def tournament_select(
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    *,
    k: int = 4,
) -> tuple[int, int]:
    """Sample `k` indices uniformly without replacement; return the indices
    of the two highest-fitness members of that sample. `k` is clamped to
    `len(fitnesses)`.
    """
    n = len(fitnesses)
    k = min(k, n)
    if k < 2:
        raise ValueError("tournament_select requires k >= 2")
    pool = rng.choice(n, size=k, replace=False)
    # Sort the pool indices by their fitness, descending; take top 2.
    pool_sorted = pool[np.argsort(-fitnesses[pool], kind="stable")]
    return int(pool_sorted[0]), int(pool_sorted[1])


def breed(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    *,
    elitism: int = 1,
    tournament_k: int = 4,
    mutation_rate: float = 0.1,
    mutation_sigma: float = 0.1,
) -> list[np.ndarray]:
    """Produce the next generation. The top `elitism` genomes pass through
    unchanged; the rest are children of `crossover(parents) → mutate(child)`
    with tournament-selected parents.
    """
    n = len(population)
    if len(fitnesses) != n:
        raise ValueError("fitnesses must match population size")
    if elitism < 0 or elitism > n:
        raise ValueError("invalid elitism count")

    # Elitism: copy the top `elitism` genomes unchanged.
    elite_order = np.argsort(-fitnesses, kind="stable")[:elitism]
    next_gen: list[np.ndarray] = [population[int(i)].copy() for i in elite_order]

    while len(next_gen) < n:
        i, j = tournament_select(fitnesses, rng, k=tournament_k)
        child = crossover(population[i], population[j], rng)
        child = mutate(child, rng, rate=mutation_rate, sigma=mutation_sigma)
        next_gen.append(child)

    return next_gen
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/ga.py tests/test_ai_smoke.py
git commit -m "feat: GA operators (mutate, crossover, tournament, breed)"
```

---

## Task 2: `Observation` → input-vector adapter

**Goal:** A single helper `observation_to_inputs(obs)` that packs the existing v1 `Observation` into the 14-float vector the FTNN expects, with a clear assertion if `rays` ever changes shape. This is the only place that depends on `Observation`'s layout — when the level-design branch enriches `Observation`, only `RAY_COUNT` updates here.

**Files:**
- Create: `src/blueball/ai/observation.py`
- Modify: `tests/test_ai_smoke.py` (append new tests)

**Acceptance Criteria:**
- [ ] `RAY_COUNT == 8` (matches the current `Observation.rays.shape`).
- [ ] `observation_to_inputs(obs).shape == (14,)` and dtype is `float32`.
- [ ] Indices 0–7 are `obs.rays`; 8–9 are `obs.vel`; 10 is `obs.ang_vel`; 11 is `1.0` if `grounded` else `0.0`; 12–13 are the `nearest_collectible` offset (zeros when `None`).
- [ ] Passing an `Observation` whose `rays.shape != (8,)` raises `AssertionError` with a message that names both the expected and actual shapes.

**Verify:** `pytest -q tests/test_ai_smoke.py -v` → 12 + 4 new tests pass

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 2: Observation → input-vector adapter -----

def _make_obs(
    *,
    rays=None,
    vel=(0.0, 0.0),
    ang_vel=0.0,
    grounded=False,
    nearest_collectible=None,
):
    from blueball.agent import Observation
    if rays is None:
        rays = np.zeros(8, dtype=np.float32)
    return Observation(
        rays=rays,
        vel=np.asarray(vel, dtype=np.float32),
        ang_vel=float(ang_vel),
        grounded=bool(grounded),
        nearest_collectible=nearest_collectible,
    )


def test_observation_to_inputs_shape_and_dtype():
    from blueball.ai.observation import observation_to_inputs
    x = observation_to_inputs(_make_obs())
    assert x.shape == (14,)
    assert x.dtype == np.float32


def test_observation_to_inputs_layout_matches_spec():
    from blueball.ai.observation import observation_to_inputs
    rays = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    obs = _make_obs(
        rays=rays,
        vel=(11.0, -22.0),
        ang_vel=3.5,
        grounded=True,
        nearest_collectible=(50.0, -25.0),
    )
    x = observation_to_inputs(obs)
    np.testing.assert_allclose(x[0:8], rays)
    assert x[8] == 11.0 and x[9] == -22.0
    assert x[10] == 3.5
    assert x[11] == 1.0
    assert x[12] == 50.0 and x[13] == -25.0


def test_observation_to_inputs_handles_none_collectible():
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(nearest_collectible=None, grounded=False)
    x = observation_to_inputs(obs)
    assert x[11] == 0.0          # grounded=False → 0.0
    assert x[12] == 0.0 and x[13] == 0.0


def test_observation_to_inputs_rejects_wrong_ray_count():
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(rays=np.zeros(7, dtype=np.float32))
    with pytest.raises(AssertionError, match=r"\(8,\)"):
        observation_to_inputs(obs)
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 12 pass; 4 new fail (`ModuleNotFoundError: No module named 'blueball.ai.observation'`).

- [ ] **Step 3: Create `src/blueball/ai/observation.py`**

```python
"""Observation → FTNN input adapter.

Packs the existing v1 `Observation` into the 14-float input vector the
FTNN expects.

Layout (indices):
    0–7:  obs.rays                              (8 floats)
    8–9:  obs.vel[0], obs.vel[1]                (2 floats)
     10:  obs.ang_vel                           (1 float)
     11:  1.0 if obs.grounded else 0.0          (1 float)
    12–13: nearest_collectible offset (0, 0 when None)

DEPENDENCY: if the level-design branch's Observation enrichment ever changes
rays.shape from (8,) to a different size, update RAY_COUNT here AND FTNN_INPUTS
in ai/ftnn.py in lockstep. The assertion below catches the mismatch with a
clear message.
"""

from __future__ import annotations

import numpy as np

from ..agent import Observation

RAY_COUNT = 8


def observation_to_inputs(obs: Observation) -> np.ndarray:
    assert obs.rays.shape == (RAY_COUNT,), (
        f"observation_to_inputs expects rays of shape ({RAY_COUNT},), "
        f"got {obs.rays.shape} — update RAY_COUNT and FTNN_INPUTS together."
    )
    x = np.empty(14, dtype=np.float32)
    x[0:8] = obs.rays
    x[8] = obs.vel[0]
    x[9] = obs.vel[1]
    x[10] = obs.ang_vel
    x[11] = 1.0 if obs.grounded else 0.0
    if obs.nearest_collectible is None:
        x[12] = 0.0
        x[13] = 0.0
    else:
        x[12] = obs.nearest_collectible[0]
        x[13] = obs.nearest_collectible[1]
    return x
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/observation.py tests/test_ai_smoke.py
git commit -m "feat: Observation -> FTNN input-vector adapter"
```

---

## Task 3: Fitness function

**Goal:** Pure function from a snapshot dataclass (`FitnessInputs`) to a single float. The v1 spec's starting fitness shape verbatim. Decoupled from live `Player` entities so the function is trivially testable.

**Files:**
- Create: `src/blueball/ai/fitness.py`
- Modify: `tests/test_ai_smoke.py` (append new tests)

**Acceptance Criteria:**
- [ ] All-zero `FitnessInputs` returns exactly `0.0`.
- [ ] `reached_goal=True` adds 200.0 to the result; `died=True` subtracts 100.0; one collectible adds 50.0; `steps_taken` subtracts 0.01 each.
- [ ] `progress_x` adds linearly (1:1).
- [ ] Function does not import `Player` or any pymunk type.

**Verify:** `pytest -q tests/test_ai_smoke.py -v` → 16 + 3 new tests pass

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 3: Fitness -----

def test_fitness_all_zero_returns_zero():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=0.0, collectibles=0, reached_goal=False,
        died=False, steps_taken=0,
    ))
    assert f == 0.0


def test_fitness_shape_matches_spec_formula():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=500.0,
        collectibles=3,
        reached_goal=True,
        died=False,
        steps_taken=1000,
    ))
    # 500 + 50*3 + 200 - 0.01*1000 - 0  = 500 + 150 + 200 - 10 = 840
    assert f == pytest.approx(840.0)


def test_fitness_penalizes_death_and_charges_step_cost():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=10.0, collectibles=0, reached_goal=False,
        died=True, steps_taken=500,
    ))
    # 10 + 0 + 0 - 5 - 100 = -95
    assert f == pytest.approx(-95.0)
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 16 pass; 3 new fail.

- [ ] **Step 3: Create `src/blueball/ai/fitness.py`**

```python
"""Fitness function for GA training. v1 spec's starting shape.

Tunable; this is the function most likely to be iterated during real
training. Lives in its own module so iteration touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float    # player.body.position.x - spawn_x
    collectibles: int    # player.collectibles_collected
    reached_goal: bool   # player.reached_goal
    died: bool           # player.dead
    steps_taken: int     # the loop counter from evaluate()


def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 50.0  * inputs.collectibles
        + 200.0 * (1.0 if inputs.reached_goal else 0.0)
        -   0.01 * inputs.steps_taken
        - 100.0 * (1.0 if inputs.died else 0.0)
    )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/fitness.py tests/test_ai_smoke.py
git commit -m "feat: GA fitness function"
```

---

## Task 4: `Player` shared collision-filter group + `reached_goal` flag

**Goal:** Make N `Player` entities coexist non-interactively in one `World` by giving every player a single shared non-zero `ShapeFilter.group`, and track per-player goal arrival so the trainer can score each agent. `PlayScene` (single-player) is unaffected because the "shapes in same group don't collide" rule is trivially satisfied with one player.

**Files:**
- Modify: `src/blueball/collision.py` (add `PLAYER_GROUP = 99`; update `on_goal` handler)
- Modify: `src/blueball/entities/player.py` (set `shape.filter`; init `reached_goal`)
- Modify: `tests/test_player.py`
- Modify: `tests/test_collision.py`

**Acceptance Criteria:**
- [ ] `collision.PLAYER_GROUP == 99`.
- [ ] A freshly-constructed `Player` has `player.shape.filter.group == collision.PLAYER_GROUP` and `player.reached_goal is False`.
- [ ] Two `Player` entities placed at the same spawn position in one `World` do not collide with each other across at least 30 physics ticks (their positions stay identical, modulo floating-point noise).
- [ ] `on_goal` collision handler sets `player.reached_goal = True` on the involved player **and** still calls `world_ref.complete_level()` (so `world.level_complete` flips for PlayScene).
- [ ] No regression: the existing test suite passes (including PlayScene tests and existing collision tests).

**Verify:** `pytest -q tests/test_player.py tests/test_collision.py -v` → all pass, then `pytest -q` → full suite green

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_player.py`**

Append (the file already imports `Player`, `World`, `Action`, `_ScriptedAgent`, etc.):

```python
def test_player_shape_has_player_group_collision_filter():
    from blueball import collision
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.shape.filter.group == collision.PLAYER_GROUP
    assert collision.PLAYER_GROUP == 99


def test_player_reached_goal_defaults_false():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.reached_goal is False


def test_two_players_in_one_world_do_not_collide():
    """Both players spawn at the same point, share PLAYER_GROUP, and so
    must not exert contact forces on each other. With identical agents
    they should follow identical trajectories under gravity."""
    w = World()
    p1 = Player(agent=_ScriptedAgent([Action.IDLE] * 50), spawn_xy=(200, 100))
    p2 = Player(agent=_ScriptedAgent([Action.IDLE] * 50), spawn_xy=(200, 100))
    w.add_entity(p1)
    w.add_entity(p2)
    for _ in range(30):
        w.step(1 / 60)
    # Same spawn + same inputs + no mutual contact = identical positions.
    assert abs(p1.body.position.x - p2.body.position.x) < 1e-6
    assert abs(p1.body.position.y - p2.body.position.y) < 1e-6
```

- [ ] **Step 2: Add failing tests to `tests/test_collision.py`**

Append (the file already provides `_player_world()` and a basic `World`/`Player` setup):

```python
def test_goal_handler_sets_player_reached_goal():
    """Player overlapping the goal sensor → player.reached_goal True AND
    world.level_complete True (the existing PlayScene path is preserved)."""
    from blueball.entities.goal import Goal
    w, p = _player_world()
    # Position the goal directly on the player so contact happens on the next step.
    goal = Goal(position=(p.body.position.x, p.body.position.y), width=40, height=40)
    w.add_entity(goal)

    for _ in range(5):
        w.step(1 / 60)
        if p.reached_goal:
            break
    assert p.reached_goal is True
    assert w.level_complete is True
```

If `tests/test_collision.py` does not have a `_player_world()` helper that returns `(world, player)` ready for stepping, mirror the helper used by the existing tests in the file (the abilities-framework plan's Task 3 introduced one for the AbilityPickup case — reuse that pattern).

If the test imports `Goal` from a different path than `blueball.entities.goal`, adjust the import to match what `tests/test_entities.py` uses for `Goal`.

- [ ] **Step 3: Run, confirm failure**

Run: `pytest -q tests/test_player.py tests/test_collision.py -v`
Expected: new tests fail because `PLAYER_GROUP` constant missing, `reached_goal` attribute missing, and goal handler does not set the flag.

- [ ] **Step 4: Modify `src/blueball/collision.py`**

Add a new constant alongside the existing `CT_*`:

```python
# Shared shape-filter group for all Players. Shapes that share a non-zero
# group don't collide with each other (pymunk semantics), so N agents in one
# World coexist non-interactively without per-agent group assignment.
PLAYER_GROUP = 99
```

Update the existing `on_goal` handler. Locate:

```python
    def on_goal(arbiter, space_, data):
        world_ref.complete_level()
        return False  # sensor
```

Replace with:

```python
    def on_goal(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        if player is not None:
            player.reached_goal = True
        world_ref.complete_level()
        return False  # sensor
```

- [ ] **Step 5: Modify `src/blueball/entities/player.py`**

Add the import alongside the existing `from .. import config`:

```python
from ..collision import PLAYER_GROUP
```

In `Player.__init__`, after `self.shape.collision_type = 1`, add:

```python
        self.shape.filter = pymunk.ShapeFilter(group=PLAYER_GROUP)
```

In `Player.__init__`, alongside the existing `self.dead = False` initialization, add:

```python
        self.reached_goal = False
```

Motion / physics tuning code is unchanged.

- [ ] **Step 6: Run targeted tests**

Run: `pytest -q tests/test_player.py tests/test_collision.py -v`
Expected: all pass.

- [ ] **Step 7: Run the full suite — no regression**

Run: `pytest -q`
Expected: all green. (`PlayScene._reset()` still constructs a single `Player`; `world.level_complete` still flips in PlayScene's normal flow.)

- [ ] **Step 8: Commit**

```bash
git add src/blueball/collision.py src/blueball/entities/player.py tests/test_player.py tests/test_collision.py
git commit -m "feat: shared PLAYER_GROUP collision filter + per-player reached_goal"
```

---

## Task 5: `FTNNAgent` in `agent.py`

**Goal:** Add a new `Agent` subclass that wraps a genome, runs each observation through an `FTNN`, and picks the argmax action. **Do not touch the `Observation` dataclass** (the parallel level-design branch owns it).

**Files:**
- Modify: `src/blueball/agent.py`
- Modify: `tests/test_ai_smoke.py` (append new tests)

**Acceptance Criteria:**
- [ ] `FTNNAgent(genome)` constructs without raising for any `genome` of shape `(GENOME_SIZE,)` and dtype `float32`.
- [ ] `agent.act(observation)` returns an `Action` enum value.
- [ ] Given two `FTNNAgent` instances built from the **same** genome, both return the **same** action for the same `Observation`.
- [ ] Given an `FTNNAgent` built from the all-zero genome, `act(observation)` returns `Action.IDLE` for any observation (zero output → argmax==0 by numpy's `argmax` tie-breaking → `Action(0) == Action.IDLE`).

**Verify:** `pytest -q tests/test_ai_smoke.py -v` → all pass

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 5: FTNNAgent -----

def test_ftnn_agent_returns_action_enum():
    from blueball.agent import FTNNAgent, Action
    from blueball.ai.genome import random_genome
    rng = np.random.default_rng(0)
    agent = FTNNAgent(random_genome(rng))
    action = agent.act(_make_obs())
    assert isinstance(action, Action)


def test_ftnn_agent_is_deterministic_for_same_genome():
    from blueball.agent import FTNNAgent
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(7))
    a1 = FTNNAgent(g)
    a2 = FTNNAgent(g)
    obs = _make_obs(vel=(50.0, -30.0), ang_vel=2.0, grounded=True)
    assert a1.act(obs) == a2.act(obs)


def test_ftnn_agent_all_zero_genome_returns_idle():
    from blueball.agent import FTNNAgent, Action
    from blueball.ai.ftnn import GENOME_SIZE
    agent = FTNNAgent(np.zeros(GENOME_SIZE, dtype=np.float32))
    assert agent.act(_make_obs()) == Action.IDLE
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: `ImportError: cannot import name 'FTNNAgent' from 'blueball.agent'`.

- [ ] **Step 3: Modify `src/blueball/agent.py`**

Append (do NOT modify `Observation`, `Action`, `Agent`, or `HumanAgent`):

```python
class FTNNAgent(Agent):
    """An Agent driven by a fixed-topology neural network (FTNN). Reads the
    observation, packs it into the 14-float input vector, runs it through
    the network, and returns the argmax Action.

    Imports of the `ai` package are lazy so that importing `agent` (which
    PlayScene and tests do) doesn't pull in the AI scaffolding transitively.
    """

    def __init__(self, genome: np.ndarray) -> None:
        from .ai.ftnn import FTNN
        from .ai.observation import observation_to_inputs

        self._net = FTNN(genome)
        self._to_inputs = observation_to_inputs

    def act(self, observation: Observation) -> Action:
        x = self._to_inputs(observation)
        y = self._net.forward(x)
        return Action(int(np.argmax(y)))
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite — no regression**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/agent.py tests/test_ai_smoke.py
git commit -m "feat: FTNNAgent — Agent subclass driven by a numpy FTNN"
```

---

## Task 6: Trainer (`evaluate` + `train`) + smoke test

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Goal:** Ship the trainer: a headless `evaluate(args)` that scores one genome on one level, and a `train(...)` that runs a full GA loop and returns a `TrainingResult`. Pin the entire scaffolding with a smoke test that runs 5 generations × pop_size 8 on `tutorial_hill.json` and asserts no-crash + well-formed fitnesses. This is the load-bearing pinning test for the slice — the user explicitly requested it.

**Files:**
- Create: `src/blueball/ai/trainer.py`
- Modify: `src/blueball/config.py` (append `TRAIN_*` + `GA_*` + `MAX_STEPS` defaults)
- Modify: `tests/test_ai_smoke.py` (append `test_evaluate_runs_one_genome_to_completion` + `test_trainer_smoke_5gens_no_crash`)

**Acceptance Criteria:**
- [ ] `config.MAX_STEPS == 3000`; `config.TRAIN_POP_SIZE == 80`; `config.TRAIN_GENERATIONS == 200`; `config.GA_MUTATION_RATE == 0.1`; `config.GA_MUTATION_SIGMA == 0.1`; `config.GA_TOURNAMENT_K == 4`; `config.GA_ELITISM == 1`.
- [ ] `evaluate((0, genome, DEFAULT_SEED, level_path, 200))` returns `(0, finite_float)` without raising.
- [ ] `train(pop_size=8, generations=5, level_path=tutorial_hill, max_steps=600, ga_seed=0)` returns a `TrainingResult` whose `history` has length 5, `best_genome.shape == (258,)` and dtype `float32`, every history entry's `best`/`mean`/`min` is a finite float, and `final_population` has 8 entries each of shape `(258,)`.
- [ ] `train()` defaults to `map_fn=map` (serial); does not import `multiprocessing` at module level.
- [ ] Two `train()` runs with the same `ga_seed` and `world_seed` produce identical `best_genome` arrays (full reproducibility).

**Verify:** `pytest -q tests/test_ai_smoke.py::test_trainer_smoke_5gens_no_crash tests/test_ai_smoke.py::test_evaluate_runs_one_genome_to_completion tests/test_ai_smoke.py::test_trainer_is_deterministic_under_same_seed -v` → all pass

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 6: Trainer + smoke -----

def _level_path():
    from pathlib import Path
    import blueball
    return Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"


def test_evaluate_runs_one_genome_to_completion():
    from blueball import config
    from blueball.ai.trainer import evaluate
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(0))
    idx, fit = evaluate((0, g, config.DEFAULT_SEED, _level_path(), 200))
    assert idx == 0
    assert np.isfinite(fit)


def test_trainer_smoke_5gens_no_crash():
    from blueball.ai.trainer import train
    from blueball.ai.ftnn import GENOME_SIZE
    result = train(
        pop_size=8,
        generations=5,
        level_path=_level_path(),
        max_steps=600,
        ga_seed=0,
    )
    assert len(result.history) == 5
    for entry in result.history:
        assert {"gen", "best", "mean", "min"} <= set(entry)
        assert np.isfinite(entry["best"])
        assert np.isfinite(entry["mean"])
        assert np.isfinite(entry["min"])
    assert result.best_genome.shape == (GENOME_SIZE,)
    assert result.best_genome.dtype == np.float32
    assert len(result.final_population) == 8
    for g in result.final_population:
        assert g.shape == (GENOME_SIZE,)


def test_trainer_is_deterministic_under_same_seed():
    from blueball.ai.trainer import train
    a = train(pop_size=6, generations=3, level_path=_level_path(),
              max_steps=300, ga_seed=42, world_seed=1)
    b = train(pop_size=6, generations=3, level_path=_level_path(),
              max_steps=300, ga_seed=42, world_seed=1)
    assert np.array_equal(a.best_genome, b.best_genome)
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: new tests fail (`ModuleNotFoundError: No module named 'blueball.ai.trainer'`).

- [ ] **Step 3: Modify `src/blueball/config.py`** — append a new "AI / GA training" section

Append at the end of the file:

```python
# AI / GA training
TRAIN_POP_SIZE      = 80      # spec default for real training
TRAIN_GENERATIONS   = 200     # spec default for real training
MAX_STEPS           = 3000    # per-evaluation timeout (~25s at PHYS_HZ=120)
GA_MUTATION_RATE    = 0.1
GA_MUTATION_SIGMA   = 0.1
GA_TOURNAMENT_K     = 4
GA_ELITISM          = 1
```

- [ ] **Step 4: Create `src/blueball/ai/trainer.py`**

```python
"""Headless GA trainer.

`evaluate((idx, genome, world_seed, level_path, max_steps))` is the worker
function; it builds a fresh headless World, registers collisions, loads
the level, spawns one Player(FTNNAgent(genome)) at the level's spawn,
steps physics at PHYS_DT up to max_steps (or until the player dies or
reaches the goal), and returns (idx, fitness).

`train(...)` is the generation loop; `map_fn` defaults to `map` (serial,
in-process). Real training callers pass `multiprocessing.Pool(...).imap`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .. import config
from ..agent import FTNNAgent
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..world import World
from .fitness import FitnessInputs, fitness
from .ftnn import GENOME_SIZE
from .ga import breed
from .genome import random_genome


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict]                       # per-gen stats
    best_genome: np.ndarray                   # shape (GENOME_SIZE,)
    final_population: list[np.ndarray]        # for follow-up runs / TrainScene re-entry


def evaluate(args: tuple) -> tuple[int, float]:
    """One genome -> one fitness. Picklable input/output so it works under
    multiprocessing.Pool. Args is (idx, genome, world_seed, level_path, max_steps).
    """
    idx, genome, world_seed, level_path, max_steps = args

    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    meta = load_level(level_path, world)

    spawn_x, spawn_y = float(meta.spawn[0]), float(meta.spawn[1])
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y))
    world.add_entity(player)

    steps = 0
    while steps < max_steps:
        # Step exactly one physics tick. Using world.step(PHYS_DT) here
        # would route through the accumulator and produce the same one
        # substep, but calling the pymunk space directly skips the
        # accumulator bookkeeping for the headless path.
        world.space.step(config.PHYS_DT)
        for entity in world.entities:
            upd = getattr(entity, "update", None)
            if upd is not None:
                upd(config.PHYS_DT)
        steps += 1
        if player.dead or player.reached_goal:
            break

    f = fitness(FitnessInputs(
        progress_x=float(player.body.position.x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(player.reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
    ))
    return idx, float(f)


def train(
    *,
    pop_size: int,
    generations: int,
    level_path: Path,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable[[Callable, Iterable], Iterable] = map,
    on_generation: Callable[[int, np.ndarray, list[np.ndarray]], None] | None = None,
) -> TrainingResult:
    """Run a GA training loop. Returns a TrainingResult.

    `map_fn` is the parallelism strategy: defaults to the builtin `map`
    (serial). For real training runs pass `multiprocessing.Pool(N).imap`.

    `ga_seed` controls all GA randomness (population init, mutation, crossover,
    tournament). `world_seed` controls physics. Two runs with the same
    `(ga_seed, world_seed)` produce byte-identical `best_genome`.
    """
    ga_rng = np.random.default_rng(ga_seed)
    population: list[np.ndarray] = [random_genome(ga_rng) for _ in range(pop_size)]
    history: list[dict] = []
    best_genome = population[0].copy()
    best_fitness = -np.inf

    for gen in range(generations):
        args_iter = [
            (i, population[i], world_seed, level_path, max_steps)
            for i in range(pop_size)
        ]
        results = list(map_fn(evaluate, args_iter))
        # Restore order: results may arrive out-of-order from a Pool.
        results.sort(key=lambda r: r[0])
        fitnesses = np.array([r[1] for r in results], dtype=np.float64)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = float(fitnesses[gen_best_idx])
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[gen_best_idx].copy()

        history.append({
            "gen": gen,
            "best": gen_best,
            "mean": float(fitnesses.mean()),
            "min": float(fitnesses.min()),
        })

        if on_generation is not None:
            on_generation(gen, best_genome, population)

        population = breed(
            population, fitnesses, ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )

    return TrainingResult(
        history=history,
        best_genome=best_genome,
        final_population=population,
    )
```

- [ ] **Step 5: Run smoke tests**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: all pass. Smoke test runtime budget: roughly 5–15 seconds depending on hardware (pop=8 × 5 gens × max 600 steps = at most 24,000 physics steps + 24,000 FTNN forward passes total).

If the smoke test exceeds 30s on the developer's machine, halve `max_steps` to 300 — the test exists to pin no-crash, not learning. Do NOT remove the determinism assertion.

- [ ] **Step 6: Run the full suite — no regression**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/trainer.py src/blueball/config.py tests/test_ai_smoke.py
git commit -m "feat: GA trainer + smoke test pinning the scaffolding"
```

---

## Task 7: `Camera.scale` field + `FreeCamera` subclass

**Goal:** Give the existing `Camera` a uniform scale factor that flows through `world_to_screen` (so renderer code requires zero changes), and add a `FreeCamera` subclass with arrow-key pan and `+`/`-` zoom for `TrainScene`. `FollowCamera` is unchanged (it inherits the default `scale=1.0` and never touches it).

**Files:**
- Modify: `src/blueball/camera.py`
- Modify: `tests/test_camera.py`

**Acceptance Criteria:**
- [ ] `Camera()` has `scale == 1.0` by default; `world_to_screen` multiplies the world-space offset by `scale` before adding the viewport center.
- [ ] `FollowCamera` continues to behave identically to its v1 behavior (its existing tests in `test_camera.py` still pass).
- [ ] `FreeCamera` exposes `handle_events(events)` (consumes `K_EQUALS`/`K_PLUS`/`K_KP_PLUS` for zoom-in, `K_MINUS`/`K_KP_MINUS` for zoom-out) and `update(keys_pressed, dt)` (consumes the four arrow keys for pan; speed `PAN_SPEED` in world units / sec, scaled by `1/scale` so panning feels consistent at any zoom).
- [ ] Zoom is clamped to `[ZOOM_MIN, ZOOM_MAX] == [0.1, 4.0]` and multiplied by `ZOOM_STEP = 1.1` per keypress.

**Verify:** `pytest -q tests/test_camera.py -v` → all existing + 3 new tests pass

**Steps:**

- [ ] **Step 1: Add failing tests to `tests/test_camera.py`**

Append (the file already imports `Camera` and `FollowCamera`):

```python
def test_camera_scale_defaults_to_one_and_affects_world_to_screen():
    cam = Camera(viewport_w=200, viewport_h=100)
    assert cam.scale == 1.0
    # At scale 1, world (0,0) with camera position (0,0) → viewport center.
    assert cam.world_to_screen((0.0, 0.0)) == (100.0, 50.0)
    cam.scale = 2.0
    # Doubling scale doubles the per-unit offset.
    assert cam.world_to_screen((10.0, 0.0)) == (120.0, 50.0)


def test_free_camera_zoom_keys_change_scale_within_bounds():
    import pygame
    from blueball.camera import FreeCamera
    cam = FreeCamera(viewport_w=200, viewport_h=100)
    initial = cam.scale
    events = [pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_EQUALS})]
    cam.handle_events(events)
    assert cam.scale == pytest.approx(initial * cam.ZOOM_STEP)
    # Repeatedly zoom out below the floor: scale must clamp at ZOOM_MIN.
    for _ in range(50):
        cam.handle_events([pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_MINUS})])
    assert cam.scale == pytest.approx(cam.ZOOM_MIN)


def test_free_camera_arrow_keys_pan_position():
    from blueball.camera import FreeCamera
    cam = FreeCamera(viewport_w=200, viewport_h=100)
    # Mock keys_pressed: right + down held.
    class _Keys:
        def __init__(self, held):
            self._held = set(held)
        def __getitem__(self, key):
            return key in self._held
    import pygame
    keys = _Keys({pygame.K_RIGHT, pygame.K_DOWN})
    cam.update(keys_pressed=keys, dt=1.0)   # 1 second of right+down
    px, py = cam.position
    assert px > 0      # camera moved right (positive world-x)
    assert py > 0      # camera moved down (positive world-y in our y-down coords)
```

If `tests/test_camera.py` does not already import `pytest`, add `import pytest` at the top.

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_camera.py -v`
Expected: new tests fail (`AttributeError: ... has no attribute 'scale'` and `ImportError: cannot import name 'FreeCamera'`).

- [ ] **Step 3: Modify `src/blueball/camera.py`**

Replace the file with:

```python
"""Camera — converts world coordinates to screen coordinates.

`Camera` is the pure-math base; `FollowCamera` tracks a target with a
smoothed lerp; `FreeCamera` (new) is driven by arrow keys + plus/minus
for developer use in TrainScene.
"""

from __future__ import annotations

import pygame

from . import config


class Camera:
    """A free camera; pure math, no PyGame dependencies for math itself.

    `scale` is a uniform zoom factor. world_to_screen multiplies the world
    offset by `scale` so that zoom flows through to the renderer with no
    renderer-side change. Primitive sizes drawn by the renderer stay in
    screen-space (debug-tool intent).
    """

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.position: tuple[float, float] = (0.0, 0.0)
        self.scale: float = 1.0

    def world_to_screen(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        wx, wy = world_xy
        cx, cy = self.position
        s = self.scale
        return ((wx - cx) * s + self.viewport_w / 2,
                (wy - cy) * s + self.viewport_h / 2)


class FollowCamera(Camera):
    """A camera that trails a target with a dead-zone and a lerp."""

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        super().__init__(viewport_w, viewport_h)
        self.dead_zone_w = config.CAMERA_DEAD_ZONE_W
        self.dead_zone_h = config.CAMERA_DEAD_ZONE_H
        self.lerp = config.CAMERA_LERP

    def update(self, target: tuple[float, float], dt: float) -> None:
        tx, ty = target
        cx, cy = self.position
        dx = tx - cx
        dy = ty - cy
        half_w = self.dead_zone_w / 2
        half_h = self.dead_zone_h / 2

        chase_x = 0.0
        chase_y = 0.0
        if dx > half_w:
            chase_x = dx - half_w
        elif dx < -half_w:
            chase_x = dx + half_w
        if dy > half_h:
            chase_y = dy - half_h
        elif dy < -half_h:
            chase_y = dy + half_h

        self.position = (cx + chase_x * self.lerp, cy + chase_y * self.lerp)


class FreeCamera(Camera):
    """A free camera driven by keyboard input. TrainScene uses this so the
    developer can pan around and zoom while the GA population trains.

    - Arrow keys pan in world units per second; pan speed is divided by
      the current scale so panning feels consistent at any zoom level.
    - `+` / `-` (top row or numpad) zoom in / out multiplicatively, clamped
      to [ZOOM_MIN, ZOOM_MAX].
    """

    PAN_SPEED = 500.0
    ZOOM_STEP = 1.1
    ZOOM_MIN = 0.1
    ZOOM_MAX = 4.0

    def handle_events(self, events) -> None:
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.scale = min(self.ZOOM_MAX, self.scale * self.ZOOM_STEP)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.scale = max(self.ZOOM_MIN, self.scale / self.ZOOM_STEP)

    def update(self, keys_pressed, dt: float) -> None:
        dx = 0.0
        dy = 0.0
        if keys_pressed[pygame.K_LEFT]:
            dx -= 1.0
        if keys_pressed[pygame.K_RIGHT]:
            dx += 1.0
        if keys_pressed[pygame.K_UP]:
            dy -= 1.0
        if keys_pressed[pygame.K_DOWN]:
            dy += 1.0
        if dx == 0.0 and dy == 0.0:
            return
        # Divide by scale so panning at a low zoom doesn't whip the camera around.
        step = self.PAN_SPEED * dt / max(self.scale, 1e-6)
        cx, cy = self.position
        self.position = (cx + dx * step, cy + dy * step)
```

- [ ] **Step 4: Run camera tests**

Run: `pytest -q tests/test_camera.py -v`
Expected: existing + 3 new pass.

- [ ] **Step 5: Run the full suite — no regression**

Run: `pytest -q`
Expected: all green. Renderer code reads `camera.world_to_screen` only; the new `scale` factor flows through automatically.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/camera.py tests/test_camera.py
git commit -m "feat: Camera.scale + FreeCamera with pan/zoom"
```

---

## Task 8: `TrainScene` + `train_main.py` entry script

**Goal:** A new in-process scene that runs the GA trainer and renders the visible population live on `tutorial_hill.json`. Free-camera pan/zoom; minimal HUD; rebuilt fresh `World` between generations. Plus a small `train_main.py` script at the repo root so the developer can launch it without touching `main.py`.

**Files:**
- Create: `src/blueball/scenes/train.py`
- Create: `train_main.py` (at repo root, alongside `main.py`)
- Modify: `tests/test_ai_smoke.py` (append a single TrainScene construction test)

**Acceptance Criteria:**
- [ ] `TrainScene(screen, level_path)` constructs without raising; its `world` is a `World`, and `len(scene._players) == min(scene.n_visible, scene.pop_size)`.
- [ ] All scene players share `PLAYER_GROUP` and have `agent` instances of `FTNNAgent` (one per visible genome).
- [ ] Calling `scene.update(frame_dt=1/60)` repeatedly does not raise, and after sufficient ticks (`max_steps`) the scene advances to the next generation: `scene.current_gen` increments and `scene.world` is a different `World` instance.
- [ ] `handle_events([pygame.event.Event(pygame.QUIT, {})])` returns `None`; an `ESCAPE` keydown returns `None`; other events return `self`.
- [ ] Running `python train_main.py` opens a window titled "Blue Ball — Train" and the scene runs (manual verification step at the end of the task).

**Verify:** `pytest -q tests/test_ai_smoke.py -v` (TrainScene construction + step test pass) → then manual playtest: `python train_main.py` shows the population balls moving and the developer can pan with arrows + zoom with `+`/`-`.

**Steps:**

- [ ] **Step 1: Append failing tests to `tests/test_ai_smoke.py`**

Append:

```python
# ----- Task 8: TrainScene -----

@pytest.fixture
def headless_pygame():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.display.quit()


def test_train_scene_constructs_and_steps(headless_pygame):
    """TrainScene builds without raising, owns N FTNNAgent-driven Players,
    and a single update() tick does not crash."""
    from blueball.scenes.train import TrainScene
    from blueball.agent import FTNNAgent
    from blueball import collision
    scene = TrainScene(
        headless_pygame,
        _level_path(),
        pop_size=4,
        n_visible=4,
        generations=2,
        max_steps=60,
    )
    assert len(scene._players) == 4
    for p in scene._players:
        assert isinstance(p.agent, FTNNAgent)
        assert p.shape.filter.group == collision.PLAYER_GROUP
    # Multiple ticks should not raise. Don't rely on hitting a gen boundary here
    # because frame_dt at 1/60 only advances 2 physics substeps per call.
    for _ in range(10):
        scene.update(1 / 60)


def test_train_scene_advances_generation_after_max_steps(headless_pygame):
    """After max_steps elapsed gen ticks, scene rebuilds the world for gen 1."""
    from blueball.scenes.train import TrainScene
    scene = TrainScene(
        headless_pygame,
        _level_path(),
        pop_size=4,
        n_visible=4,
        generations=3,
        max_steps=20,    # short enough to roll a generation in a handful of frames
    )
    initial_world = scene.world
    # Each scene.update at 1/60s advances PHYS_HZ/60 = 2 physics ticks. So
    # max_steps=20 advances in ~10 scene-updates. We give it a margin.
    for _ in range(40):
        scene.update(1 / 60)
        if scene.current_gen >= 1:
            break
    assert scene.current_gen >= 1
    assert scene.world is not initial_world
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: new tests fail (`ModuleNotFoundError: No module named 'blueball.scenes.train'`).

- [ ] **Step 3: Create `src/blueball/scenes/train.py`**

```python
"""TrainScene — runs the GA trainer in-process and renders the live population.

Owns one shared World with N visible Player entities, each driven by an
FTNNAgent. All players share PLAYER_GROUP so they don't collide with each
other. At each generation boundary the World is rebuilt fresh (cheap; this
keeps reset semantics dead simple).

The trainer's evaluation happens INSIDE this scene's update loop — we don't
call ai.trainer.train() because we want to render mid-generation. Instead
the scene runs its own per-tick generation accumulator using the same
fitness/breed primitives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .. import collision, config
from ..agent import FTNNAgent
from ..ai.fitness import FitnessInputs, fitness
from ..ai.ftnn import GENOME_SIZE
from ..ai.ga import breed
from ..ai.genome import random_genome
from ..camera import FreeCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


class TrainScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        level_path: Path,
        *,
        pop_size: int = config.TRAIN_POP_SIZE,
        n_visible: int = 16,
        generations: int = config.TRAIN_GENERATIONS,
        ga_seed: int = 0,
        world_seed: int = config.DEFAULT_SEED,
        max_steps: int = config.MAX_STEPS,
    ) -> None:
        self.screen = screen
        self.level_path = level_path
        self.pop_size = pop_size
        self.n_visible = min(n_visible, pop_size)
        self.generations = generations
        self.ga_seed = ga_seed
        self.world_seed = world_seed
        self.max_steps = max_steps

        pygame.display.set_caption("Blue Ball — Train")
        self.camera = FreeCamera(screen.get_width(), screen.get_height())
        self.renderer = Renderer(screen, self.camera)
        self._font = pygame.font.Font(None, 20)

        self._ga_rng = np.random.default_rng(ga_seed)
        self.population: list[np.ndarray] = [
            random_genome(self._ga_rng) for _ in range(pop_size)
        ]
        self.current_gen = 0
        self.best_fitness = float("-inf")
        self.best_mean = 0.0

        self._build_world_for_current_gen()

    # ---- Generation lifecycle ----

    def _build_world_for_current_gen(self) -> None:
        """Construct a fresh World, load the level, spawn n_visible Players
        driven by the first n_visible genomes of the current population."""
        self.world = World(seed=self.world_seed)
        register_collisions(self.world.space, world_ref=self.world)
        self.level_meta = load_level(self.level_path, self.world)
        self._spawn_xy = (float(self.level_meta.spawn[0]),
                          float(self.level_meta.spawn[1]))
        self._players: list[Player] = []
        for i in range(self.n_visible):
            agent = FTNNAgent(self.population[i])
            p = Player(agent=agent, spawn_xy=self._spawn_xy)
            self.world.add_entity(p)
            self._players.append(p)
        # Snap camera to the spawn so the developer sees something on gen start.
        self.camera.position = self._spawn_xy
        self._gen_steps = 0

    def _all_visible_done(self) -> bool:
        return all(p.dead or p.reached_goal for p in self._players)

    def _score_visible_players(self) -> np.ndarray:
        """Compute fitness for the n_visible players. The remaining
        (pop_size - n_visible) population members get a baseline-zero fitness;
        a future task can evaluate them headless if we want them to compete."""
        fits = np.zeros(self.pop_size, dtype=np.float64)
        for i, p in enumerate(self._players):
            fits[i] = fitness(FitnessInputs(
                progress_x=float(p.body.position.x - self._spawn_xy[0]),
                collectibles=int(p.collectibles_collected),
                reached_goal=bool(p.reached_goal),
                died=bool(p.dead),
                steps_taken=self._gen_steps,
            ))
        return fits

    def _advance_generation(self) -> None:
        fits = self._score_visible_players()
        self.best_fitness = max(self.best_fitness, float(fits.max()))
        self.best_mean = float(fits[:self.n_visible].mean())
        self.population = breed(
            self.population, fits, self._ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )
        self.current_gen += 1
        if self.current_gen < self.generations:
            self._build_world_for_current_gen()

    # ---- Scene API ----

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
        self.camera.handle_events(events)
        return self

    def update(self, frame_dt: float) -> None:
        if self.current_gen >= self.generations:
            return
        self.renderer.begin_frame(self.world)
        keys = pygame.key.get_pressed()
        self.camera.update(keys_pressed=keys, dt=frame_dt)
        # Drive physics via the world's accumulator (real-time pace) so the
        # visualization plays back at the same rate the game does.
        substeps = self.world.step(frame_dt)
        self._gen_steps += substeps
        if self._gen_steps >= self.max_steps or self._all_visible_done():
            self._advance_generation()

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        live = sum(1 for p in self._players if not (p.dead or p.reached_goal))
        text = (
            f"gen {self.current_gen + 1}/{self.generations}  "
            f"best {self.best_fitness:.1f}  mean {self.best_mean:.1f}  "
            f"live {live}/{self.n_visible}"
        )
        surf = self._font.render(text, True, (255, 255, 255))
        self.screen.blit(surf, (12, 12))
```

- [ ] **Step 4: Create `train_main.py` at the repo root**

```python
"""Entry script for the GA training scene.

Kept separate from main.py so the play loop entry point stays minimal.
Run with:  python train_main.py
"""

import sys
from pathlib import Path

import pygame

from blueball import config
from blueball.scenes.train import TrainScene


def main() -> int:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    level_path = Path(__file__).parent / "src" / "blueball" / "levels" / "tutorial_hill.json"
    scene = TrainScene(screen, level_path)

    while scene is not None:
        events = pygame.event.get()
        scene = scene.handle_events(events)
        if scene is None:
            break
        frame_dt = clock.tick(config.TARGET_FPS) / 1000.0
        scene.update(frame_dt)
        scene.draw()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run TrainScene tests**

Run: `pytest -q tests/test_ai_smoke.py -v`
Expected: all pass, including the new TrainScene cases.

- [ ] **Step 6: Run the full suite — no regression**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 7: Manual playtest**

Run: `python train_main.py`

Expected:
- A 1280×720 window opens titled "Blue Ball — Train".
- 16 blue balls appear at the level's spawn point and start moving (poorly — they're random networks). The level's ground, spikes, boost pad, etc. render.
- A small HUD strip in the top-left shows `gen 1/200 best X.X mean Y.Y live N/16` and updates as balls die or reach the goal.
- Arrow keys pan the camera.
- `+` and `-` zoom in / out (clamped between 0.1× and 4×).
- After a generation finishes (max_steps elapsed or all balls done), the population resets visually and the gen counter increments.
- `ESC` closes the window.

Don't expect the agents to learn — `Observation.rays` is still zeros until the level-design branch's enrichment lands. The point of this playtest is the scaffolding works.

- [ ] **Step 8: Commit**

```bash
git add src/blueball/scenes/train.py train_main.py tests/test_ai_smoke.py
git commit -m "feat: TrainScene + train_main.py entry script"
```

---

## Self-review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| `ai/__init__.py` + package | Task 0 |
| `ai/ftnn.py` (FTNN class + topology constants + GENOME_SIZE=258) | Task 0 |
| `ai/genome.py` (random_genome) | Task 0 |
| `ai/ga.py` (mutate, crossover, tournament_select, breed) | Task 1 |
| `ai/observation.py` (observation_to_inputs, RAY_COUNT, shape assertion) | Task 2 |
| `ai/fitness.py` (FitnessInputs + fitness function) | Task 3 |
| Per-player `reached_goal` flag + `PLAYER_GROUP` filter | Task 4 |
| Updated `on_goal` collision handler | Task 4 |
| `FTNNAgent` subclass added to `agent.py` (Observation untouched) | Task 5 |
| `ai/trainer.py` (evaluate + train + TrainingResult) | Task 6 |
| `config.py` AI/GA constants (TRAIN_*, GA_*, MAX_STEPS) | Task 6 |
| `Camera.scale` + `FreeCamera` (pan + zoom + clamps) | Task 7 |
| `TrainScene` (shared World, N players, gen lifecycle, HUD) | Task 8 |
| `train_main.py` entry script (does not touch main.py) | Task 8 |
| Smoke test (no-crash + shape/range sanity) | Task 6 |
| Determinism test (same ga_seed + world_seed → same best_genome) | Task 6 |

No spec section is unaddressed.

**Type / name consistency check:**
- `GENOME_SIZE` defined in `ai/ftnn.py`, re-exported from `ai/genome.py`, referenced in tests across Tasks 0, 5, 6, 8 — all consistent (258).
- `RAY_COUNT = 8` (Task 2) ↔ `FTNN_INPUTS = 14` (Task 0): documented coupling, asserted at runtime in `observation_to_inputs`.
- `PLAYER_GROUP` defined in `collision.py` (Task 4), imported in `entities/player.py` (Task 4), referenced in `tests/test_player.py` (Task 4) and `tests/test_ai_smoke.py` (Task 8) — consistent (99).
- `Player.reached_goal` ↔ `on_goal` handler ↔ `FitnessInputs.reached_goal` — consistent across Tasks 3, 4, 6.
- `train(map_fn=map)` default ↔ `multiprocessing` opt-in: documented in trainer module docstring, never imported at module level. ✓
- `TrainScene` ↔ `train_main.py` ↔ window caption "Blue Ball — Train" — consistent in Task 8.

**Placeholder scan:** none. Every step contains the actual code or command an engineer needs.
