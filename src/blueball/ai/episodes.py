"""Multi-episode training: episode specs, fitness aggregation, per-level
normalization, and episode-list constructors.

A genome is scored across a *list* of EpisodeSpecs; each per-episode raw
fitness is divided by that episode's `norm` and the results are aggregated as
mean - lam*std. A single episode aggregates to itself (population std 0), so
single-episode training reproduces the pre-multi-episode behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ..collision import register as register_collisions
from ..levels.loader import load_level
from ..world import World


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
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("aggregate_fitness requires at least one score")
    return float(arr.mean() - lam * arr.std())


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
        s = int(rng.integers(0, 2**31))
        if s not in seeds:
            seeds.append(s)
    return seeds


def infinite_episodes(seeds: Sequence[int], world_seed: int, max_steps: int) -> list[EpisodeSpec]:
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


def resolve_level_paths(names: Sequence[str]) -> list[str]:
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


def static_episodes(level_paths: Sequence[str], world_seed: int, max_steps: int) -> list[EpisodeSpec]:
    """One static EpisodeSpec per level, each normalized by its level par."""
    return [
        EpisodeSpec(kind="static", seed=0, level_path=str(p),
                    world_seed=int(world_seed), max_steps=int(max_steps),
                    norm=compute_level_par(p))
        for p in level_paths
    ]
