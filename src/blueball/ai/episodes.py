"""Multi-episode training: episode specs, fitness aggregation, per-level
normalization, and episode-list constructors.

A genome is scored across a *list* of EpisodeSpecs; each per-episode raw
fitness is divided by that episode's `norm` and the results are aggregated as
mean - lam*std (or min, depending on mode). A single episode aggregates to
itself (population std 0), so single-episode training reproduces the
pre-multi-episode behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ..collision import register as register_collisions
from ..levels.loader import load_level
from ..world import World
from .. import config


@dataclass(frozen=True)
class EpisodeSpec:
    """One evaluation episode. Picklable so it survives multiprocessing.Pool."""

    kind: str                 # "infinite" | "static" | "gym"
    seed: int                 # sampler_seed for infinite/gym; ignored for static
    level_path: str | None    # for static (str so it pickles cleanly)
    world_seed: int
    max_steps: int
    norm: float = 1.0         # divisor applied to this episode's raw fitness
    abilities: tuple[str, ...] = ()  # gym: granted ability names; () elsewhere


def aggregate_fitness(scores: Sequence[float], lam: float, mode: str = "mean_std") -> float:
    """Combine per-episode fitnesses into one selection score.

    mode="mean_std": mean - lam*std (population std, ddof=0) — the default.
    mode="min":      min(scores) — the worst-case objective; lam is ignored.
                     Forces selection to raise the weakest episode's score.

    For a single score both modes return that score exactly (population std is
    0), which keeps single-episode training numerically identical. Empty input
    is a programming error.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("aggregate_fitness requires at least one score")
    if mode == "min":
        return float(arr.min())
    if mode == "mean_std":
        return float(arr.mean() - lam * arr.std())
    raise ValueError(f"unknown aggregate mode {mode!r}")


_GOAL_NAME = "Goal"
_KEY_NAME = "Key"
_COLLECTIBLE_NAME = "Collectible"


def compute_level_par(level: Union[str, Path, dict]) -> float:
    """Reference 'fully-solved' fitness for a static level, used to normalize it
    so big levels don't dominate multi-level selection. Built once per level
    (never inside the eval loop). Same weights as ai/fitness.py:

        par = total_width * (1 + GOAL_MULT*has_goal) + 100*keys + 50*collectibles

    The goal term is GOAL_MULT*total_width (matching the width-scaled goal bonus
    in ai/fitness.py), so a solved agent's raw fitness ~= par -> norm ~= 1.0 and
    a full-traversal-but-no-goal run lands at ~= 1/(1+GOAL_MULT).

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
        + config.GOAL_MULT * float(meta.total_width) * (1.0 if _GOAL_NAME in names else 0.0)
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


def gym_episodes(seeds: Sequence[int], world_seed: int, max_steps: int,
                 abilities: Sequence[str]) -> list[EpisodeSpec]:
    """One completion-gym EpisodeSpec per chain seed. `abilities` is the
    granted ability-name set, shared across all seeds (norm=1.0: all gym
    chains share the same reward scale)."""
    ab = tuple(str(a) for a in abilities)
    return [
        EpisodeSpec(kind="gym", seed=int(s), level_path=None,
                    world_seed=int(world_seed), max_steps=int(max_steps),
                    abilities=ab)
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


def static_episodes(level_paths: Sequence[str], world_seed: int, max_steps: int,
                    abilities: Sequence[str] = ()) -> list[EpisodeSpec]:
    """One static EpisodeSpec per level, each normalized by its level par.

    `abilities` (default ()) is a granted ability-name set carried on every
    episode and unioned with the level's own starting_abilities at eval time;
    the empty default keeps existing `train levels` byte-identical."""
    ab = tuple(str(a) for a in abilities)
    return [
        EpisodeSpec(kind="static", seed=0, level_path=str(p),
                    world_seed=int(world_seed), max_steps=int(max_steps),
                    norm=compute_level_par(p), abilities=ab)
        for p in level_paths
    ]


def mixed_episodes(infinite_seeds: Sequence[int], level_names: Sequence[str],
                   gym_seeds: Sequence[int], world_seed: int, max_steps: int,
                   abilities: Sequence[str] = ()) -> list[EpisodeSpec]:
    """The generalist objective: infinite + static + gym EpisodeSpecs, IN THAT
    ORDER. `abilities` (e.g. ("double_jump",)) is set on all three kinds so the
    generalist trains double-jump-capable, not only where a level's JSON declares
    it. NOTE: static (via `evaluate` union with the level's starting_abilities)
    and gym (via `evaluate_gym`) actually CONSUME the granted abilities today;
    the infinite path carries them as metadata but `evaluate_infinite` does not
    yet read them, so infinite trains single-jump until Track D's abilities flag
    lands and the infinite dispatch threads `ep.abilities` through.

    Cross-kind normalization: static keeps its per-level par norm; infinite and
    gym are given `GENERALIST_INFINITE_PAR` / `GENERALIST_GYM_PAR` divisors (NOT
    1.0) so all three kinds land at ~0-1 ("fraction of a competent run"). Without
    this, the `min` objective is driven entirely by the worst static level
    (infinite/gym raw fitness is hundreds-to-thousands, never the min) and `mean`
    by infinite/gym — neither balances the three kinds. The single-mode
    `infinite_episodes`/`gym_episodes` constructors keep norm=1.0; the par
    divisor is applied here (via `replace`) only for the mixed objective."""
    ab = tuple(str(a) for a in abilities)
    inf = [replace(ep, abilities=ab, norm=config.GENERALIST_INFINITE_PAR)
           for ep in infinite_episodes(infinite_seeds, world_seed, max_steps)]
    static = static_episodes(resolve_level_paths(level_names), world_seed,
                             max_steps, abilities=ab)
    gym = [replace(ep, abilities=ab, norm=config.GENERALIST_GYM_PAR)
           for ep in gym_episodes(gym_seeds, world_seed, max_steps, ab)]
    return inf + static + gym
