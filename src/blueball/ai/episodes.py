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
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("aggregate_fitness requires at least one score")
    return float(arr.mean() - lam * arr.std())
