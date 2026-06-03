"""Trained-genome persistence.

Layout (one folder per run, never overwritten):

    genomes/<key>_w<world_seed>_<timestamp>/
        run.json          # seeds, pop/gen counts, fitness history
        best_gen000.npy   # running best genome after generation 0
        best_gen001.npy
        ...
        final_best.npy    # best genome of the whole run

`<key>` is `inf<sampler_seed>` for Infinite Run or the level name for a static
level. Genomes are committed to the repo (golden agents travel with the code),
so keep run folders intentional — each `train(save_dir=...)` call makes one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GENOMES_ROOT = Path("genomes")


def run_dir_name(
    *,
    world_seed: int,
    timestamp: str,
    infinite_seed: int | None = None,
    level_name: str | None = None,
) -> str:
    """Build the per-run folder name from the seeds + a timestamp string."""
    key = f"inf{infinite_seed}" if infinite_seed is not None else (level_name or "level")
    return f"{key}_w{world_seed}_{timestamp}"


class TrainingRunWriter:
    """Writes per-generation and final genome snapshots into one run folder."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_generation(self, gen: int, genome: np.ndarray) -> Path:
        """Snapshot the best genome after generation *gen*. Returns the path."""
        path = self.run_dir / f"best_gen{gen:03d}.npy"
        np.save(path, genome)
        return path

    def finalize(self, best_genome: np.ndarray, meta: dict) -> None:
        """Write the run's final best genome and a run.json metadata sidecar."""
        np.save(self.run_dir / "final_best.npy", best_genome)
        (self.run_dir / "run.json").write_text(json.dumps(meta, indent=2))
