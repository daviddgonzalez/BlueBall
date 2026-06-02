"""ChunkSampler — deterministic procedural chunk emitter."""

from __future__ import annotations

import math
import random
from typing import Iterator

# Importing the chunks package registers every chunk type
from . import chunks  # noqa: F401
from .chunks.base import CHUNK_REGISTRY, Chunk


class ChunkSampler:
    def __init__(
        self,
        seed: int,
        target_chunks: int = 500,
        ramp_per_chunk: float = 0.02,
        sigma: float = 1.0,
        checkpoint_every: int = 25,
        emit_checkpoints: bool = True,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.target = target_chunks
        self.emit_checkpoints = emit_checkpoints
        # Difficulty climbs ~3x faster than the original 0.006 so escalation
        # (and variety) arrives within a typical run instead of after hundreds
        # of chunks. A wider sigma (1.0 vs 0.7) mixes adjacent difficulty tiers
        # at any given progress so segments don't feel uniform.
        self.ramp = ramp_per_chunk
        self.sigma = sigma
        self.checkpoint_every = checkpoint_every
        self.progress = 0
        # Last obstacle type emitted, to suppress back-to-back duplicates.
        self._last_obstacle_type: str | None = None
        # Stable-sorted pool of sampler-included chunks. Sorting by name removes
        # dict-ordering as a determinism risk.
        self._pool: list[tuple[str, type[Chunk]]] = sorted(
            ((name, cls) for name, cls in CHUNK_REGISTRY.items() if cls.sampler_include),
            key=lambda item: item[0],
        )

    def __iter__(self) -> Iterator[dict]:
        while True:
            entry = self.emit_next()
            if entry is None:
                return
            yield entry

    def emit_next(self) -> dict | None:
        if self.progress > self.target:
            return None
        if self.progress == self.target:
            self.progress += 1  # advance past so subsequent calls return None
            return {"type": "goal"}
        # Checkpoint every N (but not at index 0)
        if self.emit_checkpoints and self.progress > 0 and self.progress % self.checkpoint_every == 0:
            cid = self.progress // self.checkpoint_every
            self.progress += 1
            return {"type": "checkpoint", "id": cid}
        # Weighted pick by closeness to target difficulty
        target_diff = min(3.0, self.progress * self.ramp)
        weights = [
            math.exp(-((cls.difficulty - target_diff) ** 2) / (2 * self.sigma ** 2))
            for _, cls in self._pool
        ]
        idx = self._weighted_pick(weights)
        name, cls = self._pool[idx]
        # Suppress an immediate repeat of the same obstacle — the single biggest
        # source of the "wall of identical chunks" feel — by zeroing its weight
        # and re-picking once. Still fully deterministic (one extra rng draw).
        if name == self._last_obstacle_type and len(self._pool) > 1:
            weights[idx] = 0.0
            idx = self._weighted_pick(weights)
            name, cls = self._pool[idx]
        self._last_obstacle_type = name
        params = cls.random_params(self.rng)
        self.progress += 1
        return {"type": name, **params}

    def _weighted_pick(self, weights: list[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1
