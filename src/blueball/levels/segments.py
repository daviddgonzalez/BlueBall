"""Completion-gym segment templates.

A *segment* is a small, self-contained, solvable unit ending in a goal, built
from the existing completion chunks on the flat GROUND_Y baseline. Templates
mirror the loader's calling convention — `chunk.build(world, x_offset=...)` with
no base_y — so the per-chunk base_y signature inconsistency never bites.
Segments are the gym's analogue of Infinite Run's chunks.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

from ..abilities import Ability
from .. import config
from .chunks.flat import Flat
from .chunks.key import KeyChunk
from .chunks.door import DoorChunk
from .chunks.goal import GoalChunk
from .chunks.box_lava_gap import BoxLavaGap


class SegmentTemplate:
    """Base class. `build` lays chunks left-to-right from `x_offset` on the
    GROUND_Y baseline and returns the segment's total width in px."""

    tier: int = 0
    min_abilities: frozenset[Ability] = frozenset()

    @classmethod
    def random(cls, rng: random.Random) -> "SegmentTemplate":
        """Instantiate with any per-segment randomization. Default: no params."""
        return cls()

    def build(self, world, x_offset: float) -> float:
        raise NotImplementedError

    @staticmethod
    def _chunk(chunk, world, x_offset: float) -> float:
        # Mirror loader.py: no base_y kwarg; every chunk defaults to GROUND_Y.
        return chunk.build(world, x_offset=x_offset)


class GoalSegment(SegmentTemplate):
    """Tier 0 — flat approach then a goal. The 'run to the goal' lesson."""

    tier = 0
    min_abilities = frozenset()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=4), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class KeyDoorGoalSegment(SegmentTemplate):
    """Tier 1 — collect a key, pass the door it unlocks, reach the goal. The key
    sits low (y_offset=40) so a rolling ball (radius 16) collects it without
    jumping; the door chunk seals the gap above the doorway, so the key is the
    only way through."""

    tier = 1
    min_abilities = frozenset()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class BoxLavaSegment(SegmentTemplate):
    """Tier 2 — shove the box into the lava pit as a stepping stone, then reach
    the goal. The pit is wide (20-24 tiles, matching campaign maze.json's
    pit_tiles=24) so a DOUBLE_JUMP agent cannot vault it without the box — the
    box-push is mandatory. Requires DOUBLE_JUMP (granted by default)."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, pit_tiles: int = 22) -> None:
        self.pit_tiles = pit_tiles

    @classmethod
    def random(cls, rng: random.Random) -> "BoxLavaSegment":
        return cls(pit_tiles=rng.randint(20, 24))

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=self.pit_tiles), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class KeyDoorBoxLavaSegment(SegmentTemplate):
    """Tier 3 — unlock a door, then cross a box/lava pit, then the goal.
    The pit uses vault-proof pit_tiles=24 (matching campaign maze.json) so the
    box-push is mandatory even with DOUBLE_JUMP granted."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=24), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


# Registry that the sampler draws from.
SEGMENT_TEMPLATES: list[type[SegmentTemplate]] = [
    GoalSegment,
    KeyDoorGoalSegment,
    BoxLavaSegment,
    KeyDoorBoxLavaSegment,
]


class SegmentSampler:
    """Deterministic, depth-ramped segment emitter. Tier target rises with
    depth; templates are Gaussian-weighted by closeness to the target tier and
    immediate repeats are suppressed. Only templates whose `min_abilities` are
    all granted are eligible. Mirrors levels/sampler.py:ChunkSampler."""

    def __init__(
        self,
        seed: int,
        granted_abilities: frozenset[Ability],
        *,
        ramp_per_segment: float = config.GYM_RAMP_PER_SEGMENT,
        sigma: float = config.GYM_SIGMA,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.ramp = ramp_per_segment
        self.sigma = sigma
        self.depth = 0
        self._last_name: str | None = None
        granted = frozenset(granted_abilities)
        self._pool = sorted(
            (t for t in SEGMENT_TEMPLATES if t.min_abilities <= granted),
            key=lambda t: t.__name__,
        )
        if not self._pool:
            raise ValueError(
                "no segment templates are solvable under the granted abilities"
            )
        self._max_tier = max(t.tier for t in self._pool)

    def emit_next(self) -> SegmentTemplate:
        target = min(float(self._max_tier), self.depth * self.ramp)
        weights = [
            math.exp(-((t.tier - target) ** 2) / (2 * self.sigma ** 2))
            for t in self._pool
        ]
        idx = self._weighted_pick(weights)
        if self._pool[idx].__name__ == self._last_name and len(self._pool) > 1:
            weights[idx] = 0.0
            idx = self._weighted_pick(weights)
        cls = self._pool[idx]
        self._last_name = cls.__name__
        self.depth += 1
        return cls.random(self.rng)

    def __iter__(self) -> Iterator[SegmentTemplate]:
        while True:
            yield self.emit_next()

    def _weighted_pick(self, weights: list[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1
