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
from .chunks.boost_pad import BoostPadChunk


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
    """Tier 2 — shove the box across the lava pit as a stepping stone, then
    reach the goal. A boost_pad (width 3, multiplier 1.8) immediately before the
    pit drives the box far across the low-friction pit floor, splitting the wide
    24-tile pit into two jumpable gaps (this mirrors campaign maze.json, where a
    boost_pad precedes a pit_tiles=24 box_lava_gap). The pit is fixed at the
    campaign-proven, vault-proof width: even with the boost, a DOUBLE_JUMP agent
    cannot vault the bare pit, so the box-push is mandatory. Requires
    DOUBLE_JUMP (granted by default)."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, pit_tiles: int = 24) -> None:
        self.pit_tiles = pit_tiles

    @classmethod
    def random(cls, rng: random.Random) -> "BoxLavaSegment":
        # Fixed at the campaign-proven, vault-proof width. Randomizing narrower
        # risks re-introducing a vaultable (box-optional) pit.
        return cls(pit_tiles=24)

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoostPadChunk(width_tiles=3, multiplier=1.8), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=self.pit_tiles), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


_BOX_STEP_PIT_TILES = 24
_BOX_STEP_DEPTH = 72
_BOX_STEP_BOX = 64


class BoxStepSegment(SegmentTemplate):
    """Tier 2 — curriculum stage 1: a single jump ONTO a pre-placed box, then a
    single jump OFF it to the goal. The box is centered in a vault-proof lava pit
    (a bare DOUBLE_JUMP cannot clear it), so the box is the only way across; but
    it sits exactly where a natural single jump lands, so the lesson is gentle."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "BoxStepSegment":
        return cls()  # fixed, probe-tuned geometry

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(Flat(width_tiles=3), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=_BOX_STEP_PIT_TILES,
                                    depth=_BOX_STEP_DEPTH,
                                    box_size=_BOX_STEP_BOX,
                                    box_frac=0.5), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


_BOX_LEAP_PIT_TILES = 40
_BOX_LEAP_DEPTH = 96
_BOX_LEAP_BOX = 96
_BOX_LEAP_FRAC = 0.55


class BoxLeapSegment(SegmentTemplate):
    """Tier 3 — curriculum stage 2: a DOUBLE jump ONTO a bigger pre-placed box,
    then a double jump OFF it to the goal. The pit is wider and the box larger
    than stage 1, placed where a natural max double-jump lands: a SINGLE jump
    cannot reach it (that's the stage-1→stage-2 discriminator), and a bare
    DOUBLE_JUMP cannot vault the box-removed pit, so the box-leap is the only way
    across. Probe-tuned (probes/tune_box_leap.py) to a robust SAFE cell: 24/24
    DoubleStepAgent combos solve, vault-proof, not single-jumpable."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "BoxLeapSegment":
        return cls()  # fixed, probe-tuned geometry

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(Flat(width_tiles=3), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=_BOX_LEAP_PIT_TILES,
                                    depth=_BOX_LEAP_DEPTH,
                                    box_size=_BOX_LEAP_BOX,
                                    box_frac=_BOX_LEAP_FRAC), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class KeyDoorBoxLavaSegment(SegmentTemplate):
    """Tier 3 — unlock a door, then cross a box/lava pit, then the goal. A
    boost_pad (width 3, multiplier 1.8) immediately before the pit drives the
    box far across the low-friction pit floor, splitting the wide 24-tile pit
    into two jumpable gaps (mirroring campaign maze.json). The pit uses the
    vault-proof pit_tiles=24 width so the box-push is mandatory even with the
    boost and DOUBLE_JUMP granted."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(BoostPadChunk(width_tiles=3, multiplier=1.8), world, x)
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
