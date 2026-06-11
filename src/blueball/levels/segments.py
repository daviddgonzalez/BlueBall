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


# Stage-3 box-lava cell. Pit=24 is the MINIMUM width that is vault-proof against
# the strongest cheese (the apex-fired max double jump, DoubleJumpVaultAgent):
# that maneuver flies ~990px and clears pit<=23 (it lands past the far edge of a
# 22- or 23-tile pit with the box removed), but NOT a 24-tile / 768px pit. The
# earlier "pit=22 is vault-proof" claim came from the WEAKER RIGHT_JUMP-spam
# agent and was false confidence; 24 also matches campaign maze.json's pit_tiles.
# depth=72 (== box=64 floor clearance) keeps the box-on-floor a clean step.
_BOX_LAVA_PIT_TILES = 24
_BOX_LAVA_DEPTH = 72


class BoxLavaSegment(SegmentTemplate):
    """Tier 2 — curriculum stage 3 (EXPERT): PUSH the box yourself off the
    approach ledge into the lava pit, then box-step across (near ledge -> box
    top -> far ledge) to the goal. Unlike stages 1-2 the box is NOT pre-placed
    (box_frac=None = it starts on the approach ledge), so the shove is the whole
    lesson: the player must drive the box into the pit before any crossing is
    possible.

    The pit is fixed at the vault-proof cell (pit=24, depth=72, box=64): even the
    strongest cheese — the apex-fired MAX double jump — cannot vault the
    box-removed pit, so the box-push is mandatory. This is human-solvable but the
    shove-then-step maneuver is too precise to script reliably — stage 3 is the
    EXPERT tier and has NO scripted solvable-by-agent guarantee (only the
    vault-proof + composition invariants are tested). Requires DOUBLE_JUMP
    (granted by default)."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def __init__(self, pit_tiles: int = _BOX_LAVA_PIT_TILES,
                 depth: int = _BOX_LAVA_DEPTH) -> None:
        self.pit_tiles = pit_tiles
        self.depth = depth

    @classmethod
    def random(cls, rng: random.Random) -> "BoxLavaSegment":
        # Fixed at the controller-validated vault-proof cell. The box stays on
        # the approach ledge (box_frac=None) — pushing it in is the lesson.
        return cls(pit_tiles=_BOX_LAVA_PIT_TILES, depth=_BOX_LAVA_DEPTH)

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        # Flat spacer (not a boost pad) keeps pit_left at x=256 — the geometry
        # the scripted BoxHopAgent is calibrated to (Flat2=64 + Flat3=96 +
        # BoxLavaGap.approach 3t=96 = 256).
        x += self._chunk(Flat(width_tiles=3), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=self.pit_tiles,
                                    depth=self.depth), world, x)
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
    """Tier 3 — unlock a door, then PUSH the box across the lava pit (stage-3
    box-step: shove it off the approach ledge, then near ledge -> box top -> far
    ledge), then reach the goal. A Flat(3) spacer sits before the pit (no boost
    pad). The pit uses the vault-proof cell (pit=24, depth=72) so the box-push is
    mandatory even with DOUBLE_JUMP granted (the strongest max double jump cannot
    clear it without the box)."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(KeyChunk(width_tiles=2, key_id=0, y_offset=40), world, x)
        x += self._chunk(DoorChunk(width_tiles=2, key_id=0), world, x)
        x += self._chunk(Flat(width_tiles=2), world, x)
        x += self._chunk(Flat(width_tiles=3), world, x)
        x += self._chunk(BoxLavaGap(pit_tiles=_BOX_LAVA_PIT_TILES,
                                    depth=_BOX_LAVA_DEPTH), world, x)
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
