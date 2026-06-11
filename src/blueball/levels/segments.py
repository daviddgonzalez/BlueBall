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
from .chunks.lava_gap import LavaGapChunk
from .chunks.double_ledge import DoubleLedge
from .chunks.double_step import DoubleStep
from .chunks.double_gap import DoubleGap


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
# depth=72 (> box=64) keeps the box-top below the ledge for a clean step.
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
        # These params exist ONLY so tests/probes can inject geometry; they are
        # NOT a sampler knob. The sampler always calls .random(), which pins the
        # fixed vault-proof cell regardless of these defaults.
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
        # NOTE: the scripted-solver pit_left=256 calibration (shared by
        # BoxStep/BoxLeap/BoxLavaSegment) does NOT hold here — the leading Key+
        # Door chunks push the pit wall out to ~x=448. That 256 invariant is
        # therefore NOT global. This is a composition-only segment with NO
        # scripted-solver test; don't assume the 256 calibration applies.
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


# Probe-chosen gap width (probes/tune_boost_gap.py): the largest lava-gap width
# inside the boost-or-die corridor [23, 28] that still has a tile of margin on
# each side. At this width a BOOSTED apex-fired double jump clears the pit but a
# bare (no-boost) double jump falls short into the lava and dies, so the boost
# pad is mandatory.
_BOOST_GAP_TILES = 27


class BoostGapSegment(SegmentTemplate):
    """Tier 2 — a boost-or-die lava gap. A boost pad sits on a LONG runway
    before a wide, full-height lava pit (any fall is lethal). A BOOSTED double
    jump is the ONLY way across: a bare double jump falls short into the lava
    and dies, so the boost is mandatory. The runway is generous (8 tiles before
    the 3-tile pad) so the player never spawns on top of the pad (per playtest
    feedback). Requires DOUBLE_JUMP (granted by default).

    Once the player crosses the pad the boost stays locked in for ~2s
    (config.BOOST_DURATION_S = 2.0), so the apex jump fired after the pad is
    still boosted and clears the gap.

    Probe-tuned (probes/tune_boost_gap.py): pit_tiles=27 sits inside the
    boost-or-die corridor [23, 28] with a tile of margin on each side."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "BoostGapSegment":
        return cls()  # fixed, probe-tuned geometry

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=8), world, x)
        x += self._chunk(BoostPadChunk(width_tiles=3, multiplier=2.0), world, x)
        x += self._chunk(LavaGapChunk(pit_tiles=_BOOST_GAP_TILES), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


# Generous landing flat after the double-jump obstacle. Even the strongest
# (max-distance) double jump overshoots a tightly-placed goal into the void; this
# catches the landing so the segment is solvable, while a single jump never
# reaches it (it can't clear the obstacle at all).
_DJ_LANDING_TILES = 14


class DoubleHopSegment(SegmentTemplate):
    """Tier 2 — gentle double-jump rung: leap a small gap and double-jump up onto
    a raised ledge. A single jump can't reach the cliff top behind the gap
    (measured: a 5-tile gap caps the single-jump cliff mount at ~136px, the ledge
    is 172px), so the second jump is mandatory; then land on the run-out flat and
    roll to the goal. The vertical 'hop' lesson — the most forgiving of the pair.

    Unlike the box segments, the double jump here is the *traversal* itself, not a
    box-step. Solvable by the max double-jump maneuver, unsolvable without it."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "DoubleHopSegment":
        return cls()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=6), world, x)
        x += self._chunk(DoubleLedge(gap_tiles=5, height=172), world, x)
        x += self._chunk(Flat(width_tiles=_DJ_LANDING_TILES), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class DoubleWallSegment(SegmentTemplate):
    """Tier 2 — pure-vertical double-jump rung: mount a flush wall (200px) too
    tall for a single jump (measured single-jump flush mount ceiling ~172px) but
    inside a double's ~260px. No gap — just 'jump, jump again to get up' — then
    land on the run-out flat and roll to the goal. Solvable by the max double-jump
    maneuver, unsolvable without it."""

    tier = 2
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "DoubleWallSegment":
        return cls()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=6), world, x)
        x += self._chunk(DoubleStep(height=200), world, x)
        x += self._chunk(Flat(width_tiles=_DJ_LANDING_TILES), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


class DoubleVaultSegment(SegmentTemplate):
    """Tier 3 — demanding double-jump rung: vault a wide fall-death gap (16 tiles
    / 512px) that's past a single jump's ~420px reach but inside a double's
    ~720px, then land on the run-out flat and roll to the goal. The horizontal
    'vault' lesson — tighter timing than the hop. Solvable by the max double-jump
    maneuver, unsolvable without it."""

    tier = 3
    min_abilities = frozenset({Ability.DOUBLE_JUMP})

    @classmethod
    def random(cls, rng: random.Random) -> "DoubleVaultSegment":
        return cls()

    def build(self, world, x_offset: float) -> float:
        x = x_offset
        x += self._chunk(Flat(width_tiles=6), world, x)
        x += self._chunk(DoubleGap(width_tiles=16), world, x)
        x += self._chunk(Flat(width_tiles=_DJ_LANDING_TILES), world, x)
        x += self._chunk(GoalChunk(width_tiles=2), world, x)
        return x - x_offset


# Registry that the sampler draws from. (List order is cosmetic — the sampler
# sorts by class name — but kept tier-ordered for readability.)
SEGMENT_TEMPLATES: list[type[SegmentTemplate]] = [
    GoalSegment,            # tier 0
    KeyDoorGoalSegment,     # tier 1
    BoxStepSegment,         # tier 2 (curriculum stage 1 — single jump onto box)
    BoxLavaSegment,         # tier 2 (curriculum stage 3 — expert box-push)
    BoostGapSegment,        # tier 2 (boost-or-die lava gap)
    DoubleHopSegment,       # tier 2 (double-jump traversal — gap-to-ledge hop)
    DoubleWallSegment,      # tier 2 (double-jump traversal — flush wall mount)
    BoxLeapSegment,         # tier 3 (curriculum stage 2 — double jump onto box)
    KeyDoorBoxLavaSegment,  # tier 3
    DoubleVaultSegment,     # tier 3 (double-jump traversal — demanding vault)
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
