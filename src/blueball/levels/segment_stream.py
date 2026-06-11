"""Completion-gym segment streaming — pygame-free sibling of TerrainStream.

Materializes solvable goal-segments from a SegmentSampler ahead of the ball and
culls those behind it. Records the cumulative end-x of every segment ever built
(cheap floats, never culled) in `segment_ends`, so the evaluator can count how
many segments the ball has passed even after their physics objects are removed.
Builds on the flat GROUND_Y baseline (no base_y threading needed).
"""

from __future__ import annotations

from .. import config
from .chunks.flat import Flat
from .segments import SegmentSampler


class SegmentStream:
    def __init__(
        self,
        world,
        seed: int,
        granted_abilities: frozenset,
        *,
        load_ahead: float = config.GYM_LOAD_AHEAD,
        load_behind: float = config.GYM_LOAD_BEHIND,
        initial_segments: int = config.GYM_INITIAL_SEGMENTS,
    ) -> None:
        self.world = world
        self.load_ahead = load_ahead
        self.load_behind = load_behind
        self.sampler = SegmentSampler(int(seed), frozenset(granted_abilities))
        self.build_x: float = 0.0
        self.built: list[dict] = []
        self.segment_ends: list[float] = []

        # Spawn footing: a guaranteed flat at x=0 (recorded for culling, but NOT
        # a counted segment — it has no goal).
        self._materialize(Flat(width_tiles=4))
        for _ in range(initial_segments):
            self._build_next_segment()

    def _materialize(self, builder) -> float:
        """Build `builder` (a chunk or segment exposing
        `build(world, x_offset)`) at the cursor, recording exactly what got added
        to the space so it can be culled later. Returns the width."""
        pre_shapes = set(self.world.space.shapes)
        pre_bodies = set(self.world.space.bodies)
        pre_entities = set(self.world.entities)
        pre_constraints = set(self.world.space.constraints)

        width = builder.build(self.world, x_offset=self.build_x)

        from .seams import weld_ground_seams
        weld_ground_seams(self.world.space)

        self.built.append({
            "x_end": self.build_x + width,
            "shapes": set(self.world.space.shapes) - pre_shapes,
            "bodies": set(self.world.space.bodies) - pre_bodies,
            "entities": set(self.world.entities) - pre_entities,
            "constraints": set(self.world.space.constraints) - pre_constraints,
        })
        self.build_x += width
        return width

    def _build_next_segment(self) -> None:
        template = self.sampler.emit_next()
        self._materialize(template)
        self.segment_ends.append(self.build_x)  # cursor now sits at the segment end

    def maintain(self, player_x: float) -> None:
        """Per-tick: build ahead to keep load_ahead px materialized, cull units
        fully behind player_x - load_behind."""
        while self.build_x < player_x + self.load_ahead:
            self._build_next_segment()
        cutoff = player_x - self.load_behind
        while self.built and self.built[0]["x_end"] < cutoff:
            info = self.built.pop(0)
            for shape in info["shapes"]:
                if shape in self.world.space.shapes:
                    self.world.space.remove(shape)
                self.world._shape_to_entity.pop(shape, None)
            for constraint in info["constraints"]:
                if constraint in self.world.space.constraints:
                    self.world.space.remove(constraint)
            for body in info["bodies"]:
                if body is self.world.space.static_body:
                    continue
                if body in self.world.space.bodies:
                    self.world.space.remove(body)
            for entity in info["entities"]:
                if entity in self.world.entities:
                    self.world.entities.remove(entity)
