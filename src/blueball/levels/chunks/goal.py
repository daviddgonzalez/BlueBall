"""GoalChunk — places a Goal entity. Always the last chunk in a v1 level."""

from __future__ import annotations

import pymunk

from ...entities.goal import Goal
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("goal")
class GoalChunk(Chunk):
    def __init__(self, width_tiles: int = 2) -> None:
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        goal_cx = x_offset + w / 2
        world.add_entity(Goal(world, position=(goal_cx, GROUND_Y - 40), width=40, height=80))
        return w
