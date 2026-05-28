"""GoalChunk — places a Goal entity. Always the last chunk in a v1 level."""

from __future__ import annotations

import pymunk

from ...entities.goal import Goal
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("goal")
class GoalChunk(Chunk):
    sampler_include: bool = False

    def __init__(self, width_tiles: int = 2, y_offset: float = 0) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        y = base_y - self.y_offset
        seg = pymunk.Segment(world.space.static_body, (x_offset, y), (x_offset + w, y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        goal_cx = x_offset + w / 2
        world.add_entity(Goal(world, position=(goal_cx, y - 40), width=40, height=80))
        return w
