"""PatrolPlatform — a flat span with one patroller walking back and forth."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.patroller import Patroller
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("patrol_platform")
class PatrolPlatform(Chunk):
    def __init__(self, length_tiles: int = 6, patroller_speed: float = config.PATROLLER_SPEED) -> None:
        self.length_tiles = length_tiles
        self.patroller_speed = patroller_speed

    def build(self, world, x_offset: float) -> float:
        w = self.length_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        left = x_offset + 16
        right = x_offset + w - 16
        world.add_entity(
            Patroller(
                world,
                position=((left + right) / 2, GROUND_Y - 12),
                left_bound=left,
                right_bound=right,
                speed=self.patroller_speed,
            )
        )
        return w
