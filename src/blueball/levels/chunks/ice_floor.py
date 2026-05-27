"""IceFloor — flat ground with very low friction. Momentum-management chunk."""

from __future__ import annotations

import pymunk

from ... import config
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("ice_floor")
class IceFloor(Chunk):
    difficulty: int = 1

    def __init__(self, width_tiles: int = 4) -> None:
        self.width_tiles = width_tiles

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = config.ICE_FLOOR_FRICTION
        world.space.add(seg)
        return w
