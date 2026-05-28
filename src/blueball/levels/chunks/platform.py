"""Platform — a single floating horizontal segment. No ground beneath; the
chunk consumes its horizontal slot but the level's ground is invisible here.
"""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("platform")
class Platform(Chunk):
    difficulty: int = 0

    def __init__(self, width_tiles: int = 4, y_offset: int = 96) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "y_offset": rng.choice([64, 96, 128])}

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        y = base_y - self.y_offset
        seg = pymunk.Segment(world.space.static_body, (x_offset, y), (x_offset + w, y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        return w
