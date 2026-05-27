"""one_way_platform chunk — places one OneWayPlatform floating above ground."""

from __future__ import annotations

from ...entities.one_way_platform import OneWayPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("one_way_platform")
class OneWayPlatformChunk(Chunk):
    difficulty: int = 1

    def __init__(self, width_tiles: int = 4, y_offset: int = 96) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "y_offset": rng.choice([64, 96, 128])}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        y = GROUND_Y - self.y_offset
        world.add_entity(OneWayPlatform(world, position=(x_offset + w / 2, y), width=w))
        return w
