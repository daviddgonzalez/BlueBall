"""spring chunk — flat ground with a sensor Spring on top."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.spring import Spring
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("spring")
class SpringChunk(Chunk):
    difficulty: int = 1

    def __init__(
        self,
        width_tiles: int = 2,
        impulse: float = config.SPRING_DEFAULT_IMPULSE,
    ) -> None:
        self.width_tiles = width_tiles
        self.impulse = impulse

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(2, 3),
            "impulse": rng.choice([500.0, 600.0, 720.0]),
        }

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, base_y),
            (x_offset + w, base_y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(
            Spring(
                world,
                position=(x_offset + w / 2, base_y - 8),
                width=w,
                impulse=self.impulse,
            )
        )
        return w
