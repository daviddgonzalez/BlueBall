"""key chunk — a flat ground segment with a floating Key sensor pickup."""

from __future__ import annotations

import pymunk

from ...entities.key import Key
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("key")
class KeyChunk(Chunk):
    sampler_include: bool = False
    difficulty: int = 1

    def __init__(
        self,
        width_tiles: int = 2,
        key_id: int = 0,
    ) -> None:
        self.width_tiles = width_tiles
        self.key_id = key_id

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, GROUND_Y),
            (x_offset + w, GROUND_Y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(Key(
            world,
            position=(x_offset + w / 2, GROUND_Y - 48),
            key_id=self.key_id,
        ))
        return w
