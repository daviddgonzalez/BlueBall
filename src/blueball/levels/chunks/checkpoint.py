"""checkpoint chunk — a flat ground segment with a Checkpoint sensor."""

from __future__ import annotations

import pymunk

from ...entities.checkpoint import Checkpoint
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("checkpoint")
class CheckpointChunk(Chunk):
    sampler_include: bool = False
    difficulty: int = 0

    def __init__(
        self,
        width_tiles: int = 2,
        id: int = 0,
        y_offset: int = 32,
    ) -> None:
        self.width_tiles = width_tiles
        self.id = id
        self.y_offset = y_offset

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
        world.add_entity(Checkpoint(
            world,
            position=(x_offset + w / 2, GROUND_Y - self.y_offset),
            id=self.id,
        ))
        return w
