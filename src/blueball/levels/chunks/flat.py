"""Flat ground chunk — a single horizontal segment of width N tiles."""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk

GROUND_Y = 600  # baseline ground height; consistent across chunks


@register_chunk("flat")
class Flat(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}

    def __init__(self, width_tiles: int = 6) -> None:
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        return w
