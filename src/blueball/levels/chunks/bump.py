"""Bump — a small hill the ball can roll over."""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("bump")
class Bump(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"height": rng.randint(24, 48), "width_tiles": rng.randint(2, 3)}

    def __init__(self, height: int = 32, width_tiles: int = 2) -> None:
        self.height = height
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        verts = [
            (x_offset, GROUND_Y),
            (x_offset + w / 2, GROUND_Y - self.height),
            (x_offset + w, GROUND_Y),
        ]
        # Two segments forming a triangular bump on top of the ground
        for a, b in [(verts[0], verts[1]), (verts[1], verts[2])]:
            seg = pymunk.Segment(world.space.static_body, a, b, 5)
            seg.friction = 1.0
            world.space.add(seg)
        return w
