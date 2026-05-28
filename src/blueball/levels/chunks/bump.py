"""Bump — a small hill the ball rolls over.

Hand-authored levels get the classic triangular bump. The Infinite Run sampler
sets ``rounded=True`` (via random_params) to make it a smooth rounded hump.
"""

from __future__ import annotations

import pymunk

from ._curve import smoothstep_ramp
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("bump")
class Bump(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"height": rng.randint(24, 48), "width_tiles": rng.randint(2, 3), "rounded": True}

    def __init__(self, height: int = 32, width_tiles: int = 2, rounded: bool = False) -> None:
        self.height = height
        self.width_tiles = width_tiles
        self.rounded = rounded

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        peak_y = base_y - self.height
        mid = x_offset + w / 2
        if self.rounded:
            # Two smoothstep ramps: ground -> peak -> ground. Both are flat at
            # the shared peak, so the top is a rounded crest, not a sharp apex.
            smoothstep_ramp(world, x_offset, base_y, mid, peak_y)
            smoothstep_ramp(world, mid, peak_y, x_offset + w, base_y)
            return w
        # Classic triangular bump: two segments meeting at a sharp apex.
        verts = [
            (x_offset, base_y),
            (mid, peak_y),
            (x_offset + w, base_y),
        ]
        for a, b in [(verts[0], verts[1]), (verts[1], verts[2])]:
            seg = pymunk.Segment(world.space.static_body, a, b, 5)
            seg.friction = 1.0
            world.space.add(seg)
        return w
