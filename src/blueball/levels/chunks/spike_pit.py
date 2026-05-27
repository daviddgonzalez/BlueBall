"""SpikePit — a ground span with N spikes spaced across it."""

from __future__ import annotations

import pymunk

from ...entities.spike import Spike
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("spike_pit")
class SpikePit(Chunk):
    difficulty: int = 2

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 4), "spikes": rng.randint(2, 4)}

    def __init__(self, width_tiles: int = 3, spikes: int = 3) -> None:
        self.width_tiles = width_tiles
        self.spikes = spikes

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        # Add the ground segment beneath the spikes
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        # Space spikes evenly across the span
        for i in range(self.spikes):
            cx = x_offset + (i + 0.5) * w / self.spikes
            world.add_entity(Spike(world, position=(cx, GROUND_Y), width=24, height=24))
        return w
