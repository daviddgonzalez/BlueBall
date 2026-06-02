"""spike_wall chunk — N oriented spikes along a wall (ceiling or side)."""

from __future__ import annotations

import pymunk

from ...entities.spike import Spike
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("spike_wall")
class SpikeWall(Chunk):
    difficulty: int = 2

    def __init__(self, width_tiles: int = 3, spikes: int = 3, orientation: str = "down", ceiling_y_offset: int = 160) -> None:
        self.width_tiles = width_tiles
        self.spikes = spikes
        self.orientation = orientation
        self.ceiling_y_offset = ceiling_y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(2, 4),
            "spikes": rng.randint(2, 4),
            "orientation": rng.choice(["down", "left", "right"]),
            "ceiling_y_offset": rng.choice([128, 160, 200]),
        }

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        # For "up" behave like spike_pit (spikes at ground level).
        # For "down"/"left"/"right" place spikes at ceiling height.
        if self.orientation == "up":
            y = base_y
        else:
            y = base_y - self.ceiling_y_offset
        # Flat ground segment beneath the spike row
        seg = pymunk.Segment(world.space.static_body, (x_offset, base_y), (x_offset + w, base_y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        for i in range(self.spikes):
            cx = x_offset + (i + 0.5) * w / self.spikes
            world.add_entity(Spike(world, position=(cx, y), width=24, height=24, orientation=self.orientation))
        return w
