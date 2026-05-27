"""VerticalColumn — N stacked floating platforms inside the chunk's horizontal
slot, alternating left/right hugs so the player must zig-zag jump.
"""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("vertical_column")
class VerticalColumn(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 6,
        steps: int = 5,
        step_height: int = 80,
        bottom_offset: int = 96,
        platform_tiles: int = 2,
    ) -> None:
        self.width_tiles = width_tiles
        self.steps = steps
        self.step_height = step_height
        self.bottom_offset = bottom_offset
        self.platform_tiles = platform_tiles

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": 6,
            "steps": rng.randint(3, 6),
            "step_height": rng.choice([64, 80, 96]),
            "bottom_offset": 96,
            "platform_tiles": 2,
        }

    def build(self, world, x_offset: float) -> float:
        slot_w = self.width_tiles * TILE
        plat_w = self.platform_tiles * TILE
        for i in range(self.steps):
            y = GROUND_Y - (self.bottom_offset + i * self.step_height)
            if i % 2 == 0:
                a = (x_offset, y)
                b = (x_offset + plat_w, y)
            else:
                a = (x_offset + slot_w - plat_w, y)
                b = (x_offset + slot_w, y)
            seg = pymunk.Segment(world.space.static_body, a, b, 5)
            seg.friction = 1.0
            world.space.add(seg)
        return slot_w
