"""pushable_box chunk — ground segment + a single PushableBox sitting on it."""

from __future__ import annotations

import pymunk

from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("pushable_box")
class PushableBoxChunk(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 2,
        size_px: int = 32,
        mass: float = 0.5,
    ) -> None:
        self.width_tiles = width_tiles
        self.size_px = size_px
        self.mass = mass

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(2, 3),
            "size_px": rng.choice([28, 32, 40]),
            "mass": round(rng.uniform(0.4, 0.8), 2),
        }

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
        cx = x_offset + w / 2
        # Spawn slightly above ground so the box doesn't intersect the segment.
        world.add_entity(
            PushableBox(
                world,
                position=(cx, GROUND_Y - self.size_px / 2 - 1),
                size=self.size_px,
                mass=self.mass,
            )
        )
        return w
