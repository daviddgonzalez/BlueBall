"""crumbling_platform chunk — a floating CrumblingPlatform that starts a timer
on first dynamic-body contact and removes itself when the timer expires."""

from __future__ import annotations

from ...entities.crumbling_platform import CrumblingPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("crumbling_platform")
class CrumblingPlatformChunk(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 4,
        y_offset: int = 96,
        crumble_delay_s: float = 1.0,
    ) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset
        self.crumble_delay_s = crumble_delay_s

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(3, 5),
            "y_offset": rng.choice([64, 96, 128]),
        }

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        y = GROUND_Y - self.y_offset
        world.add_entity(
            CrumblingPlatform(
                world,
                position=(x_offset + w / 2, y),
                width=w,
                crumble_delay_s=self.crumble_delay_s,
            )
        )
        return w
