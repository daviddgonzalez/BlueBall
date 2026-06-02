"""Gap — empty space the player must jump across. No geometry."""

from __future__ import annotations

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("gap")
class Gap(Chunk):
    difficulty: int = 1

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}

    def __init__(self, width_tiles: int = 3) -> None:
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        return self.width_tiles * TILE
