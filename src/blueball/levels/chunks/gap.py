"""Gap — empty space the player must jump across. No geometry."""

from __future__ import annotations

from .base import Chunk, TILE, register_chunk


@register_chunk("gap")
class Gap(Chunk):
    def __init__(self, width_tiles: int = 3) -> None:
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float) -> float:
        return self.width_tiles * TILE
