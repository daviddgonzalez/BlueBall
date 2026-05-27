"""moving_platform chunk — places a MovingPlatform entity."""

from __future__ import annotations

from ... import config
from ...entities.moving_platform import MovingPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("moving_platform")
class MovingPlatformChunk(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 4,
        length_tiles: int = 2,
        axis: str = "x",
        range_px: float = 160,
        speed: float = config.MOVING_PLATFORM_DEFAULT_SPEED,
        y_offset: int = 96,
    ) -> None:
        self.width_tiles = width_tiles
        self.length_tiles = length_tiles
        self.axis = axis
        self.range_px = range_px
        self.speed = speed
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(4, 6),
            "length_tiles": 2,
            "axis": rng.choice(["x", "y"]),
            "range_px": rng.choice([120, 160, 200]),
            "speed": rng.choice([60.0, 80.0, 100.0]),
            "y_offset": rng.choice([64, 96, 128]),
        }

    def build(self, world, x_offset: float) -> float:
        slot_w = self.width_tiles * TILE
        plat_len = self.length_tiles * TILE
        cx = x_offset + slot_w / 2
        cy = GROUND_Y - self.y_offset
        world.add_entity(
            MovingPlatform(
                world,
                position=(cx, cy),
                length=plat_len,
                axis=self.axis,
                range_px=self.range_px,
                speed=self.speed,
            )
        )
        return slot_w
