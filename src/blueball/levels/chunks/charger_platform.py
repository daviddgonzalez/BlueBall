"""charger_platform chunk — flat ground with a Charger patrolling it."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.charger import Charger
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("charger_platform")
class ChargerPlatformChunk(Chunk):
    difficulty: int = 3

    def __init__(
        self,
        length_tiles: int = 8,
        facing: str = "right",
        sight_range: float = config.CHARGER_DEFAULT_SIGHT_RANGE,
        sight_arc_deg: float = config.CHARGER_DEFAULT_SIGHT_ARC_DEG,
        charge_speed: float = config.CHARGER_DEFAULT_CHARGE_SPEED,
        patrol_speed: float = config.CHARGER_DEFAULT_PATROL_SPEED,
    ) -> None:
        self.length_tiles = length_tiles
        self.facing = facing
        self.sight_range = sight_range
        self.sight_arc_deg = sight_arc_deg
        self.charge_speed = charge_speed
        self.patrol_speed = patrol_speed

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "length_tiles": rng.randint(6, 10),
            "facing": rng.choice(["left", "right"]),
            "sight_range": rng.choice([160, 200, 240]),
            "charge_speed": rng.choice([140.0, 180.0, 220.0]),
        }

    def build(self, world, x_offset: float) -> float:
        w = self.length_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        left = x_offset + 16
        right = x_offset + w - 16
        world.add_entity(Charger(
            world,
            position=((left + right) / 2, GROUND_Y - 12),
            left_bound=left,
            right_bound=right,
            facing=self.facing,
            sight_range=self.sight_range,
            sight_arc_deg=self.sight_arc_deg,
            charge_speed=self.charge_speed,
            patrol_speed=self.patrol_speed,
        ))
        return w
