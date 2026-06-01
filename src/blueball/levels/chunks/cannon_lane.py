"""cannon_lane chunk — a flat ground lane with a Cannon firing across it.

A streaming-friendly way to expose projectiles to the AI: the player must time a
jump (or its run) to cross the lane without being hit. The cannon sits low at one
end and fires toward the other end with a randomized cadence.
"""

from __future__ import annotations

import pymunk

from ...entities.cannon import Cannon
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("cannon_lane")
class CannonLane(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 6,
        direction: str = "right",
        speed: float = 240.0,
        barrel_height: int = 24,
    ) -> None:
        self.width_tiles = width_tiles
        self.direction = direction
        self.speed = speed
        # How high the barrel (and so the projectile lane) sits above the floor;
        # kept low so the player can clear shots with a jump.
        self.barrel_height = barrel_height

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(5, 7),
            "direction": rng.choice(["left", "right"]),
            "speed": rng.choice([200.0, 240.0, 280.0]),
        }

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body, (x_offset, base_y), (x_offset + w, base_y), 5
        )
        seg.friction = 1.0
        world.space.add(seg)
        # Mount the cannon at the end opposite its firing direction so shots
        # travel the length of the lane the player has to cross.
        if self.direction == "right":
            cx = x_offset + 8
        else:
            cx = x_offset + w - 8
        world.add_entity(Cannon(
            world,
            position=(cx, base_y - self.barrel_height),
            direction=self.direction,
            interval_min_s=0.3,
            interval_max_s=2.0,
            speed=self.speed,
            pulse_period_s=0.5,
            max_travel=w,
            projectile_radius=10,
        ))
        return w
