"""SwingingHazardChunk — a flat span with a pendulum bob hanging from above."""

from __future__ import annotations

import pymunk

from ...entities.swinging_hazard import SwingingHazard
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("swinging_hazard")
class SwingingHazardChunk(Chunk):
    difficulty: int = 3

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(3, 6),
            "rope_length": rng.randint(60, 150),
            "bob_mass": round(rng.uniform(1.0, 3.0), 2),
            "initial_angle_deg": round(rng.uniform(-30, 30), 1),
        }

    def __init__(
        self,
        width_tiles: int = 4,
        rope_length: float = 100,
        bob_mass: float = 2.0,
        bob_radius: int = 14,
        initial_angle_deg: float = 15.0,
    ) -> None:
        self.width_tiles = width_tiles
        self.rope_length = rope_length
        self.bob_mass = bob_mass
        self.bob_radius = bob_radius
        self.initial_angle_deg = initial_angle_deg

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)

        anchor_x = x_offset + w / 2
        anchor_y = GROUND_Y - self.rope_length - 30  # anchor hangs above ground

        world.add_entity(
            SwingingHazard(
                world=world,
                anchor_pos=(anchor_x, anchor_y),
                rope_length=self.rope_length,
                bob_mass=self.bob_mass,
                bob_radius=self.bob_radius,
                initial_angle_deg=self.initial_angle_deg,
            )
        )
        return w
