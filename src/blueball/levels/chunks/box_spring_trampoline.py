"""box_spring_trampoline chunk — push a box onto a spring, then bounce off the
rising box to reach a high exit ledge that's out of reach otherwise.

No catch ledge: the box never lands as a permanent step. The bounce itself is
the only way up. Exact heights are tuned by playtest.
"""

from __future__ import annotations

import pymunk

from ...entities.spring import Spring
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("box_spring_trampoline")
class BoxSpringTrampoline(Chunk):
    sampler_include: bool = False
    difficulty: int = 4

    def __init__(
        self,
        width_tiles: int = 6,
        impulse: float = 720.0,
        box_size: int = 58,
        box_mass: float = 0.5,
        exit_height: int = 220,   # exit ledge px above base_y (playtest-tuned)
        exit_tiles: int = 2,
    ) -> None:
        self.width_tiles = width_tiles
        self.impulse = impulse
        self.box_size = box_size
        self.box_mass = box_mass
        self.exit_height = exit_height
        self.exit_tiles = exit_tiles

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        ground = pymunk.Segment(
            world.space.static_body, (x_offset, base_y), (x_offset + w, base_y), 5
        )
        ground.friction = 1.0
        world.space.add(ground)

        spring_cx = x_offset + w * 0.62
        world.add_entity(Spring(
            world, position=(spring_cx, base_y - 8), width=2 * TILE, impulse=self.impulse
        ))

        # Box starts left of the spring, ready to push onto it.
        world.add_entity(PushableBox(
            world,
            position=(x_offset + w * 0.25, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))

        # High exit ledge above the spring, out of normal-jump reach.
        ledge_y = base_y - self.exit_height
        ledge = pymunk.Segment(
            world.space.static_body,
            (x_offset + w - self.exit_tiles * TILE, ledge_y),
            (x_offset + w, ledge_y),
            5,
        )
        ledge.friction = 1.0
        world.space.add(ledge)
        return w
