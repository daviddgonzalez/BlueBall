"""box_spring_relay chunk — push a box onto a ground spring; it arcs onto a
second spring on a raised platform, which relaunches it higher, to reach an
exit above the second spring. Guide walls constrain the box's arc.

The horizontal arc depends on the box's launch velocity; exact geometry is
tuned by playtest (see plan Task 6). If it proves unsolvable after tuning,
fall back to a trampoline variant.
"""

from __future__ import annotations

import pymunk

from ...entities.spring import Spring
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("box_spring_relay")
class BoxSpringRelay(Chunk):
    sampler_include: bool = False
    difficulty: int = 5

    def __init__(
        self,
        width_tiles: int = 8,
        impulse1: float = 720.0,
        impulse2: float = 760.0,
        platform_height: int = 200,   # spring-2 platform px above base_y
        relay_dx_tiles: int = 4,      # horizontal offset of spring 2
        platform_tiles: int = 2,
        box_size: int = 36,
        box_mass: float = 0.5,
        exit_height: int = 360,       # exit ledge px above base_y (tuned)
        exit_tiles: int = 2,
        wall_height: int = 240,       # guide-wall height (tuned)
    ) -> None:
        self.width_tiles = width_tiles
        self.impulse1 = impulse1
        self.impulse2 = impulse2
        self.platform_height = platform_height
        self.relay_dx_tiles = relay_dx_tiles
        self.platform_tiles = platform_tiles
        self.box_size = box_size
        self.box_mass = box_mass
        self.exit_height = exit_height
        self.exit_tiles = exit_tiles
        self.wall_height = wall_height

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE

        def seg(a, b):
            s = pymunk.Segment(world.space.static_body, a, b, 5)
            s.friction = 1.0
            world.space.add(s)

        # Ground.
        seg((x_offset, base_y), (x_offset + w, base_y))

        # Spring 1 near the left, on the ground.
        s1_cx = x_offset + 1.5 * TILE
        world.add_entity(Spring(
            world, position=(s1_cx, base_y - 8), width=2 * TILE, impulse=self.impulse1
        ))

        # Raised platform carrying spring 2.
        plat_y = base_y - self.platform_height
        plat_left = s1_cx + self.relay_dx_tiles * TILE - (self.platform_tiles * TILE) / 2
        plat_right = plat_left + self.platform_tiles * TILE
        seg((plat_left, plat_y), (plat_right, plat_y))
        s2_cx = (plat_left + plat_right) / 2
        world.add_entity(Spring(
            world, position=(s2_cx, plat_y - 8), width=self.platform_tiles * TILE,
            impulse=self.impulse2,
        ))

        # Two guide walls to keep the box's arc on course (left of spring 1 and
        # right of the platform), so a too-hard push doesn't fling it offscreen.
        seg((x_offset, base_y), (x_offset, base_y - self.wall_height))
        seg((x_offset + w, base_y), (x_offset + w, base_y - self.wall_height))

        # Box starts just right of spring 1, ready to shove onto it.
        world.add_entity(PushableBox(
            world,
            position=(s1_cx + TILE, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))

        # Exit ledge above spring 2.
        ledge_y = base_y - self.exit_height
        seg((s2_cx - self.exit_tiles * TILE / 2, ledge_y),
            (s2_cx + self.exit_tiles * TILE / 2, ledge_y))
        return w
