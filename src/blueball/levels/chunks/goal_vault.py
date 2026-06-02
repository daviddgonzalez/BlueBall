"""goal_vault chunk — the goal flag sealed inside a walled box at elevation; the
only way in is through locked doors in series (need every key).

The box has a floor, a ceiling, and a far (right) wall; the left face is the
entry mouth at floor level, blocked by two locked doors in series with
full-height walls above each so they can't be jumped. The goal sits beyond the
inner door.
"""

from __future__ import annotations

import pymunk

from ...entities.goal import Goal
from ...entities.door import Door
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("goal_vault")
class GoalVault(Chunk):
    sampler_include: bool = False

    def __init__(
        self,
        width_tiles: int = 8,
        y_offset: float = 520,
        vault_height: int = 192,
        door_height: int = 128,
        key_id_outer: int = 0,
        key_id_inner: int = 1,
    ) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset
        self.vault_height = vault_height
        self.door_height = door_height
        self.key_id_outer = key_id_outer
        self.key_id_inner = key_id_inner

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        floor_y = base_y - self.y_offset
        ceiling_y = floor_y - self.vault_height

        def seg(a, b):
            s = pymunk.Segment(world.space.static_body, a, b, 5)
            s.friction = 1.0
            world.space.add(s)

        # Box: floor, ceiling, far (right) wall. Left face is the entry mouth.
        seg((x_offset, floor_y), (x_offset + w, floor_y))
        seg((x_offset, ceiling_y), (x_offset + w, ceiling_y))
        seg((x_offset + w, floor_y), (x_offset + w, ceiling_y))

        # Two locked doors in series, each sealed to the ceiling above so they
        # can't be jumped. Outer first (nearer the entry), then inner.
        for dx_tiles, key_id in ((2, self.key_id_outer), (4, self.key_id_inner)):
            dx = x_offset + dx_tiles * TILE
            world.add_entity(Door(
                world, position=(dx, floor_y), height=self.door_height, key_id=key_id
            ))
            seg((dx, floor_y - self.door_height), (dx, ceiling_y))  # wall above door

        # Goal beyond the inner door, on the vault floor.
        goal_cx = x_offset + w - TILE
        world.add_entity(Goal(world, position=(goal_cx, floor_y - 40), width=40, height=80))
        return w
