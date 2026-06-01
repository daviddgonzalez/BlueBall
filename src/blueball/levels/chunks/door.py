"""door chunk — a flat ground segment with a Door entity blocking passage."""

from __future__ import annotations

import pymunk

from ...entities.door import Door
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

# Top of the playfield. A locked door fills only its (short) opening near the
# ground; the wall above seals everything from the door's top up to here so the
# player cannot jump over the gate.
CEILING_Y = 0


@register_chunk("door")
class DoorChunk(Chunk):
    sampler_include: bool = False
    difficulty: int = 0

    def __init__(
        self,
        width_tiles: int = 2,
        key_id: int = 0,
        door_height: int = 128,
        height_tiles: int | None = None,
    ) -> None:
        self.width_tiles = width_tiles
        self.key_id = key_id
        # height_tiles is a convenience alias: 1 tile = 32 px
        if height_tiles is not None:
            self.door_height = height_tiles * 32
        else:
            self.door_height = door_height

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, GROUND_Y),
            (x_offset + w, GROUND_Y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        cx = x_offset + w / 2
        world.add_entity(Door(
            world,
            position=(cx, GROUND_Y),
            height=self.door_height,
            key_id=self.key_id,
        ))
        # Permanent wall sealing the gap above the door opening. This is static
        # geometry (not part of the Door entity, which removes only its own
        # shapes when it opens), so the gate stays closed above the doorway and
        # the player must use the key rather than jumping over.
        wall = pymunk.Segment(
            world.space.static_body,
            (cx, GROUND_Y - self.door_height),
            (cx, CEILING_Y),
            4,
        )
        wall.friction = 1.0
        world.space.add(wall)
        return w
