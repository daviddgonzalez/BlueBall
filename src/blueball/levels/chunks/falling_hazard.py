"""FallingHazardChunk — a flat span with a hanging hazard that drops when the
player crosses a trigger x-coordinate.
"""

from __future__ import annotations

import pymunk

from ...entities.falling_hazard import FallingHazard
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("falling_hazard")
class FallingHazardChunk(Chunk):
    difficulty: int = 3

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "hazard_height": rng.randint(160, 240)}

    def __init__(self, width_tiles: int = 4, hazard_height: int = 200) -> None:
        self.width_tiles = width_tiles
        self.hazard_height = hazard_height

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, base_y), (x_offset + w, base_y), 5)
        seg.friction = 1.0
        world.space.add(seg)

        # The trigger is at the start of the chunk; hazard drops in the middle.
        trigger_x = x_offset + 16
        hazard_x = x_offset + w / 2

        def find_player_body():
            from ...entities.player import Player

            for entity in world.entities:
                if isinstance(entity, Player):
                    return entity.body
            return None

        world.add_entity(
            FallingHazard(
                world,
                position=(hazard_x, base_y - self.hazard_height),
                trigger_x=trigger_x,
                player_provider=find_player_body,
            )
        )
        return w
