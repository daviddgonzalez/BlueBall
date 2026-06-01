"""ability_pickup chunk — a flat ground segment with a floating AbilityPickup."""

from __future__ import annotations

import pymunk

from ... import config
from ...abilities import Ability
from ...entities.ability_pickup import AbilityPickup
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("ability_pickup")
class AbilityPickupChunk(Chunk):
    difficulty: int = 1

    def __init__(
        self,
        width_tiles: int = 2,
        ability: str = "double_jump",
        height: int = config.ABILITY_PICKUP_DEFAULT_HEIGHT,
    ) -> None:
        # Validate eagerly so a broken level JSON fails at load, not at collision time
        self.ability = Ability(ability)
        self.width_tiles = width_tiles
        self.height = height

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"ability": rng.choice([a.value for a in Ability])}

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, base_y),
            (x_offset + w, base_y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(AbilityPickup(
            world,
            position=(x_offset + w / 2, base_y - self.height),
            ability=self.ability,
        ))
        return w
