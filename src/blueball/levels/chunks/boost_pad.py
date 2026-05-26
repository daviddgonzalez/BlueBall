"""boost_pad chunk — a flat ground segment with a BoostPad sensor on top."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.boost_pad import BoostPad
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("boost_pad")
class BoostPadChunk(Chunk):
    def __init__(
        self,
        width_tiles: int = 4,
        multiplier: float = config.BOOST_PAD_DEFAULT_MULTIPLIER,
    ) -> None:
        self.width_tiles = width_tiles
        self.multiplier = multiplier

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        # Ground segment under the pad
        seg = pymunk.Segment(
            world.space.static_body,
            (x_offset, GROUND_Y),
            (x_offset + w, GROUND_Y),
            5,
        )
        seg.friction = 1.0
        world.space.add(seg)
        # Pad sits flush at the top of the ground so the ball rolls over it.
        # Center the sensor on the segment; the half-thickness lift puts the
        # pad's top edge at GROUND_Y.
        world.add_entity(BoostPad(
            world,
            position=(x_offset + w / 2, GROUND_Y - config.BOOST_PAD_THICKNESS / 2),
            width=w,
            multiplier=self.multiplier,
        ))
        return w
