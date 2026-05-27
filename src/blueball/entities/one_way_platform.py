"""OneWayPlatform — a static segment that allows passage from below and blocks
from above. Implemented as a single Segment with CT_ONE_WAY; the collision
dispatcher's pre_solve handler filters out 'rising' contacts.
"""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class OneWayPlatform(Entity):
    def __init__(self, world, position: tuple[float, float], width: float) -> None:
        super().__init__()
        cx, cy = position
        hw = width / 2
        # Static-body segment; we use space.static_body so no new pymunk.Body needed.
        self.shape = pymunk.Segment(
            world.space.static_body,
            (cx - hw, cy), (cx + hw, cy), 5,
        )
        self.shape.collision_type = _col.CT_ONE_WAY
        self.shape.friction = 1.0
        self.shapes.append(self.shape)
        self.position = position
        self.width = width

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_one_way_platform(self.position, self.width)
