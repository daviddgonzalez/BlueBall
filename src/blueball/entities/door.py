"""Door — a solid vertical Segment that opens when the player holds the matching key.

The door is a static Segment with CT_DOOR. When the player contacts it while
holding the matching key, the collision handler sets `_opening=True`. On the next
`update()` call the shape is removed from the physics space and `is_open` is set.
"""

from __future__ import annotations

import pymunk

from ..collision import CT_DOOR
from .base import Entity


class Door(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        height: int,
        key_id: int,
    ) -> None:
        super().__init__()
        self._world = world
        self.position = position
        self.height = height
        self.key_id = key_id
        self.is_open: bool = False
        self._opening: bool = False

        x, y = position
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (0, 0)
        shape = pymunk.Segment(self.body, (x, y), (x, y - height), 4)
        shape.friction = 1.0
        shape.collision_type = CT_DOOR
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def update(self, dt: float) -> None:
        if self._opening and not self.is_open:
            self._remove_from_space()
            self.is_open = True

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_door(self.position, self.height, self.is_open)
