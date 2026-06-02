"""Key — a static sensor entity that sets a bit in player.keys_held on contact.

Mirrors the structure of Checkpoint: a sensor circle that removes itself from
the physics space after being collected.
"""

from __future__ import annotations

import pymunk

from ..collision import CT_KEY
from .base import Entity


class Key(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        key_id: int,
        radius: int = 18,
    ) -> None:
        super().__init__()
        self._world = world
        self.key_id = key_id
        self.position = position
        self.radius = radius
        self._collected = False

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, radius)
        shape.sensor = True
        shape.collision_type = CT_KEY
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def update(self, dt: float) -> None:
        if self._collected:
            self._remove_from_space()

    def draw(self, renderer, alpha: float) -> None:
        if self._collected:
            return
        renderer.draw_key(self.position, self.radius, self.key_id)
