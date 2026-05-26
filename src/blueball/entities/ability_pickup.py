"""AbilityPickup — a sensor entity that grants an Ability to the Player on contact.

Mirrors the structure of Collectible: a static-body sensor circle that tears
down its physics presence once collected.
"""

from __future__ import annotations

import pymunk

from ..abilities import Ability
from ..collision import CT_ABILITY_PICKUP
from .base import Entity


class AbilityPickup(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        ability: Ability,
        radius: int = 18,
    ) -> None:
        super().__init__()
        self._world = world
        self.ability = ability
        self.position = position
        self.radius = radius

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, radius)
        shape.sensor = True
        shape.collision_type = CT_ABILITY_PICKUP
        self.bodies.append(self.body)
        self.shapes.append(shape)
        self._collected = False

    def consume(self) -> None:
        """Remove this pickup from the physics space and flag it dead."""
        if self._collected:
            return
        self._collected = True
        self.alive = False
        for shape in self.shapes:
            if shape in self._world.space.shapes:
                self._world.space.remove(shape)
        for body in self.bodies:
            if body in self._world.space.bodies:
                self._world.space.remove(body)

    def draw(self, renderer, alpha: float) -> None:
        if self.alive:
            renderer.draw_ability_pickup(self.position, self.radius, str(self.ability))
