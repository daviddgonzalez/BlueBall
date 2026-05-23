"""FallingHazard — a kinematic body that becomes dynamic when the player
crosses a trigger x-coordinate. Acts as a Spike for collision purposes.
"""

from __future__ import annotations

from typing import Callable

import pymunk

from ..collision import CT_SPIKE
from .base import Entity


class FallingHazard(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        trigger_x: float,
        player_provider: Callable[[], pymunk.Body],
        radius: int = 18,
    ) -> None:
        super().__init__()
        self._world = world
        self._player_provider = player_provider
        self._trigger_x = trigger_x
        self._triggered = False

        self._mass = 2.0
        self._moment = pymunk.moment_for_circle(self._mass, 0, radius)
        self.body = pymunk.Body(mass=self._mass, moment=self._moment, body_type=pymunk.Body.KINEMATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, radius)
        shape.collision_type = CT_SPIKE
        shape.friction = 0.5
        self.bodies.append(self.body)
        self.shapes.append(shape)
        self.position = position
        self.radius = radius

    def update(self, dt: float) -> None:
        if self._triggered:
            return
        player_body = self._player_provider()
        if player_body is None:
            return
        if player_body.position.x >= self._trigger_x:
            # Pymunk requires bodies be removed from the space before changing
            # body_type, and mass/moment must be re-set on KINEMATIC->DYNAMIC.
            saved_pos = self.body.position
            self._world.space.remove(self.body, *self.shapes)
            self.body.body_type = pymunk.Body.DYNAMIC
            self.body.mass = self._mass
            self.body.moment = self._moment
            self.body.position = saved_pos
            self._world.space.add(self.body, *self.shapes)
            self._triggered = True

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_falling_hazard(self.body, self.radius, alpha)
