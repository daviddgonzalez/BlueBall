"""Patroller — a kinematic enemy that walks back and forth on a platform."""

from __future__ import annotations

import pymunk

from .. import config
from ..collision import CT_PATROLLER
from .base import Entity


class Patroller(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        left_bound: float,
        right_bound: float,
        speed: float = config.PATROLLER_SPEED,
        size: tuple[float, float] = (24, 24),
    ) -> None:
        super().__init__()
        self._world = world
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.speed = speed
        self.size = size
        # KINEMATIC: we control velocity directly; gravity doesn't apply
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = position
        self.body.velocity = (speed, 0)
        hw, hh = size[0] / 2, size[1] / 2
        shape = pymunk.Poly(self.body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        shape.friction = 0.5
        shape.collision_type = CT_PATROLLER
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        x = self.body.position.x
        vx = self.body.velocity.x
        if vx > 0 and x >= self.right_bound:
            self.body.velocity = (-abs(self.speed), self.body.velocity.y)
        elif vx < 0 and x <= self.left_bound:
            self.body.velocity = (abs(self.speed), self.body.velocity.y)

    def die(self) -> None:
        if not self.alive:
            return
        self.alive = False
        self._remove_from_space()

    def draw(self, renderer, alpha: float) -> None:
        if self.alive:
            renderer.draw_patroller(self.body, self.size, alpha)
