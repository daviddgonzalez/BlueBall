"""Projectile — a kinematic sensor fired by a Cannon. Travels along one
horizontal direction with a pulsing speed v = V * sin^2(phase). Because sin^2
is never negative, the projectile surges forward in pulses but never reverses.
Lethal to the player on contact (CT_PROJECTILE); despawns after max_travel.
"""

from __future__ import annotations

import math

import pymunk

from .. import collision as _col
from .. import config
from .base import Entity


class Projectile(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        direction: float,
        speed: float = config.PROJECTILE_DEFAULT_SPEED,
        pulse_period_s: float = config.PROJECTILE_DEFAULT_PULSE_PERIOD_S,
        max_travel: float = 200.0,
        radius: int = config.PROJECTILE_DEFAULT_RADIUS,
    ) -> None:
        super().__init__()
        self._world = world
        # Sign of travel: +1 = right, -1 = left. sin^2 keeps |v| on one side.
        self.direction = 1.0 if direction >= 0 else -1.0
        self.speed = speed
        self.pulse_period_s = max(1e-3, pulse_period_s)
        self.max_travel = max_travel
        self.radius = radius
        self._spawn_x = position[0]
        self._t = 0.0

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        self.body = body
        self.bodies.append(body)
        self.shape = pymunk.Circle(body, radius)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_PROJECTILE
        self.shapes.append(self.shape)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        self._t += dt
        s = math.sin(math.pi * self._t / self.pulse_period_s)
        vx = self.direction * self.speed * (s * s)  # sin^2 >= 0 -> never backward
        self.body.velocity = (vx, 0)
        if abs(self.body.position.x - self._spawn_x) >= self.max_travel:
            self.despawn()

    def despawn(self) -> None:
        self.alive = False
        self._remove_from_space()

    def draw(self, renderer, alpha: float) -> None:
        if not self.alive:
            return
        renderer.draw_projectile(self.body, alpha, self.radius)
