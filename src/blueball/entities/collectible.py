"""Collectible — a sensor circle that disappears on contact."""

from __future__ import annotations

import pymunk

from .. import config
from ..collision import CT_COLLECTIBLE
from .base import Entity


class Collectible(Entity):
    def __init__(self, world, position: tuple[float, float]) -> None:
        super().__init__()
        self._world = world
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, config.COLLECTIBLE_RADIUS)
        shape.sensor = True
        shape.collision_type = CT_COLLECTIBLE
        self.bodies.append(self.body)
        self.shapes.append(shape)
        self.position = position
        self._collected = False

    def collect(self) -> None:
        if self._collected:
            return
        self._collected = True
        self.alive = False
        self._remove_from_space()

    def draw(self, renderer, alpha: float) -> None:
        if self.alive:
            renderer.draw_collectible(self.position)
