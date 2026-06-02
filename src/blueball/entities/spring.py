"""Spring — sensor strip that launches any dynamic body upward on contact."""

from __future__ import annotations

import pygame
import pymunk

from .. import collision as _col
from .base import Entity


class Spring(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        width: float,
        impulse: float,
    ) -> None:
        super().__init__()
        self.impulse = impulse
        self.position = position
        self.width = width
        cx, cy = position
        hw = width / 2
        half_thick = 8
        verts = [
            (-hw, -half_thick),
            (hw, -half_thick),
            (hw, half_thick),
            (-hw, half_thick),
        ]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        self.shape = pymunk.Poly(body, verts)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_SPRING
        self.bodies.append(body)
        self.shapes.append(self.shape)

    def draw(self, renderer, alpha: float) -> None:
        t = pygame.time.get_ticks() / 1000.0
        renderer.draw_spring(self.position, self.width, t)
