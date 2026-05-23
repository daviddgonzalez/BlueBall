"""Goal — sensor rectangle that ends the level on contact."""

from __future__ import annotations

import pymunk

from ..collision import CT_GOAL
from .base import Entity


class Goal(Entity):
    def __init__(self, world, position: tuple[float, float], width: int = 40, height: int = 80) -> None:
        super().__init__()
        x, y = position
        hw, hh = width / 2, height / 2
        vertices = [(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh), (x - hw, y + hh)]
        shape = pymunk.Poly(world.space.static_body, vertices)
        shape.sensor = True
        shape.collision_type = CT_GOAL
        self.shapes.append(shape)
        self.position = position
        self.width = width
        self.height = height

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_goal(self.position, self.width, self.height)
