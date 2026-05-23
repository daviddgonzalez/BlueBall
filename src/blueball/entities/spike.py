"""Spike — a static triangle hazard. Instant death on contact."""

from __future__ import annotations

import pymunk

from ..collision import CT_SPIKE
from .base import Entity


class Spike(Entity):
    def __init__(self, world, position: tuple[float, float], width: int = 32, height: int = 24) -> None:
        super().__init__()
        x, y = position
        half_w = width / 2
        # Triangle pointing up: base at (x-hw, y), (x+hw, y); apex at (x, y-height)
        vertices = [(x - half_w, y), (x + half_w, y), (x, y - height)]
        shape = pymunk.Poly(world.space.static_body, vertices)
        shape.friction = 1.0
        shape.collision_type = CT_SPIKE
        self.shapes.append(shape)
        self.position = position
        self.width = width
        self.height = height

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_spike(self.position, self.width, self.height)
