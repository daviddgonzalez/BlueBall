"""Spike — a static triangle hazard. Instant death on contact."""

from __future__ import annotations

import pymunk

from ..collision import CT_SPIKE
from .base import Entity


def _spike_verts(width: int, height: int, orientation: str) -> list[tuple[float, float]]:
    hw = width / 2
    if orientation == "up":
        return [(-hw, 0), (hw, 0), (0, -height)]
    if orientation == "down":
        return [(-hw, 0), (hw, 0), (0, height)]
    if orientation == "left":
        return [(0, -hw), (0, hw), (-height, 0)]
    if orientation == "right":
        return [(0, -hw), (0, hw), (height, 0)]
    raise ValueError(
        f"Spike orientation must be 'up', 'down', 'left', or 'right'; got {orientation!r}"
    )


class Spike(Entity):
    def __init__(self, world, position: tuple[float, float], width: int = 32, height: int = 24, orientation: str = "up") -> None:
        super().__init__()
        x, y = position
        local_verts = _spike_verts(width, height, orientation)
        vertices = [(x + lx, y + ly) for lx, ly in local_verts]
        shape = pymunk.Poly(world.space.static_body, vertices)
        shape.friction = 1.0
        shape.collision_type = CT_SPIKE
        self.shapes.append(shape)
        self.position = position
        self.width = width
        self.height = height
        self.orientation = orientation

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_spike(self.position, self.width, self.height)
