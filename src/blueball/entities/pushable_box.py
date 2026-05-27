"""PushableBox — dynamic body that the player can push laterally.

Pymunk's default solid contact provides the push reaction; no custom
collision handler is required for CT_PUSHABLE.
"""

from __future__ import annotations

import pymunk

from ..collision import CT_PUSHABLE
from .base import Entity


class PushableBox(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        size: float = 32,
        mass: float = 0.5,
    ) -> None:
        super().__init__()
        self.size = size
        moment = pymunk.moment_for_box(mass, (size, size))
        self.body = pymunk.Body(mass=mass, moment=moment)
        self.body.position = position
        hs = size / 2
        self.shape = pymunk.Poly(
            self.body,
            [(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)],
        )
        self.shape.friction = 0.6
        self.shape.collision_type = CT_PUSHABLE
        self.bodies.append(self.body)
        self.shapes.append(self.shape)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_pushable_box(self.body, alpha, self.size)
