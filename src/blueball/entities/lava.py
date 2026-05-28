"""Lava — a wide kinematic sensor that rises from below at constant speed.
Kills the player on contact. Spawned by the level loader when the level JSON
contains an optional `"lava"` block.
"""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class Lava(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        width: float,
        rise_speed: float,
        height: float = 600,
    ) -> None:
        super().__init__()
        self.position = position
        self.width = width
        self.rise_speed = rise_speed
        self.height = height
        # Kinematic body — velocity-driven, gravity ignores it. In pymunk y-down
        # "up" is -y, so a positive rise_speed translates to negative y velocity.
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.velocity = (0, -rise_speed)
        self.body = body
        self.bodies.append(body)
        # Box shape extending DOWN from the body center so the visible top edge
        # is the lava's surface. Height is intentionally large so the bottom is
        # always off-screen.
        hw = width / 2
        # Vertices in body-local space: top edge at y=0, bottom edge at y=height
        verts = [(-hw, 0), (hw, 0), (hw, height), (-hw, height)]
        self.shape = pymunk.Poly(body, verts)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_LAVA
        self.shapes.append(self.shape)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_lava(self.body, alpha, self.width, self.height)
