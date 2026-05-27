"""MovingPlatform — kinematic body oscillating between two waypoints."""

from __future__ import annotations

import pymunk

from .base import Entity


class MovingPlatform(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        length: float,
        axis: str = "x",
        range_px: float = 160,
        speed: float = 80,
    ) -> None:
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        self.axis = axis
        self.range_px = range_px
        self.speed = speed
        self.length = length
        self._spawn = position
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        hl = length / 2
        # Lay the segment horizontally regardless of axis — it's the platform's
        # visible width; vertical movers also use a horizontal top edge.
        self.shape = pymunk.Segment(body, (-hl, 0), (hl, 0), 5)
        self.shape.friction = 1.0
        self.bodies.append(body)
        self.shapes.append(self.shape)
        # Initial velocity along axis
        if axis == "x":
            body.velocity = (speed, 0)
        else:
            body.velocity = (0, speed)
        self.body = body

    def update(self, dt: float) -> None:
        if self.axis == "x":
            delta = self.body.position.x - self._spawn[0]
            if delta > self.range_px / 2 and self.body.velocity.x > 0:
                self.body.velocity = (-self.speed, 0)
            elif delta < -self.range_px / 2 and self.body.velocity.x < 0:
                self.body.velocity = (self.speed, 0)
        else:
            delta = self.body.position.y - self._spawn[1]
            if delta > self.range_px / 2 and self.body.velocity.y > 0:
                self.body.velocity = (0, -self.speed)
            elif delta < -self.range_px / 2 and self.body.velocity.y < 0:
                self.body.velocity = (0, self.speed)

    def draw(self, renderer, alpha: float) -> None:
        pass
