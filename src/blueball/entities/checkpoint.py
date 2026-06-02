"""Checkpoint — a static sensor that records the player's respawn position on contact.

In-memory only: the handler never writes to the save file. Persisting checkpoints
(if ever needed) is the responsibility of a higher-level system, not this entity.
"""

from __future__ import annotations

import pygame
import pymunk

from ..collision import CT_CHECKPOINT
from .base import Entity


class Checkpoint(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        id: int,
        radius: int = 18,
    ) -> None:
        super().__init__()
        self._world = world
        self.id = id
        self.position = position
        self.radius = radius
        self.activated = False

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        shape = pymunk.Circle(self.body, radius)
        shape.sensor = True
        shape.collision_type = CT_CHECKPOINT
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def draw(self, renderer, alpha: float) -> None:
        t = pygame.time.get_ticks() / 1000.0
        renderer.draw_checkpoint(self.position, self.radius, t, self.activated)
