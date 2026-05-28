"""BoostPad — a floor-strip sensor that raises the player's speed cap.

The pad is a static-body sensor rectangle. On contact the collision dispatcher
reads `pad.multiplier` and calls `player.receive_boost(m)`. The pad is NOT
consumed and can fire again the next time the player enters its volume.
"""

from __future__ import annotations

import pymunk

from .. import config
from ..collision import CT_BOOST_PAD
from .base import Entity


class BoostPad(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        width: int = 128,
        multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self._world = world
        self.position = position
        self.width = width
        self.multiplier = multiplier

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        hw = width / 2
        hh = config.BOOST_PAD_THICKNESS / 2
        # Extend the sensor upward (toward -y) so a ball gliding just over the
        # pad still triggers it; the visible strip stays at the pad surface.
        catch = config.BOOST_PAD_CATCH_HEIGHT
        shape = pymunk.Poly(self.body, [(-hw, -hh - catch), (hw, -hh - catch), (hw, hh), (-hw, hh)])
        shape.sensor = True
        shape.collision_type = CT_BOOST_PAD
        self.bodies.append(self.body)
        self.shapes.append(shape)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_boost_pad(self.position, self.width)
