"""SwingingHazard — pendulum bob on a PinJoint rope. Instant death on contact."""

from __future__ import annotations

import math

import pymunk

from ..collision import CT_SWINGING
from .base import Entity


class SwingingHazard(Entity):
    def __init__(
        self,
        world,
        anchor_pos: tuple[float, float],
        rope_length: float,
        bob_mass: float,
        bob_radius: int = 14,
        initial_angle_deg: float = 0.0,
    ) -> None:
        super().__init__()

        # Static anchor body at anchor_pos
        self.anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_body.position = anchor_pos

        # Dynamic bob body; spawn offset from anchor by rope_length at initial_angle
        angle_rad = math.radians(initial_angle_deg)
        bob_x = anchor_pos[0] + rope_length * math.sin(angle_rad)
        bob_y = anchor_pos[1] + rope_length * math.cos(angle_rad)

        moment = pymunk.moment_for_circle(bob_mass, 0, bob_radius)
        self.bob_body = pymunk.Body(mass=bob_mass, moment=moment)
        self.bob_body.position = (bob_x, bob_y)

        # Bob circle shape with CT_SWINGING
        self.bob_shape = pymunk.Circle(self.bob_body, bob_radius)
        self.bob_shape.collision_type = CT_SWINGING
        self.bob_shape.friction = 0.5

        # PinJoint links anchor to bob
        joint = pymunk.PinJoint(self.anchor_body, self.bob_body, (0, 0), (0, 0))

        # Register for world.add_entity to add to space
        self.bodies.append(self.anchor_body)
        self.bodies.append(self.bob_body)
        self.shapes.append(self.bob_shape)
        self.constraints.append(joint)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_swinging_hazard(self.anchor_body, self.bob_body, self.bob_shape.radius, alpha)
