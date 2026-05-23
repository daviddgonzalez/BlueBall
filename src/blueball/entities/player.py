"""Player entity — the rolling blue ball."""

from __future__ import annotations

import math

import numpy as np
import pymunk

from .. import config
from ..agent import Action, Agent, Observation
from ..input_feel import JumpController
from .base import Entity


_MOVE_LEFT = {Action.LEFT, Action.LEFT_JUMP}
_MOVE_RIGHT = {Action.RIGHT, Action.RIGHT_JUMP}
_GROUNDED_TOL_COS = math.cos(math.radians(config.GROUNDED_NORMAL_TOLERANCE_DEG))


class Player(Entity):
    """The rolling blue ball. Wraps a single pymunk circle body and routes
    Agent actions through the JumpController.
    """

    def __init__(self, agent: Agent, spawn_xy: tuple[float, float]) -> None:
        super().__init__()
        self.agent = agent
        moment = pymunk.moment_for_circle(config.BALL_MASS, 0, config.BALL_RADIUS)
        self.body = pymunk.Body(mass=config.BALL_MASS, moment=moment)
        self.body.position = spawn_xy
        self.shape = pymunk.Circle(self.body, config.BALL_RADIUS)
        self.shape.friction = config.BALL_FRICTION
        self.shape.elasticity = config.BALL_ELASTICITY
        # collision_type=1 matches CT_PLAYER in collision.py (added in Task 8)
        self.shape.collision_type = 1
        self.bodies.append(self.body)
        self.shapes.append(self.shape)

        self.jump_ctrl = JumpController()
        self.dead = False
        self.collectibles_collected = 0
        self._contact_normals: list = []

    def die(self) -> None:
        self.dead = True
        self.alive = False

    @property
    def grounded(self) -> bool:
        """True if any current contact normal points sufficiently up."""
        # Pymunk y-down: "up" means normal.y < 0 with magnitude close to 1
        for n in self._contact_normals:
            if -n.y >= _GROUNDED_TOL_COS:
                return True
        return False

    def _refresh_contact_normals(self) -> None:
        """Walk this body's current arbiters and snapshot their normals.

        Pymunk 7 requires `each_arbiter` (the Arbiter must not outlive the
        callback), so we extract the normal inside the callback.
        """
        self._contact_normals.clear()

        def collect(arbiter):
            n = arbiter.contact_point_set.normal
            # Pymunk's contact normal points from shape_a into shape_b.
            # We want the *support normal* (pointing from the contact surface
            # toward us), so invert when we are shape_a.
            if arbiter.shapes[0] is self.shape:
                self._contact_normals.append(-n)
            else:
                self._contact_normals.append(n)

        self.body.each_arbiter(collect)

    def update(self, dt: float) -> None:
        if self.dead:
            return
        if self.body.position.y > config.FALL_DEATH_Y:
            self.die()
            return
        self._refresh_contact_normals()

        observation = self._observe()
        action = self.agent.act(observation)

        # Horizontal: torque (rolls the ball on the ground via friction) plus,
        # when airborne, a direct horizontal force so the player can change
        # direction in midair instead of just spinning in place.
        grounded = self.grounded
        air_factor = 1.0 if grounded else config.AIR_CONTROL
        if action in _MOVE_LEFT:
            self.body.torque -= config.MOVE_TORQUE * air_factor
            if not grounded:
                self.body.apply_force_at_world_point(
                    (-config.AIR_MOVE_FORCE, 0), self.body.position
                )
        if action in _MOVE_RIGHT:
            self.body.torque += config.MOVE_TORQUE * air_factor
            if not grounded:
                self.body.apply_force_at_world_point(
                    (config.AIR_MOVE_FORCE, 0), self.body.position
                )

        # Cap angular velocity so the ball doesn't infinite-spin
        av = self.body.angular_velocity
        if av > config.MAX_ANGULAR_VEL:
            self.body.angular_velocity = config.MAX_ANGULAR_VEL
        elif av < -config.MAX_ANGULAR_VEL:
            self.body.angular_velocity = -config.MAX_ANGULAR_VEL

        # Jump
        decision = self.jump_ctrl.tick(action, grounded, dt)
        if decision.fire:
            # World-frame impulse so the ball's spin doesn't rotate the impulse
            # into horizontal components. Pymunk y-down -> up is negative y.
            self.body.apply_impulse_at_world_point(
                (0, -config.JUMP_IMPULSE), self.body.position
            )
        if decision.cut and self.body.velocity.y < 0:
            vx, vy = self.body.velocity
            self.body.velocity = (vx, vy * config.JUMP_CUT_FACTOR)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_ball(self.body, alpha)

    def _observe(self) -> Observation:
        # v1 only needs grounded for the JumpController; richer fields land
        # when AI training arrives.
        return Observation(
            rays=np.zeros(8, dtype=np.float32),
            vel=np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32),
            ang_vel=self.body.angular_velocity,
            grounded=self.grounded,
            nearest_collectible=None,
        )
