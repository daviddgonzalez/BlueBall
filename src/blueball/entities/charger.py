"""Charger — kinematic enemy with directional FOV that charges at the player
when seen, otherwise patrols a bounded segment.
"""

from __future__ import annotations

import math

import pymunk

from .. import collision as _col
from .. import config
from .base import Entity


# Match-everything filter for the line-of-sight query. ShapeFilter is an
# immutable value, so a single shared instance is safe and avoids a per-tick
# allocation on the hot charge path.
_LOS_FILTER = pymunk.ShapeFilter()


class Charger(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        left_bound: float,
        right_bound: float,
        facing: str = "right",
        sight_range: float = config.CHARGER_DEFAULT_SIGHT_RANGE,
        sight_arc_deg: float = config.CHARGER_DEFAULT_SIGHT_ARC_DEG,
        charge_speed: float = config.CHARGER_DEFAULT_CHARGE_SPEED,
        patrol_speed: float = config.CHARGER_DEFAULT_PATROL_SPEED,
        radius: int = 12,
    ) -> None:
        super().__init__()
        if facing not in ("left", "right"):
            raise ValueError(f"facing must be 'left' or 'right'; got {facing!r}")
        self.facing = facing
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.sight_range = sight_range
        self.sight_arc_cos = math.cos(math.radians(sight_arc_deg / 2))
        self.charge_speed = charge_speed
        self.patrol_speed = patrol_speed
        self.state = "patrol"
        self.alive = True

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        self.body = body
        self.bodies.append(body)
        self.shape = pymunk.Circle(body, radius)
        self.shape.collision_type = _col.CT_CHARGER
        self.shape.friction = 0.5
        self.shapes.append(self.shape)

        body.velocity = (patrol_speed if facing == "right" else -patrol_speed, 0)
        self._world = world

    def die(self) -> None:
        self.alive = False
        self._remove_from_space()

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        player = self._world.player
        if player is None:
            return self._patrol_tick()
        # FOV check
        dx = player.body.position.x - self.body.position.x
        dy = player.body.position.y - self.body.position.y
        dist = math.hypot(dx, dy)
        in_range = dist <= self.sight_range
        if in_range and dist > 0:
            facing_dir = 1.0 if self.facing == "right" else -1.0
            cos_to_player = (dx * facing_dir + dy * 0) / dist
            in_cone = cos_to_player >= self.sight_arc_cos
        else:
            in_cone = False
        # LOS: segment query from charger to player; if anything static is hit, LOS blocked
        los_clear = True
        if in_range and in_cone:
            hit = self._world.space.segment_query_first(
                (self.body.position.x, self.body.position.y),
                (player.body.position.x, player.body.position.y),
                0.5,
                _LOS_FILTER,
            )
            if hit is not None and hit.shape.body.body_type == pymunk.Body.STATIC:
                los_clear = False
        if in_range and in_cone and los_clear:
            self.state = "charge"
            dir_x = 1.0 if dx > 0 else -1.0
            self.body.velocity = (self.charge_speed * dir_x, 0)
            # A charge pursues the player past the patrol bounds; only patrol
            # turnaround is bounded.
        else:
            self.state = "patrol"
            self._patrol_tick()
            # Respect patrol bounds (turnaround) only while patrolling.
            if self.body.position.x <= self.left_bound:
                self.body.velocity = (abs(self.body.velocity.x) or self.patrol_speed, 0)
            elif self.body.position.x >= self.right_bound:
                self.body.velocity = (-(abs(self.body.velocity.x) or self.patrol_speed), 0)

    def _patrol_tick(self) -> None:
        # Maintain patrol speed magnitude in the current direction
        vx = self.body.velocity.x
        if vx >= 0:
            self.body.velocity = (self.patrol_speed, 0)
        else:
            self.body.velocity = (-self.patrol_speed, 0)

    def draw(self, renderer, alpha: float) -> None:
        if not self.alive:
            return
        renderer.draw_charger(self.body, alpha, self.state)
