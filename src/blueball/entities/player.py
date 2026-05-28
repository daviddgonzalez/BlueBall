"""Player entity — the rolling blue ball."""

from __future__ import annotations

import math

import numpy as np
import pymunk

from .. import config
from ..abilities import Ability
from ..agent import Action, Agent, HitType, Observation, _CT_TO_HITTYPE
from ..input_feel import JumpController
from .base import Entity


_PLAYER_RAY_GROUP = 1

# 8 ray directions, 45° apart, starting due-right (east) and going counter-
# clockwise.  Pre-computed once at import time.
_TWO_PI_OVER_8 = 2 * math.pi / 8
_RAY_ANGLES: tuple[tuple[float, float], ...] = tuple(
    (math.cos(i * _TWO_PI_OVER_8), math.sin(i * _TWO_PI_OVER_8))
    for i in range(8)
)

# Entity type name buckets for nearest-entity scan.
_PICKUP_TYPENAMES = frozenset({
    "Collectible", "AbilityPickup", "BoostPad", "Key", "Spring", "Checkpoint",
})
_HAZARD_TYPENAMES = frozenset({
    "Spike", "FallingHazard", "Patroller", "SwingingHazard", "Charger",
})


def _abilities_to_bitfield(abilities: set[Ability]) -> int:
    """Return an integer bitfield where bit *i* is set if the *i*-th member
    of the Ability enum (in declaration order) is present in *abilities*."""
    result = 0
    for i, member in enumerate(Ability):
        if member in abilities:
            result |= (1 << i)
    return result

_MOVE_LEFT = {Action.LEFT, Action.LEFT_JUMP}
_MOVE_RIGHT = {Action.RIGHT, Action.RIGHT_JUMP}
_GROUNDED_TOL_COS = math.cos(math.radians(config.GROUNDED_NORMAL_TOLERANCE_DEG))


class Player(Entity):
    """The rolling blue ball. Wraps a single pymunk circle body and routes
    Agent actions through the JumpController.
    """

    def __init__(
        self,
        agent: Agent,
        spawn_xy: tuple[float, float],
        abilities: set[Ability] | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        moment = pymunk.moment_for_circle(config.BALL_MASS, 0, config.BALL_RADIUS)
        self.body = pymunk.Body(mass=config.BALL_MASS, moment=moment)
        self.body.position = spawn_xy
        self.shape = pymunk.Circle(self.body, config.BALL_RADIUS)
        self.shape.filter = pymunk.ShapeFilter(group=_PLAYER_RAY_GROUP)
        self._ray_filter = pymunk.ShapeFilter(group=_PLAYER_RAY_GROUP)
        self.shape.friction = config.BALL_FRICTION
        self.shape.elasticity = config.BALL_ELASTICITY
        # collision_type=1 matches CT_PLAYER in collision.py (added in Task 8)
        self.shape.collision_type = 1
        self.bodies.append(self.body)
        self.shapes.append(self.shape)

        # Share the abilities set by reference with the JumpController so
        # later unlocks land in the controller without a re-push.
        self.abilities: set[Ability] = abilities if abilities is not None else set()
        self.jump_ctrl = JumpController(abilities=self.abilities)
        self.dead = False
        self.collectibles_collected = 0
        self._boost_multiplier: float = 1.0
        self._aerial_since_pickup: bool = False
        self._contact_normals: list = []
        self.keys_held: int = 0
        self.respawn_xy: tuple[float, float] | None = None

    def die(self) -> None:
        self.dead = True
        self.alive = False

    def unlock(self, ability: Ability) -> None:
        """Add `ability` to this player's set (visible to JumpController on
        the next tick via the shared reference). In-memory only — persistence
        happens at level-complete time so dying mid-run reverts the unlock.

        Idempotent: re-picking an already-unlocked ability is a no-op. New
        unlocks also notify the JumpController so a mid-air pickup grants its
        effect immediately rather than waiting for the next ground→air cycle.
        """
        if ability in self.abilities:
            return
        self.abilities.add(ability)
        self.jump_ctrl.on_ability_added(ability)

    def collect_key(self, key_id: int) -> None:
        """Set the bit for `key_id` in keys_held. Idempotent."""
        self.keys_held |= (1 << key_id)

    def has_key(self, key_id: int) -> bool:
        return bool(self.keys_held & (1 << key_id))

    def receive_spring(self, impulse: float) -> None:
        """Vertical upward impulse, mass-scaled so the resulting delta-v
        is the same regardless of body mass. Pymunk y-down → up is -y.
        World-frame so the ball's rotation doesn't twist the impulse into
        a horizontal component opposite the roll direction."""
        self.body.apply_impulse_at_world_point(
            (0, -impulse * self.body.mass), self.body.position
        )
        # A spring launch is a "landing" — refresh the air jump so the double
        # jump works off a spring even though it never registers as grounded.
        self.jump_ctrl.refresh_air_jumps()

    def refresh_air_jumps(self) -> None:
        """Restore the air jump after landing on a non-ground surface (e.g.
        stomping an enemy) that doesn't produce a sustained 'grounded' tick."""
        self.jump_ctrl.refresh_air_jumps()

    def receive_boost(self, multiplier: float) -> None:
        """Apply a boost-pad's multiplier, take-the-max'd against any active
        boost. Arms "ends on next landing" tracking: if we're already airborne
        when the boost lands, the next grounded tick ends it; if we're grounded
        we have to jump and land before it ends.

        Also snaps horizontal velocity (and matching angular velocity) up to
        the new cap in the direction of current motion, so the boost feels
        immediate instead of waiting for ground force to push the ball up to
        the higher cap.
        """
        # Landing on a boost pad refreshes the air jump like any landing.
        self.jump_ctrl.refresh_air_jumps()
        if multiplier > self._boost_multiplier:
            self._boost_multiplier = multiplier
            self._aerial_since_pickup = not self.grounded
            vx, vy = self.body.velocity
            ang = self.body.angular_velocity
            new_max_speed = config.MAX_LINEAR_SPEED * multiplier
            new_max_ang = config.MAX_ANGULAR_VEL * multiplier
            kick = config.BOOST_PAD_KICK_FACTOR
            if vx > 0:
                target_vx = vx + (new_max_speed - vx) * kick
                target_ang = ang + (new_max_ang - ang) * kick
                self.body.velocity = (max(vx, target_vx), vy)
                self.body.angular_velocity = max(ang, target_ang)
            elif vx < 0:
                target_vx = vx + (-new_max_speed - vx) * kick
                target_ang = ang + (-new_max_ang - ang) * kick
                self.body.velocity = (min(vx, target_vx), vy)
                self.body.angular_velocity = min(ang, target_ang)

    def _update_boost(self, grounded: bool) -> None:
        """Per-tick: if a boost is active, track aerial state and clear on the
        first airborne→grounded transition."""
        if self._boost_multiplier <= 1.0:
            return
        if not grounded:
            self._aerial_since_pickup = True
        elif self._aerial_since_pickup:
            self._boost_multiplier = 1.0
            self._aerial_since_pickup = False

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
        self._update_boost(self.grounded)

        observation = self._observe()
        action = self.agent.act(observation)

        # Horizontal: torque (so the ball visibly spins) plus a direct
        # horizontal force. On the ground the force bypasses the friction
        # ceiling for snappy reversals. In the air the force is asymmetric:
        # high when the input opposes current velocity (BRAKE / correction),
        # low when input matches velocity (ACCEL), so the player can fix a
        # bad jump arc but can't freely accelerate horizontally in midair.
        grounded = self.grounded
        air_factor = 1.0 if grounded else config.AIR_CONTROL
        vx = self.body.velocity.x
        if action in _MOVE_LEFT:
            self.body.torque -= config.MOVE_TORQUE * air_factor
            if grounded:
                force = config.GROUND_MOVE_FORCE
            else:
                force = config.AIR_MOVE_FORCE_BRAKE if vx > 0 else config.AIR_MOVE_FORCE_ACCEL
            if force > 0:
                self.body.apply_force_at_world_point(
                    (-force, 0), self.body.position
                )
        if action in _MOVE_RIGHT:
            self.body.torque += config.MOVE_TORQUE * air_factor
            if grounded:
                force = config.GROUND_MOVE_FORCE
            else:
                force = config.AIR_MOVE_FORCE_BRAKE if vx < 0 else config.AIR_MOVE_FORCE_ACCEL
            if force > 0:
                self.body.apply_force_at_world_point(
                    (force, 0), self.body.position
                )

        # Cap angular velocity so the ball doesn't infinite-spin. Boost pads
        # scale this in lockstep with the linear cap so the ball never visually
        # slips (effective_max_ang_vel * BALL_RADIUS ≈ effective_max_speed).
        max_ang_vel = config.MAX_ANGULAR_VEL * self._boost_multiplier
        av = self.body.angular_velocity
        if av > max_ang_vel:
            self.body.angular_velocity = max_ang_vel
        elif av < -max_ang_vel:
            self.body.angular_velocity = -max_ang_vel

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

        # Cap horizontal velocity only. A magnitude cap would clip a spring's
        # upward kick proportionally with horizontal velocity, dragging |vx|
        # down whenever the player hit a spring at speed. Vertical velocity
        # is naturally bounded by gravity, JUMP_IMPULSE, and FALL_DEATH_Y.
        max_vx = config.MAX_LINEAR_SPEED * self._boost_multiplier
        vx, vy = self.body.velocity
        if vx > max_vx:
            self.body.velocity = (max_vx, vy)
        elif vx < -max_vx:
            self.body.velocity = (-max_vx, vy)

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_ball(self.body, alpha)

    def _nearest_entity_delta(
        self, type_names: frozenset[str]
    ) -> tuple[float, float] | None:
        """Return the (dx, dy) world-frame delta to the closest entity whose
        class name is in *type_names*, or None if no such entity exists.

        Position is taken from ``e.body.position`` when the entity has a
        ``body`` attribute, otherwise from ``e.position``.
        """
        best_sq: float | None = None
        best_delta: tuple[float, float] | None = None
        px, py = self.body.position

        for e in self._world.entities:
            if type(e).__name__ not in type_names:
                continue
            if hasattr(e, "body"):
                ex, ey = e.body.position
            else:
                ex, ey = e.position
            dx = ex - px
            dy = ey - py
            sq = dx * dx + dy * dy
            if best_sq is None or sq < best_sq:
                best_sq = sq
                best_delta = (dx, dy)

        return best_delta

    def _observe(self) -> Observation:
        """Build an Observation from 8 raycasts + nearest-entity scans.

        When the player has not yet been added to a World (``_world`` is
        absent) — e.g. during unit tests that construct a Player without a
        World — raycasts are skipped (all 1.0 / MISS) and nearest-entity
        deltas are returned as None.
        """
        rays = np.empty(8, dtype=np.float32)
        hit_types = np.empty(8, dtype=np.int8)

        world = getattr(self, "_world", None)

        if world is not None:
            pos = self.body.position
            for i, (cx, cy) in enumerate(_RAY_ANGLES):
                end = pos + pymunk.Vec2d(cx, cy) * config.MAX_RAY_LEN
                hit = world.space.segment_query_first(
                    pos, end, 0.5, self._ray_filter
                )
                if hit is None:
                    rays[i] = 1.0
                    hit_types[i] = HitType.MISS
                else:
                    rays[i] = float(hit.alpha)
                    hit_types[i] = _CT_TO_HITTYPE.get(
                        hit.shape.collision_type, HitType.GROUND
                    )
            nearest_pickup = self._nearest_entity_delta(_PICKUP_TYPENAMES)
            nearest_hazard = self._nearest_entity_delta(_HAZARD_TYPENAMES)
        else:
            rays[:] = 1.0
            hit_types[:] = HitType.MISS
            nearest_pickup = None
            nearest_hazard = None

        return Observation(
            rays=rays,
            ray_hit_types=hit_types,
            vel=np.array(
                [self.body.velocity.x, self.body.velocity.y], dtype=np.float32
            ),
            ang_vel=self.body.angular_velocity,
            grounded=self.grounded,
            nearest_pickup=nearest_pickup,
            nearest_hazard=nearest_hazard,
            abilities=_abilities_to_bitfield(self.abilities),
            keys_held=self.keys_held,
        )
