"""Collision dispatcher — single source of truth for what happens when
two specific entity types touch.
"""

from __future__ import annotations

import math

import pymunk

from . import config

CT_PLAYER = 1
CT_SPIKE = 2
CT_PATROLLER = 3
CT_COLLECTIBLE = 4
CT_GOAL = 5
CT_BOOST_PAD = 6
CT_ABILITY_PICKUP = 7
CT_ONE_WAY = 8
CT_SPRING = 9
CT_PUSHABLE = 10
CT_SWINGING = 11
CT_CHARGER = 12
CT_CHECKPOINT = 13
CT_KEY = 14
CT_DOOR = 15
CT_LAVA = 16
CT_PROJECTILE = 17

# Shared shape-filter group for all Players. Shapes that share a non-zero
# group don't collide with each other (pymunk semantics), so N agents in
# one World coexist non-interactively without per-agent group assignment.
#
# WARNING: this constant is exclusive to Player shapes. Do NOT assign
# group=PLAYER_GROUP to any other entity — pymunk's narrow-phase would
# silently suppress collisions between that entity and every Player,
# bypassing all per-pair handlers below. If a future system needs its own
# non-collision group, allocate a new constant (e.g. GHOST_GROUP = 98).
PLAYER_GROUP = 99


_TOP_NORMAL_COS = math.cos(math.radians(config.GROUNDED_NORMAL_TOLERANCE_DEG))


def _player_and_others(arbiter, world):
    """Yield ``(player, shape, entity)`` for each non-player entity shape in the
    arbiter. ``player`` is ``world.player`` (every handler that uses this is
    registered with CT_PLAYER on one side, so the player is in the arbiter).
    Shapes with no owning entity, and the player's own shape, are skipped —
    reproducing the old ``_find_entity_for_shape`` scan's None/player filter.
    """
    player = world.player
    for shape in arbiter.shapes:
        entity = world._shape_to_entity.get(shape)
        if entity is None or entity is player:
            continue
        yield player, shape, entity


def register(space: pymunk.Space, world_ref) -> None:
    """Install all v1 collision handlers on `space`. `world_ref` is the World
    so handlers can mutate world-level state (level_complete, etc.).
    """

    def _lethal(solid: bool):
        """Build a 'kill the player on contact' handler. `solid` is the begin
        return value: True for solid hazards (spike, swinging), False for
        sensors (lava, projectile) which produce no physical response."""
        def handler(arbiter, space_, data):
            player = world_ref.player
            if player is not None:
                player.die()
            return solid
        return handler

    on_spike = _lethal(solid=True)
    on_swinging = _lethal(solid=True)
    on_lava = _lethal(solid=False)
    on_projectile = _lethal(solid=False)

    def on_collectible(arbiter, space_, data):
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            collect = getattr(entity, "collect", None)
            if collect is not None:
                collect()
                if player is not None:
                    player.collectibles_collected += 1
        return False  # sensor — no physical response

    def on_goal(arbiter, space_, data):
        # A dead player's body stays in the pymunk space after die() and can
        # drift through the goal sensor under gravity. Without this guard:
        # TrainScene would award +200 reached_goal on top of -100 died =
        # +100 net for "die near the goal"; PlayScene would end-of-level
        # on a corpse's body crossing the flag.
        player = world_ref.player
        if player is None or player.dead:
            return False
        player.reached_goal = True
        world_ref.complete_level()
        world_ref.emit_sound("fanfare")
        return False  # sensor

    def on_ability_pickup(arbiter, space_, data):
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            if not hasattr(entity, "ability"):
                continue
            if player is not None:
                player.unlock(entity.ability)
            entity.consume()
        return False  # sensor — no physical response

    def on_boost_pad(arbiter, space_, data):
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            if not hasattr(entity, "multiplier"):
                continue
            if player is not None:
                player.receive_boost(
                    entity.multiplier, getattr(entity, "direction", 1.0)
                )
                world_ref.emit_sound("whoosh")
        return False  # sensor — no physical response

    def _on_stompable(arbiter):
        """Shared patroller/charger handler: stomp from the top kills the enemy
        and refreshes the air jump; any other contact kills the player. Both
        shape orderings are handled so the normal-sign test stays correct
        regardless of which side the enemy is registered as."""
        player = world_ref.player
        if player is None:
            return True
        n = arbiter.contact_point_set.normal
        for shape in arbiter.shapes:
            entity = world_ref._shape_to_entity.get(shape)
            if entity is None or entity is player:
                continue
            # Normal points from shape_a into shape_b. If the enemy is shape_a,
            # n points enemy->player and a top stomp means n points up (negative
            # y in y-down); if the enemy is shape_b, the sign is mirrored.
            if arbiter.shapes[0] is shape:
                stomped = -n.y >= _TOP_NORMAL_COS
            else:
                stomped = n.y >= _TOP_NORMAL_COS
            if stomped:
                if hasattr(entity, "die"):
                    entity.die()
                player.refresh_air_jumps()  # stomping is a landing
                return True
            player.die()
            return True
        return True

    def on_patroller(arbiter, space_, data):
        return _on_stompable(arbiter)

    def on_one_way_presolve(arbiter, space_, data):
        # Identify the dynamic body (player or pushable box).  In pymunk y-down,
        # rising means velocity.y < 0.  Suppress the collision (pass through)
        # while the body is moving upward; allow it (solid landing) otherwise.
        for shape in arbiter.shapes:
            if shape.body.body_type == pymunk.Body.DYNAMIC:
                if shape.body.velocity.y < 0:
                    arbiter.process_collision = False
                    return

    def on_spring(arbiter, space_, data):
        # Find the Spring entity to get impulse value.
        spring_entity = None
        for shape in arbiter.shapes:
            if shape.collision_type == CT_SPRING:
                entity = world_ref._shape_to_entity.get(shape)
                if entity is not None and hasattr(entity, "impulse"):
                    spring_entity = entity
                    break
        if spring_entity is None:
            return False
        # on_spring is ALSO registered for (PUSHABLE, SPRING), an arbiter with no
        # player — so resolve the player from THIS arbiter's shapes, not the
        # cached world.player (which would wrongly route the box down the player
        # path). Player present only when a CT_PLAYER shape is in the arbiter.
        player = next(
            (world_ref._shape_to_entity.get(s) for s in arbiter.shapes
             if getattr(s, "collision_type", None) == CT_PLAYER),
            None,
        )
        if player is not None:
            player.receive_spring(spring_entity.impulse)
            world_ref.emit_sound("spring")
        else:
            # Non-player dynamic body (e.g. pushable box). Set a floor launch
            # speed (same as the player path) so the bounce is consistent
            # regardless of incoming velocity or body mass.
            for shape in arbiter.shapes:
                if shape.collision_type == CT_SPRING:
                    continue
                if shape.body.body_type == pymunk.Body.DYNAMIC:
                    bvx, bvy = shape.body.velocity
                    shape.body.velocity = (bvx, min(bvy, -spring_entity.impulse))
        return False  # sensor — no physical response

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_SPRING, begin=on_spring)
    space.on_collision(collision_type_a=CT_PUSHABLE, collision_type_b=CT_SPRING, begin=on_spring)

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_SPIKE, begin=on_spike)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_COLLECTIBLE, begin=on_collectible)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_GOAL, begin=on_goal)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_PATROLLER, begin=on_patroller)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_ABILITY_PICKUP, begin=on_ability_pickup)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_BOOST_PAD, begin=on_boost_pad)
    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_ONE_WAY,
        pre_solve=on_one_way_presolve,
    )
    def on_checkpoint(arbiter, space_, data):
        from .levels.chunks.flat import GROUND_Y
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            if not hasattr(entity, "activated"):
                continue
            if player is not None:
                player.respawn_xy = (
                    shape.body.position.x,
                    GROUND_Y - config.BALL_RADIUS - 4,
                )
            entity.activated = True
        return False  # sensor — no physical response

    space.on_collision(
        collision_type_a=CT_PUSHABLE, collision_type_b=CT_ONE_WAY,
        pre_solve=on_one_way_presolve,
    )
    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_CHECKPOINT, begin=on_checkpoint,
    )

    def on_key(arbiter, space_, data):
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            if not hasattr(entity, "key_id"):
                continue
            if entity._collected:
                continue
            if player is not None:
                player.collect_key(entity.key_id)
                world_ref.emit_sound("key")
            entity._collected = True
        return False  # sensor — no physical response

    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_KEY, begin=on_key,
    )

    def on_door(arbiter, space_, data):
        # If the door is already open (shape removed), contact won't reach here;
        # but guard anyway.
        for player, shape, entity in _player_and_others(arbiter, world_ref):
            if not hasattr(entity, "key_id"):
                continue
            if entity.is_open:
                # Already open — pass through (sensor-like)
                arbiter.process_collision = False
                return False
            if player is not None and player.has_key(entity.key_id):
                entity._opening = True
                # Shape is still in space this tick; remove it in next update().
                # Suppress physical response where the API supports it.
                arbiter.process_collision = False
                return False
            # No key — door is solid
            return True
        return True

    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_DOOR, begin=on_door,
    )

    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_SWINGING, begin=on_swinging,
    )

    def on_charger(arbiter, space_, data):
        return _on_stompable(arbiter)

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_CHARGER, begin=on_charger)

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_LAVA, begin=on_lava)

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_PROJECTILE, begin=on_projectile)
