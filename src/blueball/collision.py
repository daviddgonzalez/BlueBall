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


_TOP_NORMAL_COS = math.cos(math.radians(config.GROUNDED_NORMAL_TOLERANCE_DEG))


def _find_player_entity(arbiter, world):
    """Return the Player instance involved in an arbiter, or None."""
    for shape in arbiter.shapes:
        for entity in world.entities:
            if shape in getattr(entity, "shapes", ()):
                if type(entity).__name__ == "Player":
                    return entity
    return None


def _find_entity_for_shape(shape, world):
    for entity in world.entities:
        if shape in getattr(entity, "shapes", ()):
            return entity
    return None


def register(space: pymunk.Space, world_ref) -> None:
    """Install all v1 collision handlers on `space`. `world_ref` is the World
    so handlers can mutate world-level state (level_complete, etc.).
    """

    def on_spike(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        if player is not None:
            player.die()
        return True

    def on_collectible(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            collect = getattr(entity, "collect", None)
            if collect is not None:
                collect()
                if player is not None:
                    player.collectibles_collected += 1
        return False  # sensor — no physical response

    def on_goal(arbiter, space_, data):
        world_ref.complete_level()
        return False  # sensor

    def on_ability_pickup(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if not hasattr(entity, "ability"):
                continue
            if player is not None:
                player.unlock(entity.ability)
            entity.consume()
        return False  # sensor — no physical response

    def on_boost_pad(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if not hasattr(entity, "multiplier"):
                continue
            if player is not None:
                player.receive_boost(entity.multiplier)
        return False  # sensor — no physical response

    def on_patroller(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        if player is None:
            return True
        n = arbiter.contact_point_set.normal
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            # Normal points from shape_a into shape_b. If `shape` (the patroller)
            # is shape_a, then n already points from patroller toward the player;
            # the player landed on top iff n points up (negative y in y-down).
            if arbiter.shapes[0] is shape:
                if -n.y >= _TOP_NORMAL_COS:
                    if hasattr(entity, "die"):
                        entity.die()
                    return True
            else:
                # `shape` is shape_b; n points from player into patroller.
                # Player on top iff that direction is down (positive y).
                if n.y >= _TOP_NORMAL_COS:
                    if hasattr(entity, "die"):
                        entity.die()
                    return True
            player.die()
            return True
        return True

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
                entity = _find_entity_for_shape(shape, world_ref)
                if entity is not None and hasattr(entity, "impulse"):
                    spring_entity = entity
                    break
        if spring_entity is None:
            return False
        player = _find_player_entity(arbiter, world_ref)
        if player is not None:
            player.receive_spring(spring_entity.impulse)
        else:
            # Non-player dynamic body (e.g. pushable box)
            for shape in arbiter.shapes:
                if shape.collision_type == CT_SPRING:
                    continue
                if shape.body.body_type == pymunk.Body.DYNAMIC:
                    shape.body.apply_impulse_at_local_point(
                        (0, -spring_entity.impulse * shape.body.mass), (0, 0)
                    )
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
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
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
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if not hasattr(entity, "key_id"):
                continue
            if entity._collected:
                continue
            if player is not None:
                player.collect_key(entity.key_id)
            entity._collected = True
        return False  # sensor — no physical response

    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_KEY, begin=on_key,
    )

    def on_door(arbiter, space_, data):
        # If the door is already open (shape removed), contact won't reach here;
        # but guard anyway.
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
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
