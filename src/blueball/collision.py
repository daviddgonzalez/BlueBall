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
        # A dead player's body stays in the pymunk space after die() and can
        # drift through the goal sensor under gravity. Without this guard:
        # TrainScene would award +200 reached_goal on top of -100 died =
        # +100 net for "die near the goal"; PlayScene would end-of-level
        # on a corpse's body crossing the flag.
        player = _find_player_entity(arbiter, world_ref)
        if player is None or player.dead:
            return False
        player.reached_goal = True
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

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_SPIKE, begin=on_spike)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_COLLECTIBLE, begin=on_collectible)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_GOAL, begin=on_goal)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_PATROLLER, begin=on_patroller)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_ABILITY_PICKUP, begin=on_ability_pickup)
    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_BOOST_PAD, begin=on_boost_pad)
