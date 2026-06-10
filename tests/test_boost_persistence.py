"""Boost lifetime: the boost must survive INCIDENTAL airborne moments (a seam
hop, a bump, a slope) and only be revoked after a DELIBERATE jump's landing.

Regression for the "first boost pad doesn't register" bug: a fast ball trips on
a chunk-ground seam right after the pad, goes airborne for a few frames, and the
old clear-on-first-landing rule revoked the boost before it could accelerate the
player — so the pad did nothing.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blueball import config
from blueball.abilities import Ability
from blueball.agent import Action
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.chunks.boost_pad import BoostPadChunk
from blueball.levels.chunks.flat import GROUND_Y, Flat
from blueball.world import World


class StubAgent:
    def __init__(self, a=Action.IDLE):
        self._a = a
    def act(self, obs):
        return self._a


def _boosted_player():
    p = Player(agent=StubAgent(), spawn_xy=(0.0, 0.0), abilities={Ability.DOUBLE_JUMP})
    p._boost_multiplier = 2.0
    p._aerial_since_pickup = False
    p._jumped_since_boost = False
    p._boost_just_received = False
    p._boost_timer = 100.0  # large so the grounded countdown isn't the variable here
    return p


def test_incidental_airborne_keeps_boost():
    """Airborne with NO deliberate jump (seam/bump hop), then land -> boost stays."""
    p = _boosted_player()
    p._update_boost(grounded=False, dt=0.0)  # tripped airborne, never jumped
    p._update_boost(grounded=True, dt=0.0)   # landed
    assert p._boost_multiplier == 2.0


def test_deliberate_jump_clears_boost_on_landing():
    """A real jump fired while boosted -> boost is consumed on the landing."""
    p = _boosted_player()
    p._jumped_since_boost = True             # a deliberate jump fired
    p._update_boost(grounded=False, dt=0.0)  # airborne from the jump
    p._update_boost(grounded=True, dt=0.0)   # landed
    assert p._boost_multiplier == 1.0


def test_boost_pad_on_chunk_ground_gives_speed_gain():
    """End-to-end: rolling over a pad on chunk-built ground (which has seams)
    must actually speed the player past the normal cap, not fizzle."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    x = 0.0
    x += Flat(width_tiles=2).build(w, x_offset=x)
    x += BoostPadChunk(width_tiles=3, multiplier=1.8).build(w, x_offset=x)
    x += Flat(width_tiles=40).build(w, x_offset=x)
    p = Player(agent=StubAgent(Action.RIGHT), spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    max_vx = 0.0
    for _ in range(400):
        w.substep()
        max_vx = max(max_vx, p.body.velocity[0])
    assert max_vx > config.MAX_LINEAR_SPEED + 5, (
        f"boost gave no speed gain (max_vx={max_vx:.0f}, cap={config.MAX_LINEAR_SPEED})")
