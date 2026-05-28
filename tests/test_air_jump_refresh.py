"""Landing on a non-ground surface (spring, enemy stomp, boost pad) must
refresh the double jump, just like touching the ground does. These surfaces
don't produce a sustained `grounded` tick, so the air-jump counter — which
normally refills on a grounded->airborne transition — was never restored."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Action
from blueball.abilities import Ability
from blueball.input_feel import JumpController
from blueball.levels.chunks.base import TILE
from blueball.levels.chunks.spring import SpringChunk
from blueball.levels.chunks.patrol_platform import PatrolPlatform
from blueball import config


class StubAgent:
    def __init__(self, a=Action.IDLE):
        self._a = a
    def act(self, obs):
        return self._a


def _player_with_double_jump(spawn=(0.0, 0.0)):
    return Player(agent=StubAgent(), spawn_xy=spawn, abilities={Ability.DOUBLE_JUMP})


def test_jumpcontroller_refresh_restores_air_jumps():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    jc._air_jumps_remaining = 0
    jc.refresh_air_jumps()
    assert jc._air_jumps_remaining == 1


def test_jumpcontroller_refresh_noop_without_double_jump():
    jc = JumpController(abilities=set())
    jc._air_jumps_remaining = 0
    jc.refresh_air_jumps()
    assert jc._air_jumps_remaining == 0


def test_receive_spring_refreshes_air_jumps():
    p = _player_with_double_jump()
    p.jump_ctrl._air_jumps_remaining = 0
    p.receive_spring(500.0)
    assert p.jump_ctrl._air_jumps_remaining == 1


def test_receive_boost_refreshes_air_jumps():
    p = _player_with_double_jump()
    p.jump_ctrl._air_jumps_remaining = 0
    p.receive_boost(2.0)
    assert p.jump_ctrl._air_jumps_remaining == 1


def test_spring_bounce_leaves_double_jump_available():
    """Drop onto a spring with the double jump already spent; after the bounce
    the player should have their air jump back."""
    w = World(); register_collisions(w.space, world_ref=w)
    B = 400.0
    SpringChunk(width_tiles=3, impulse=600.0).build(w, x_offset=0, base_y=B)
    cx = 3 * TILE / 2
    p = Player(agent=StubAgent(), spawn_xy=(cx, B - config.BALL_RADIUS - 50),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    p.jump_ctrl._air_jumps_remaining = 0
    for _ in range(140):
        w.step(1 / 60)
    assert p.jump_ctrl._air_jumps_remaining == 1


def test_enemy_stomp_refreshes_air_jumps():
    """Stomping an enemy (which dies and never yields a grounded tick) should
    still restore the double jump."""
    w = World(); register_collisions(w.space, world_ref=w)
    B = 400.0
    PatrolPlatform(length_tiles=6, patroller_speed=0).build(w, x_offset=0, base_y=B)
    cx = 6 * TILE / 2
    p = Player(agent=StubAgent(), spawn_xy=(cx, B - config.BALL_RADIUS - 60),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    p.jump_ctrl._air_jumps_remaining = 0
    restored = 0
    for _ in range(160):
        w.step(1 / 60)
        restored = max(restored, p.jump_ctrl._air_jumps_remaining)
        if p.dead:
            break
    assert restored == 1
