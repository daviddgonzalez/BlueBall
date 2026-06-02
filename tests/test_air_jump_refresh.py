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


def test_jumpcontroller_refills_on_leaving_steep_surface():
    """Leaving a too-steep-to-be-'grounded' surface (e.g. a bump side) still
    refreshes the air jump via the lenient on_surface signal."""
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # On a steep slope: not grounded, but in contact (on_surface).
    jc.tick(Action.IDLE, grounded=False, dt=1 / 120, on_surface=True)
    jc._air_jumps_remaining = 0  # spent the double jump
    # Now airborne (left the slope): should restock.
    jc.tick(Action.IDLE, grounded=False, dt=1 / 120, on_surface=False)
    assert jc._air_jumps_remaining == 1


def test_jumpcontroller_no_refill_while_purely_airborne():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    jc.tick(Action.IDLE, grounded=False, dt=1 / 120, on_surface=False)
    jc._air_jumps_remaining = 0
    jc.tick(Action.IDLE, grounded=False, dt=1 / 120, on_surface=False)
    assert jc._air_jumps_remaining == 0  # never touched anything


def test_steep_bump_refreshes_air_jump():
    """Rolling over a steep (uncapped) bump while the double jump is spent must
    give it back when the ball leaves the bump."""
    from blueball.levels.chunks.bump import Bump
    from blueball.levels.chunks.flat import Flat
    w = World(); register_collisions(w.space, world_ref=w)
    B = 400.0
    Flat(width_tiles=4).build(w, x_offset=0, base_y=B)
    Bump(height=48, width_tiles=2, rounded=True).build(w, x_offset=4 * TILE, base_y=B)
    Flat(width_tiles=8).build(w, x_offset=6 * TILE, base_y=B)
    p = Player(agent=StubAgent(), spawn_xy=(20, B - config.BALL_RADIUS - 2),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    p.body.velocity = (500, 0)
    p.jump_ctrl._air_jumps_remaining = 0
    refilled = False
    for _ in range(160):
        w.step(1 / 60)
        if p.jump_ctrl._air_jumps_remaining == 1:
            refilled = True
        if p.dead:
            break
    assert refilled


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
