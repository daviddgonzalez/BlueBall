import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pymunk
from blueball.abilities import Ability
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.world import World
from tests.segment_maneuvers import DoubleJumpVaultAgent


def _ledge_gap_world(gap_px):
    """Two ledges at GROUND_Y with a lethal fall between near edge 256 and far."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    def seg(a, b):
        s = pymunk.Segment(w.space.static_body, a, b, 5)
        s.friction = 1.0
        w.space.add(s)
    seg((0, GROUND_Y), (256, GROUND_Y))                       # near ledge
    # Far ledge is wide (1200px) so a genuine MAX-DISTANCE double jump lands ON
    # it (the apex-fired arc clears ~700px and touches down at x~959); a short
    # far ledge would end before that landing and give a false miss.
    seg((256 + gap_px, GROUND_Y), (256 + gap_px + 1200, GROUND_Y))  # far ledge
    return w


def _clears_gap(gap_px, abilities):
    """Run DoubleJumpVaultAgent over a gap and report whether it lands grounded
    on the far ledge (True) or falls into the pit / never makes it (False)."""
    w = _ledge_gap_world(gap_px)
    agent = DoubleJumpVaultAgent(launch_x=250)
    p = Player(agent=agent, spawn_xy=(40.0, GROUND_Y - 30.0), abilities=abilities)
    agent.player = p
    w.add_entity(p)
    for _ in range(800):  # max-distance arc lands ~step 400; budget gives margin
        w.substep()
        if p.body.position[0] > 256 + gap_px + 10 and p.grounded:
            return True
        if p.dead:
            return False
    return False


def test_double_jump_clears_a_gap_a_single_jump_cannot():
    # 550px sits between single-jump reach (~411px) and double-jump reach
    # (~703px): a competent double jump clears it, but the SAME maneuver without
    # DOUBLE_JUMP (the second press is a no-op, so it's a single jump) falls into
    # the pit. The pairing is what proves the maneuver is a genuine two-impulse
    # arc and not a single jump in disguise.
    assert _clears_gap(550, {Ability.DOUBLE_JUMP})
    assert not _clears_gap(550, set())


def test_double_jump_falls_into_an_unreachable_gap():
    # 1200px is far beyond even the max-distance double-jump reach (~703px).
    assert not _clears_gap(1200, {Ability.DOUBLE_JUMP})
