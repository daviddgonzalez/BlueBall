"""Double-jump traversal gym segments — the acquisition counterpart to the
Infinite Run double-jump chunks.

The box segments teach the double jump only as a box-step; these isolate it as a
*traversal* skill (hop up a ledge / vault a wide gap) so clearing the segment —
and banking its completion bonus — requires the second jump. Each is verified
solvable by the project's strongest double-jump maneuver and unsolvable without
the ability, the same anti-cheese contract the box segments hold.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blueball.abilities import Ability
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import (
    DoubleHopSegment,
    DoubleWallSegment,
    DoubleVaultSegment,
    SEGMENT_TEMPLATES,
    SegmentSampler,
)
from tests.segment_maneuvers import DoubleJumpVaultAgent, fresh_world, run_segment

DJ = frozenset({Ability.DOUBLE_JUMP})
_LAUNCH_SWEEP = range(160, 280, 8)


def _names(world):
    return [type(e).__name__ for e in world.entities]


def _solved_launch_xs(segment, abilities):
    """Launch positions at which the max double-jump maneuver reaches the goal."""
    solved = []
    for lx in _LAUNCH_SWEEP:
        w = fresh_world()
        segment.build(w, x_offset=0.0)
        p = Player(agent=None, spawn_xy=(40.0, GROUND_Y - 30.0), abilities=set(abilities))
        w.add_entity(p)
        agent = DoubleJumpVaultAgent(launch_x=lx)
        agent.player = p
        p.agent = agent
        if run_segment(w, p) == "GOAL":
            solved.append(lx)
    return solved


# --------------------------------------------------------------------------- #
# registration + ability metadata
# --------------------------------------------------------------------------- #
def test_double_jump_segments_registered():
    assert DoubleHopSegment in SEGMENT_TEMPLATES
    assert DoubleWallSegment in SEGMENT_TEMPLATES
    assert DoubleVaultSegment in SEGMENT_TEMPLATES


def test_double_jump_segments_require_double_jump():
    assert DoubleHopSegment.min_abilities == DJ
    assert DoubleWallSegment.min_abilities == DJ
    assert DoubleVaultSegment.min_abilities == DJ
    # Gentle rungs lower-tier than the demanding vault (the curriculum ramp).
    assert DoubleHopSegment.tier < DoubleVaultSegment.tier
    assert DoubleWallSegment.tier < DoubleVaultSegment.tier


# --------------------------------------------------------------------------- #
# build + composition
# --------------------------------------------------------------------------- #
def test_double_hop_builds_a_goal_with_positive_width():
    w = fresh_world()
    width = DoubleHopSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)


def test_double_wall_builds_a_goal_with_positive_width():
    w = fresh_world()
    width = DoubleWallSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)


def test_double_vault_builds_a_goal_with_positive_width():
    w = fresh_world()
    width = DoubleVaultSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)


# --------------------------------------------------------------------------- #
# solvable WITH double jump, NOT without (the gym's anti-cheese contract)
# --------------------------------------------------------------------------- #
def test_double_hop_solvable_with_double_jump_only():
    assert len(_solved_launch_xs(DoubleHopSegment(), DJ)) >= 2
    assert _solved_launch_xs(DoubleHopSegment(), frozenset()) == []


def test_double_wall_solvable_with_double_jump_only():
    assert len(_solved_launch_xs(DoubleWallSegment(), DJ)) >= 2
    assert _solved_launch_xs(DoubleWallSegment(), frozenset()) == []


def test_double_vault_solvable_with_double_jump_only():
    assert len(_solved_launch_xs(DoubleVaultSegment(), DJ)) >= 2
    assert _solved_launch_xs(DoubleVaultSegment(), frozenset()) == []


# --------------------------------------------------------------------------- #
# sampler gating by granted abilities
# --------------------------------------------------------------------------- #
def test_sampler_excludes_double_jump_segments_without_ability():
    pool = SegmentSampler(seed=1, granted_abilities=frozenset())._pool
    assert DoubleHopSegment not in pool
    assert DoubleWallSegment not in pool
    assert DoubleVaultSegment not in pool


def test_sampler_includes_double_jump_segments_with_ability():
    pool = SegmentSampler(seed=1, granted_abilities=DJ)._pool
    assert DoubleHopSegment in pool
    assert DoubleWallSegment in pool
    assert DoubleVaultSegment in pool
