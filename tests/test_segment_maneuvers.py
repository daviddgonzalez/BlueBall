import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pymunk
from blueball.abilities import Ability
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.world import World
from tests.segment_maneuvers import (
    DoubleJumpVaultAgent,
    SingleStepAgent,
    DoubleStepAgent,
    fresh_world,
    find_entity,
    run_segment,
)


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


# ---------------------------------------------------------------------------
# Box-step solver agents (Task 2a)
# ---------------------------------------------------------------------------

def _build_step_world(pit_tiles, depth, box_size, box_frac):
    """Build Flat(2) | Flat(3) | BoxLavaGap(..., box_frac=...) | GoalChunk(2)."""
    from blueball.levels.chunks.flat import Flat
    from blueball.levels.chunks.goal import GoalChunk
    from blueball.levels.chunks.box_lava_gap import BoxLavaGap
    from blueball.levels.chunks.base import TILE

    w = fresh_world()
    x = 0.0
    x += Flat(width_tiles=2).build(w, x_offset=x)
    x += Flat(width_tiles=3).build(w, x_offset=x)
    x += BoxLavaGap(pit_tiles=pit_tiles, depth=depth, box_size=box_size,
                    box_frac=box_frac).build(w, x_offset=x)
    x += GoalChunk(width_tiles=2).build(w, x_offset=x)
    return w


def test_single_step_agent_solves_preplaced_box():
    """SingleStepAgent must reach the goal on a stage-1 pre-placed box pit."""
    sweep = [
        (lx, obr)
        for lx in (232, 238, 245)
        for obr in (3, 6)
    ]
    solved = []
    for launch_x, on_box_run in sweep:
        w = _build_step_world(pit_tiles=24, depth=72, box_size=64, box_frac=0.5)
        p = Player(
            agent=None,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        w.add_entity(p)
        agent = SingleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
        agent.player = p
        agent.box = find_entity(w, "PushableBox")
        p.agent = agent
        result = run_segment(w, p)
        if result == "GOAL":
            solved.append((launch_x, on_box_run))
    assert solved, (
        f"No SingleStepAgent config reached GOAL; tried {sweep}"
    )


def test_double_step_agent_solves_bigger_preplaced_box():
    """DoubleStepAgent mounts+clears the SHIPPED stage-2 pre-placed box.

    Builds the cell straight from BoxLeapSegment so the test always tracks the
    shipped geometry (the robust cell pit=40/depth=96/box=96/frac=0.55), never a
    hand-typed marginal one. The shipped cell solves 24/24 in the probe, so a
    small sweep of (launch_x, on_box_run) lands several GOALs comfortably — no
    brittle wide-sweep fallback needed. We assert at least two distinct configs
    reach GOAL to keep the margin honest (not a knife-edge)."""
    from blueball.levels.segments import BoxLeapSegment

    # A handful of combos drawn from the probe's known-good grid
    # (_SOLVE_LAUNCH_X x _SOLVE_ON_BOX_RUN), all of which solve the shipped cell.
    sweep = [
        (lx, obr)
        for lx in (232, 244, 256)
        for obr in (4, 8)
    ]
    solved = []
    for launch_x, on_box_run in sweep:
        w = fresh_world()
        BoxLeapSegment().build(w, x_offset=0.0)
        p = Player(
            agent=None,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        w.add_entity(p)
        agent = DoubleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
        agent.player = p
        agent.box = find_entity(w, "PushableBox")
        p.agent = agent
        result = run_segment(w, p)
        if result == "GOAL":
            solved.append((launch_x, on_box_run))

    assert len(solved) >= 2, (
        f"Expected >=2 DoubleStepAgent configs to reach GOAL on the shipped "
        f"BoxLeap cell; only {solved} did (tried {sweep})"
    )
