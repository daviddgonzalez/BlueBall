"""Tests for rounded terrain (smoothstep curves) and Infinite Run rules."""

import pymunk

from blueball.levels.chunks._curve import smoothstep_ramp, _smoothstep
from blueball.levels.chunks.stairs import StairsUp, StairsDown
from blueball.levels.chunks.bump import Bump
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.sampler import ChunkSampler
from blueball.world import World


def _segments(world):
    return [s for s in world.space.shapes if isinstance(s, pymunk.Segment)]


def test_smoothstep_endpoints_and_flat_tangents():
    assert _smoothstep(0.0) == 0.0
    assert _smoothstep(1.0) == 1.0
    # Near the ends the curve is almost flat (slope -> 0); near the middle it
    # is steep. Compare finite-difference slopes.
    eps = 1e-4
    slope_start = (_smoothstep(eps) - _smoothstep(0)) / eps
    slope_mid = (_smoothstep(0.5 + eps) - _smoothstep(0.5)) / eps
    assert slope_start < 0.05
    assert slope_mid > 1.0


def test_smoothstep_ramp_spans_endpoints_no_vertical_segments():
    world = World()
    smoothstep_ramp(world, 0, GROUND_Y, 96, GROUND_Y - 100)
    segs = _segments(world)
    assert len(segs) >= 2
    # Every segment advances in x (no vertical riser walls).
    for s in segs:
        assert s.a.x != s.b.x
    xs = sorted([s.a.x for s in segs] + [s.b.x for s in segs])
    assert min(xs) == 0 and max(xs) == 96


def test_stairs_up_rounded_is_a_ramp_not_steps():
    from blueball.levels.chunks.stairs import _MAX_RAMP_SLOPE
    world = World()
    width = StairsUp(steps=3, step_height=32, rounded=True).build(world, x_offset=0)
    rise = 3 * 32
    # Rounded ramps are widened so their average slope stays gentle (this is
    # what keeps the ball groundable in valleys — see the valley regression).
    assert width >= rise / _MAX_RAMP_SLOPE
    segs = _segments(world)
    # No vertical risers, and the ramp climbs from ground to ground-rise.
    assert all(s.a.x != s.b.x for s in segs)
    ys = [s.a.y for s in segs] + [s.b.y for s in segs]
    assert max(ys) == GROUND_Y
    assert min(ys) == GROUND_Y - 3 * 32


def test_stairs_down_rounded_ramps_back_to_ground():
    world = World()
    StairsDown(steps=3, step_height=32, rounded=True).build(world, x_offset=0)
    segs = _segments(world)
    assert all(s.a.x != s.b.x for s in segs)
    ys = [s.a.y for s in segs] + [s.b.y for s in segs]
    assert max(ys) == GROUND_Y
    assert min(ys) == GROUND_Y - 3 * 32


def test_bump_rounded_has_no_sharp_apex():
    world = World()
    Bump(height=40, width_tiles=2, rounded=True).build(world, x_offset=0)
    segs = _segments(world)
    assert all(s.a.x != s.b.x for s in segs)
    ys = [s.a.y for s in segs] + [s.b.y for s in segs]
    assert max(ys) == GROUND_Y           # starts/ends on the ground
    assert min(ys) == GROUND_Y - 40      # reaches the crest height
    # Multiple segments per side => not a single triangular apex.
    assert len(segs) >= 4


def test_stairs_default_is_blocky_with_risers():
    """Hand-authored levels (rounded defaults False) keep vertical risers."""
    world = World()
    StairsUp(steps=3, step_height=32).build(world, x_offset=0)
    segs = _segments(world)
    vertical = [s for s in segs if s.a.x == s.b.x]
    assert vertical, "blocky stairs must keep vertical riser segments"


def test_bump_default_is_sharp_triangle():
    world = World()
    Bump(height=40, width_tiles=2).build(world, x_offset=0)
    segs = _segments(world)
    # Classic bump is exactly two segments meeting at a sharp apex.
    assert len(segs) == 2


def test_sampler_rounds_stairs_and_bump():
    """The sampler turns rounding on, so Infinite Run terrain is curved."""
    for cls in (StairsUp, StairsDown, Bump):
        import random
        params = cls.random_params(random.Random(0))
        assert params.get("rounded") is True


def test_stairs_ground_offsets():
    """Stairs report how they shift the running ground height; flat chunks 0."""
    up = StairsUp(steps=3, step_height=40)
    assert up.entry_dy == 0.0 and up.exit_dy == -120  # climbs 120px (up = -y)
    down = StairsDown(steps=3, step_height=40)
    assert down.entry_dy == -120 and down.exit_dy == 0.0
    from blueball.levels.chunks.flat import Flat
    f = Flat()
    assert f.entry_dy == 0.0 and f.exit_dy == 0.0


def test_chunk_build_accepts_base_y_and_shifts_ground():
    """A chunk built at a raised base_y places its ground at that height."""
    from blueball.levels.chunks.flat import Flat
    world = World()
    Flat(width_tiles=3).build(world, x_offset=0, base_y=GROUND_Y - 100)
    seg = _segments(world)[0]
    assert seg.a.y == GROUND_Y - 100 and seg.b.y == GROUND_Y - 100


def test_rounded_ramp_stays_within_grounded_tolerance():
    """Regression: the steepest segment of a rounded ramp must stay within the
    player's grounded tolerance, so the ball is grounded (and can recharge /
    perform its jump) everywhere on the ramp — not just the gentle parts. The
    bug was steep ramp mid-sections (~37deg) reading as not-grounded."""
    import math
    from blueball import config
    from blueball.levels.chunks.stairs import StairsUp, StairsDown

    for cls in (StairsUp, StairsDown):
        for steps, step_height in [(2, 24), (3, 32), (4, 40)]:
            world = World()
            cls(steps=steps, step_height=step_height, rounded=True).build(world, x_offset=0)
            segs = _segments(world)
            max_angle = max(
                math.degrees(math.atan(abs((s.b.y - s.a.y) / (s.b.x - s.a.x))))
                for s in segs if s.b.x != s.a.x
            )
            assert max_angle <= config.GROUNDED_NORMAL_TOLERANCE_DEG, (
                f"{cls.__name__} steps={steps} sh={step_height}: {max_angle:.1f}deg too steep")


def test_rounded_stairs_valley_is_escapable():
    """Regression: a rounded stairs_down -> stairs_up valley must leave the ball
    grounded at the bottom, so it can jump out instead of softlocking. The bug
    was steep ramps wedging the ball on surfaces past the grounded tolerance."""
    from blueball.collision import register as register_collisions
    from blueball.entities.player import Player
    from blueball.agent import Action
    from blueball import config

    class _Stub:
        def __init__(self, a): self._a = a
        def act(self, obs): return self._a

    for step_height in (24, 32, 40):
        world = World()
        register_collisions(world.space, world_ref=world)
        base = 300.0
        wd = StairsDown(steps=3, step_height=step_height, rounded=True).build(
            world, x_offset=0, base_y=base)
        StairsUp(steps=3, step_height=step_height, rounded=True).build(
            world, x_offset=wd, base_y=base)
        player = Player(agent=_Stub(Action.IDLE),
                        spawn_xy=(wd, base - config.BALL_RADIUS - 2))
        world.add_entity(player)
        for _ in range(180):  # settle ~3s
            world.step(1 / 60)
        assert player.grounded, f"softlock: not grounded in valley (step_height={step_height})"


def test_sampler_emits_no_checkpoints_when_disabled():
    seq = list(ChunkSampler(seed=2, target_chunks=120, emit_checkpoints=False))
    assert not any(s["type"] == "checkpoint" for s in seq)


def test_sampler_still_emits_checkpoints_by_default():
    seq = list(ChunkSampler(seed=2, target_chunks=120, checkpoint_every=20))
    assert any(s["type"] == "checkpoint" for s in seq)
