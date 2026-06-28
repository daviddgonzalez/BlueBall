"""Authored curriculum waypoints land on real platforms (not the void), and the
easiest lava_rising stage no longer plunges into the rising lava."""
import numpy as np
import pymunk

from blueball.ai.curriculum import build_spawn_curriculum, evaluate_curriculum, make_curriculum_player
from blueball.ai.episodes import resolve_level_paths
from blueball.ai.genome import random_genome
from blueball.collision import register as register_collisions
from blueball.levels.loader import load_level
from blueball.world import World


def _ground_below(w, x, y, maxd=120):
    seg = w.space.segment_query_first((x, y), (x, y + maxd), 5, pymunk.ShapeFilter())
    return (seg.point.y - y) if seg else None


def _assert_waypoints_on_ground(level):
    path = resolve_level_paths([level])[0]
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    assert meta.curriculum_spawns, f"{level} declares no curriculum_spawns"
    for wp in meta.curriculum_spawns:
        d = _ground_below(w, float(wp["x"]), float(wp["y"]))
        assert d is not None and d >= 0, f"{level} spawn {wp} is over a void"


def test_lava_rising_waypoints_land_on_ground():
    _assert_waypoints_on_ground("lava_rising")


def test_vertical_climb_waypoints_land_on_ground():
    _assert_waypoints_on_ground("vertical_climb")


def test_lava_rising_forward_easiest_stage_is_start_safe():
    """lava_rising now trains FORWARD (start_gated): difficulty is the rising
    lava + key/door gating along the way, so every stage spawns at the true
    start and a finish-line x advances goal-ward (the doors physically gate the
    goal, forcing key collection). The easiest stage must spawn on the solid
    start footing and be frame-1 safe (no void/lava plunge)."""
    path = resolve_level_paths(["lava_rising"])[0]
    stages = build_spawn_curriculum(path)
    s = stages[0]
    assert s.checkpoint_x is not None, "forward curriculum: stage 0 has an x finish line"
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    assert s.spawn_xy == (float(meta.spawn[0]), float(meta.spawn[1])), "spawns at true start"
    pl = make_curriculum_player(w, random_genome(np.random.default_rng(0)),
                                s.spawn_xy, s.granted_keys, meta.starting_abilities)
    w.substep()
    assert not pl.dead, "start stage died on frame 1"
