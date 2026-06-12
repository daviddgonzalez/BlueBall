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


def test_lava_rising_easiest_stage_does_not_plunge():
    path = resolve_level_paths(["lava_rising"])[0]
    stages = build_spawn_curriculum(path)
    s = stages[0]  # easiest (near_goal) — was a start-y void spawn before the fix
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    pl = make_curriculum_player(w, random_genome(np.random.default_rng(0)),
                                s.spawn_xy, s.granted_keys, meta.starting_abilities)
    for _ in range(90):
        w.substep()
        if pl.dead:
            break
    assert not pl.dead, "easiest stage died within 90 steps (plunged into lava)"
    assert abs(pl.body.position.y - s.spawn_xy[1]) < 250, "ball fell far from spawn-y"
