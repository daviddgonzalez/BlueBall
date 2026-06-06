"""Tests for the reverse spawn-curriculum subsystem (ai/curriculum.py)."""

import numpy as np
import pytest


def _maze_world():
    """Load maze into a fresh world; return (path, world, meta)."""
    from blueball.ai.episodes import resolve_level_paths
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = resolve_level_paths(["maze"])[0]
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(path, world)
    return path, world, meta


def _maze_keys(world):
    """[(key_id, x)] for the maze Key entities, sorted by x."""
    keys = [(int(e.key_id), float(e.position[0]))
            for e in world.entities if type(e).__name__ == "Key"]
    return sorted(keys, key=lambda k: k[1])


def test_granted_keys_before_ors_keys_behind_spawn():
    from blueball.ai.curriculum import granted_keys_before
    keys = [(0, 1000.0), (3, 2000.0)]
    assert granted_keys_before(keys, 1500.0) == (1 << 0)
    assert granted_keys_before(keys, 2500.0) == (1 << 0) | (1 << 3)
    assert granted_keys_before(keys, 500.0) == 0
    assert granted_keys_before(keys, 1000.0) == 0  # strict <, not <=


def test_build_spawn_curriculum_maze_orders_and_grants():
    from blueball.ai.curriculum import build_spawn_curriculum, SPAWN_MARGIN, granted_keys_before
    path, world, meta = _maze_world()
    keys = _maze_keys(world)
    all_bits = 0
    for kid, _ in keys:
        all_bits |= (1 << kid)

    stages = build_spawn_curriculum(path)

    # one near_goal + one per key + one start
    assert len(stages) == len(keys) + 2
    xs = [s.spawn_xy[0] for s in stages]
    assert xs == sorted(xs, reverse=True)          # strictly receding
    assert len(xs) == len(set(xs))                 # distinct

    assert stages[0].label == "near_goal"
    assert stages[0].granted_keys == all_bits      # all keys behind near_goal

    assert stages[-1].label == "start"
    assert stages[-1].granted_keys == 0
    assert stages[-1].spawn_xy == (float(meta.spawn[0]), float(meta.spawn[1]))

    # every stage's grant matches the "keys strictly behind spawn" rule,
    # and middle stages spawn SPAWN_MARGIN before a key
    for s in stages:
        assert s.granted_keys == granted_keys_before(keys, s.spawn_xy[0])
        if s.label.startswith("before_key"):
            kid = int(s.label[len("before_key"):])
            kx = dict(keys)[kid]
            assert s.spawn_xy[0] == pytest.approx(kx - SPAWN_MARGIN)


def test_make_curriculum_player_sets_spawn_and_keys():
    from blueball.ai.curriculum import make_curriculum_player
    from blueball.ai.genome import random_genome
    _, world, _ = _maze_world()
    g = random_genome(np.random.default_rng(0))
    mask = (1 << 0) | (1 << 3)
    p = make_curriculum_player(world, g, (1200.0, 540.0), mask)
    assert (float(p.body.position.x), float(p.body.position.y)) == (1200.0, 540.0)
    assert p.keys_held == mask
    assert p in world.entities


def test_evaluate_curriculum_returns_idx_fitness_reached():
    from blueball.ai.curriculum import build_spawn_curriculum, evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    start = build_spawn_curriculum(path)[-1]  # the "start" stage
    idx, fit, reached = evaluate_curriculum(
        (7, g, 1, path, 120, start.spawn_xy, start.granted_keys))
    assert idx == 7
    assert isinstance(fit, float) and np.isfinite(fit)
    assert isinstance(reached, bool)


def test_curriculum_stage_spawns_are_frame1_safe():
    """No maze stage spawns the agent into geometry/over a pit (not dead after
    one substep)."""
    from blueball.ai.curriculum import build_spawn_curriculum, make_curriculum_player
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    for stage in build_spawn_curriculum(path):
        world = World(seed=1)
        register_collisions(world.space, world_ref=world)
        load_level(path, world)
        p = make_curriculum_player(world, g, stage.spawn_xy, stage.granted_keys)
        world.substep()
        assert not p.dead, f"agent dead on frame 1 at stage {stage.label}"


def test_evaluate_curriculum_granted_keys_dont_inflate_fitness():
    """Granted keys are scaffolding, not achievements: at the same spawn, a huge
    granted mask must not change fitness vs no grant (the agent collects the same
    real keys during the episode either way)."""
    from blueball.ai.curriculum import build_spawn_curriculum, evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    start = build_spawn_curriculum(path)[-1]  # true start; real keys are far ahead
    _, fit_no_grant, _ = evaluate_curriculum((0, g, 1, path, 120, start.spawn_xy, 0))
    _, fit_granted, _ = evaluate_curriculum((0, g, 1, path, 120, start.spawn_xy, 0xFF))
    assert fit_no_grant == pytest.approx(fit_granted)
