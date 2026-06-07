"""Tests for the reverse spawn-curriculum subsystem (ai/curriculum.py)."""

from pathlib import Path

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


def test_train_curriculum_is_deterministic():
    from blueball.ai.curriculum import train_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    a = train_curriculum(level_path=path, pop_size=6, generations=3,
                         ga_seed=0, world_seed=1, max_steps=60)
    b = train_curriculum(level_path=path, pop_size=6, generations=3,
                         ga_seed=0, world_seed=1, max_steps=60)
    assert np.array_equal(a.best_genome, b.best_genome)
    assert [h["stage"] for h in a.history] == [h["stage"] for h in b.history]
    assert len(a.history) == 3
    for h in a.history:
        assert {"gen", "stage", "stage_label", "best", "mean",
                "best_reached_goal"} <= set(h)


def test_train_curriculum_advances_stage_when_elite_clears(monkeypatch):
    """Stub the evaluator so the elite always 'reaches goal' -> the stage index
    must climb each generation until it saturates at the last stage."""
    import blueball.ai.curriculum as curr
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    n_stages = len(curr.build_spawn_curriculum(path))

    def fake_eval(args):
        idx = args[0]
        return idx, float(idx), True   # elite (max idx) always reaches goal

    monkeypatch.setattr(curr, "evaluate_curriculum", fake_eval)
    res = curr.train_curriculum(level_path=path, pop_size=4,
                                generations=n_stages + 3, ga_seed=0, world_seed=1,
                                max_steps=10)
    # advances one stage per generation, then saturates at the last stage
    assert res.history[-1]["stage"] == n_stages - 1
    assert res.history[0]["stage"] == 0


def test_train_curriculum_holds_stage_when_never_clears(monkeypatch):
    import blueball.ai.curriculum as curr
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]

    def fake_eval(args):
        idx = args[0]
        return idx, float(idx), False  # nobody ever reaches goal

    monkeypatch.setattr(curr, "evaluate_curriculum", fake_eval)
    res = curr.train_curriculum(level_path=path, pop_size=4, generations=5,
                                ga_seed=0, world_seed=1, max_steps=10)
    assert all(h["stage"] == 0 for h in res.history)


def test_train_curriculum_writes_run_json(tmp_path):
    import json
    from blueball.ai.curriculum import train_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    run_dir = tmp_path / "mazecurr_run"
    train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                     world_seed=1, max_steps=60, save_dir=run_dir)
    assert (run_dir / "final_best.npy").exists()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["mode"] == "curriculum"
    cur = meta["curriculum"]
    assert isinstance(cur["stages"], list) and cur["stages"][-1] == "start"
    assert len(cur["trajectory"]) == len(cur["stages"])
    assert "reached_gen" in cur["trajectory"][0] and "cleared_gen" in cur["trajectory"][0]


def test_train_curriculum_marks_cracked_when_all_stages_clear(tmp_path, monkeypatch):
    """When the elite clears every stage (incl. the true-start stage), run.json
    records cracked=True and the final stage as 'start'."""
    import json
    import blueball.ai.curriculum as curr
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    n = len(curr.build_spawn_curriculum(path))

    def fake_eval(args):
        return args[0], float(args[0]), True  # elite always clears

    monkeypatch.setattr(curr, "evaluate_curriculum", fake_eval)
    run_dir = tmp_path / "crack_run"
    curr.train_curriculum(level_path=path, pop_size=4, generations=n + 2,
                          ga_seed=0, world_seed=1, max_steps=10, save_dir=run_dir)
    cur = json.loads((run_dir / "run.json").read_text())["curriculum"]
    assert cur["cracked"] is True
    assert cur["final_stage_index"] == n - 1
    assert cur["final_stage_label"] == "start"
    assert cur["trajectory"][-1]["cleared_gen"] is not None


def test_run_dir_name_curriculum_variant():
    from blueball.ai.persistence import run_dir_name
    assert run_dir_name(world_seed=1, timestamp="T", level_name="maze",
                        curriculum=True) == "mazecurr_w1_T"
    # existing variants still work
    assert run_dir_name(world_seed=1, timestamp="T", num_levels=5) == "lvls5_w1_T"
    assert run_dir_name(world_seed=1, timestamp="T",
                        level_name="maze") == "maze_w1_T"
    # no level_name -> 'level' fallback
    assert run_dir_name(world_seed=1, timestamp="T", curriculum=True) == "levelcurr_w1_T"


def test_train_maze_curriculum_cli_writes_run(tmp_path):
    import json, os, subprocess, sys
    import blueball
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_maze_curriculum.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "--level", "maze", "--pop", "4",
         "--gens", "2", "--max-steps", "60", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("mazecurr_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert meta["mode"] == "curriculum"
    assert meta["curriculum"]["stages"][-1] == "start"
    assert "reached_goal" in r.stdout


def test_train_maze_curriculum_cli_unknown_level_errors(tmp_path):
    import os, subprocess, sys
    import blueball
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "train_maze_curriculum.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "--level", "nope", "--pop", "2", "--gens", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env,
    )
    assert r.returncode != 0
    assert "Available" in (r.stderr + r.stdout)


def test_evaluate_curriculum_tracks_box_progress(monkeypatch):
    """On maze, spawning just left of the PushableBox and pushing it right makes
    box-push shaping raise fitness: evaluate_curriculum with BOX_PUSH_MULT=1.0
    exceeds the identical run scored with 0.0 by the (positive) box term."""
    import blueball.config as bbconfig
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    path = resolve_level_paths(["maze"])[0]
    # Just left of the maze box (x=3294, size 64 -> left face 3262); both maze
    # keys (ids 0,1) granted. Genome 20 rolls right and shoves the box.
    spawn_xy = (3250.0, 540.0)
    granted = (1 << 0) | (1 << 1)
    g = random_genome(np.random.default_rng(20))
    args = (0, g, 1, path, 600, spawn_xy, granted)

    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 0.0)
    _, fit_off, reached_off = evaluate_curriculum(args)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 1.0)
    _, fit_on, reached_on = evaluate_curriculum(args)

    assert fit_on > fit_off          # box moved right -> positive box term
    assert reached_off == reached_on # box term doesn't change the goal verdict


def test_evaluate_curriculum_no_box_unaffected_by_box_mult(monkeypatch):
    """On a box-less level, box_progress is 0.0 so BOX_PUSH_MULT has no effect:
    fitness is identical whether shaping is on or off."""
    import blueball.config as bbconfig
    from pathlib import Path
    import blueball
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.genome import random_genome
    level = Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"
    g = random_genome(np.random.default_rng(0))
    args = (0, g, 1, level, 200, (80.0, 540.0), 0)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 0.0)
    _, fit_off, _ = evaluate_curriculum(args)
    monkeypatch.setattr(bbconfig, "BOX_PUSH_MULT", 5.0)
    _, fit_on, _ = evaluate_curriculum(args)
    assert fit_off == fit_on


def test_build_box_lava_curriculum_single_stage_left_of_box():
    from blueball.ai.curriculum import (build_box_lava_curriculum,
                                        BOX_LAVA_SPAWN_MARGIN, granted_keys_before)
    path, world, meta = _maze_world()
    keys = _maze_keys(world)
    box = next(e for e in world.entities if type(e).__name__ == "PushableBox")
    box_x = float(box.body.position.x)
    all_bits = 0
    for kid, _ in keys:
        all_bits |= (1 << kid)

    stages = build_box_lava_curriculum(path)
    assert len(stages) == 1
    s = stages[0]
    assert s.label == "box_lava"
    assert s.spawn_xy[0] == pytest.approx(box_x - box.size / 2.0 - BOX_LAVA_SPAWN_MARGIN)
    assert s.spawn_xy[0] < box_x
    assert s.spawn_xy[1] == pytest.approx(float(meta.spawn[1]))
    assert s.granted_keys == all_bits
    assert s.granted_keys == granted_keys_before(keys, s.spawn_xy[0])


def test_build_box_lava_curriculum_spawn_is_frame1_safe():
    """The box-lava spawn lands on the approach ledge, not in lava/geometry."""
    from blueball.ai.curriculum import (build_box_lava_curriculum,
                                        make_curriculum_player)
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    stage = build_box_lava_curriculum(path)[0]
    world = World(seed=1)
    register_collisions(world.space, world_ref=world)
    load_level(path, world)
    p = make_curriculum_player(world, g, stage.spawn_xy, stage.granted_keys)
    world.substep()
    assert not p.dead


def test_build_box_lava_curriculum_requires_box():
    from pathlib import Path
    import blueball
    from blueball.ai.curriculum import build_box_lava_curriculum
    level = Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"
    with pytest.raises(ValueError, match="PushableBox"):
        build_box_lava_curriculum(level)


def test_train_curriculum_custom_stages_used(tmp_path):
    """Passing stages=build_box_lava_curriculum(maze) trains at that single fixed
    stage: run.json records exactly ['box_lava'] and the loop never recedes."""
    import json
    from blueball.ai.curriculum import train_curriculum, build_box_lava_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    stages = build_box_lava_curriculum(path)
    run_dir = tmp_path / "boxlava_run"
    train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                     world_seed=1, max_steps=60, save_dir=run_dir, stages=stages)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["curriculum"]["stages"] == ["box_lava"]
    assert all(h["stage"] == 0 for h in meta["history"])


def test_train_curriculum_custom_stages_deterministic():
    from blueball.ai.curriculum import train_curriculum, build_box_lava_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    stages = build_box_lava_curriculum(path)
    a = train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                         world_seed=1, max_steps=60, stages=stages)
    b = train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                         world_seed=1, max_steps=60, stages=stages)
    assert np.array_equal(a.best_genome, b.best_genome)


def test_train_curriculum_default_stages_unchanged(tmp_path):
    """stages=None (default) still builds the full reverse curriculum:
    run.json's last stage is 'start' as before."""
    import json
    from blueball.ai.curriculum import train_curriculum
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    run_dir = tmp_path / "default_run"
    train_curriculum(level_path=path, pop_size=4, generations=2, ga_seed=0,
                     world_seed=1, max_steps=60, save_dir=run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["curriculum"]["stages"][-1] == "start"
