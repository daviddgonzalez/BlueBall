"""Tests for multi-episode training: aggregation, normalization, episode
construction, and the multi-episode trainer path."""

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pytest

import blueball


def test_aggregate_single_score_is_identity():
    # One episode -> population std is 0 -> returns the score exactly. This is
    # what makes single-episode training numerically identical to the old path.
    from blueball.ai.episodes import aggregate_fitness
    assert aggregate_fitness([42.5], lam=1.0) == 42.5


def test_aggregate_matches_mean_minus_lambda_std():
    from blueball.ai.episodes import aggregate_fitness
    # mean([10,20]) = 15; population std = 5; 15 - 0.5*5 = 12.5
    assert aggregate_fitness([10.0, 20.0], lam=0.5) == pytest.approx(12.5)


def test_aggregate_empty_raises():
    from blueball.ai.episodes import aggregate_fitness
    with pytest.raises(ValueError):
        aggregate_fitness([], lam=1.0)


def test_aggregate_min_mode_returns_minimum():
    from blueball.ai.episodes import aggregate_fitness
    # lam is ignored in min mode; the worst level is the floor.
    assert aggregate_fitness([0.2, 1.0, 0.5], lam=1.0, mode="min") == 0.2


def test_aggregate_min_single_score_is_identity():
    from blueball.ai.episodes import aggregate_fitness
    assert aggregate_fitness([0.7], lam=1.0, mode="min") == 0.7


def test_aggregate_min_empty_raises():
    from blueball.ai.episodes import aggregate_fitness
    with pytest.raises(ValueError):
        aggregate_fitness([], lam=1.0, mode="min")


def test_episodespec_is_frozen():
    from blueball.ai.episodes import EpisodeSpec
    ep = EpisodeSpec(kind="infinite", seed=1234, level_path=None,
                     world_seed=1, max_steps=100)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.seed = 5  # frozen dataclass rejects attribute assignment


def test_episodespec_is_picklable():
    from blueball.ai.episodes import EpisodeSpec
    ep = EpisodeSpec(kind="static", seed=0, level_path="x.json",
                     world_seed=1, max_steps=100, norm=123.0)
    assert pickle.loads(pickle.dumps(ep)) == ep


def test_config_has_std_penalty_default():
    from blueball import config
    assert config.GA_FITNESS_STD_PENALTY == 1.0


def _levels_dir() -> Path:
    return Path(blueball.__file__).parent / "levels"


def _load_meta(level):
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    return load_level(level, world)


def test_level_par_tutorial_hill_is_width_plus_goal():
    from blueball import config
    from blueball.ai.episodes import compute_level_par
    path = _levels_dir() / "tutorial_hill.json"
    # tutorial_hill has a goal, no keys, no collectibles ->
    # par = width * (1 + GOAL_MULT)
    meta = _load_meta(path)
    expected = meta.total_width * (1.0 + config.GOAL_MULT)
    assert compute_level_par(path) == pytest.approx(expected)


def test_level_par_flat_only_has_no_bonus():
    from blueball.ai.episodes import compute_level_par
    level = {
        "name": "Flat", "background": "#000000", "ground": "#000000",
        "spawn": [80, 540],
        "chunks": [{"type": "flat", "width_tiles": 5}],
    }
    par = compute_level_par(level)
    meta = _load_meta(level)
    assert par > 0.0
    assert par == pytest.approx(meta.total_width)  # no goal/keys/collectibles


def test_level_par_empty_returns_one():
    from blueball.ai.episodes import compute_level_par
    level = {
        "name": "Empty", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "chunks": [],
    }
    assert compute_level_par(level) == 1.0


def test_generate_seeds_single():
    from blueball.ai.episodes import generate_seeds
    assert generate_seeds(1234, 1) == [1234]


def test_generate_seeds_distinct_deterministic_and_includes_base():
    from blueball.ai.episodes import generate_seeds
    a = generate_seeds(1234, 4)
    b = generate_seeds(1234, 4)
    assert a == b                # deterministic
    assert a[0] == 1234          # base seed first
    assert len(set(a)) == 4      # distinct


def test_infinite_episodes_build():
    from blueball.ai.episodes import infinite_episodes
    eps = infinite_episodes([1, 2], world_seed=1, max_steps=100)
    assert [e.kind for e in eps] == ["infinite", "infinite"]
    assert [e.seed for e in eps] == [1, 2]
    assert all(e.norm == pytest.approx(1.0) and e.level_path is None for e in eps)


def test_resolve_level_paths_unknown_raises():
    from blueball.ai.episodes import resolve_level_paths
    with pytest.raises(ValueError) as exc:
        resolve_level_paths(["does_not_exist"])
    assert "Available" in str(exc.value)


def test_resolve_and_static_episodes_tutorial_hill():
    from blueball.ai.episodes import (compute_level_par, resolve_level_paths,
                                      static_episodes)
    paths = resolve_level_paths(["tutorial_hill"])
    assert paths[0].endswith("tutorial_hill.json")
    eps = static_episodes(paths, world_seed=1, max_steps=100)
    assert eps[0].kind == "static"
    assert eps[0].norm == pytest.approx(compute_level_par(paths[0]))


def test_evaluate_episodes_single_equals_raw_evaluate_infinite():
    from blueball.ai.episodes import EpisodeSpec
    from blueball.ai.genome import random_genome
    from blueball.ai.trainer import evaluate_episodes, evaluate_infinite
    g = random_genome(np.random.default_rng(0))
    _, raw = evaluate_infinite((0, g, 1234, 1, 120))
    ep = EpisodeSpec(kind="infinite", seed=1234, level_path=None,
                     world_seed=1, max_steps=120)
    _, agg = evaluate_episodes((0, g, (ep,), 1.0, "mean_std"))
    assert agg == pytest.approx(raw)


def test_evaluate_episodes_empty_raises():
    from blueball.ai.trainer import evaluate_episodes
    g = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError):
        evaluate_episodes((0, g, (), 1.0, "mean_std"))


def test_evaluate_episodes_min_mode_returns_min_of_episodes():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.genome import random_genome
    from blueball.ai.trainer import evaluate_episodes, evaluate_infinite
    g = random_genome(np.random.default_rng(0))
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=120)
    # infinite episodes have norm=1.0, so normalized score == raw fitness
    _, r0 = evaluate_infinite((0, g, 1234, 1, 120))
    _, r1 = evaluate_infinite((0, g, 777, 1, 120))
    _, agg = evaluate_episodes((0, g, tuple(eps), 1.0, "min"))
    assert agg == pytest.approx(min(r0, r1))


def test_train_aggregate_min_is_deterministic_and_finite():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=80)
    a = train(pop_size=6, generations=3, episodes=eps, ga_seed=0, aggregate="min")
    b = train(pop_size=6, generations=3, episodes=eps, ga_seed=0, aggregate="min")
    assert np.array_equal(a.best_genome, b.best_genome)
    assert len(a.history) == 3
    for h in a.history:
        assert np.isfinite(h["best"]) and np.isfinite(h["min"])


def test_train_multi_episode_is_deterministic():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=120)
    a = train(pop_size=6, generations=3, episodes=eps, ga_seed=0)
    b = train(pop_size=6, generations=3, episodes=eps, ga_seed=0)
    assert np.array_equal(a.best_genome, b.best_genome)


def test_train_multi_episode_smoke():
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=80)
    result = train(pop_size=8, generations=3, episodes=eps, ga_seed=0)
    assert len(result.history) == 3
    for h in result.history:
        assert np.isfinite(h["best"]) and np.isfinite(h["mean"]) and np.isfinite(h["min"])
    assert result.best_genome.shape == (510,)
    assert result.best_genome.dtype == np.float32


def test_train_multi_episode_pool_matches_serial():
    import multiprocessing as mp
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    eps = infinite_episodes([1234, 777], world_seed=1, max_steps=60)
    serial = train(pop_size=6, generations=2, episodes=eps, ga_seed=0, map_fn=map)
    with mp.Pool(2) as pool:
        par = train(pop_size=6, generations=2, episodes=eps, ga_seed=0,
                    map_fn=pool.imap)
    assert np.array_equal(serial.best_genome, par.best_genome)


def test_train_infinite_cli_writes_run(tmp_path):
    import json
    import os
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "main.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "train", "infinite", "--pop", "4", "--gens", "2",
         "--max-steps", "60", "--num-seeds", "2", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("inf1234x2_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert len(meta["episodes"]) == 2
    assert meta["lam"] == 1.0
    assert meta["aggregate"] == "mean_std"


def test_train_levels_cli_writes_run(tmp_path):
    import json
    import os
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "main.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "train", "levels", "--levels", "tutorial_hill",
         "--pop", "4", "--gens", "2", "--max-steps", "120", "--workers", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, r.stderr
    runs = list((tmp_path / "genomes").glob("lvls1_w1_*"))
    assert len(runs) == 1
    assert (runs[0] / "final_best.npy").exists()
    meta = json.loads((runs[0] / "run.json").read_text())
    assert len(meta["episodes"]) == 1
    assert meta["episodes"][0]["kind"] == "static"
    assert meta["episodes"][0]["norm"] > 1.0
    assert meta["aggregate"] == "min"


def test_train_levels_cli_unknown_level_errors(tmp_path):
    import os
    import subprocess
    import sys
    repo_root = Path(blueball.__file__).resolve().parents[2]
    script = repo_root / "main.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    r = subprocess.run(
        [sys.executable, str(script), "train", "levels", "--levels", "nope",
         "--pop", "2", "--gens", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env,
    )
    assert r.returncode != 0
    assert "Available" in (r.stderr + r.stdout)


def test_train_legacy_infinite_seed_equals_one_episode_path():
    # Backward-compat anchor: the legacy single-target arg (infinite_seed=) must
    # produce the byte-identical best_genome as the explicit one-episode path.
    from blueball.ai.episodes import infinite_episodes
    from blueball.ai.trainer import train
    legacy = train(pop_size=6, generations=3, infinite_seed=1234,
                   world_seed=1, max_steps=80, ga_seed=0)
    eps = infinite_episodes([1234], world_seed=1, max_steps=80)
    multi = train(pop_size=6, generations=3, episodes=eps, ga_seed=0)
    assert np.array_equal(legacy.best_genome, multi.best_genome)
