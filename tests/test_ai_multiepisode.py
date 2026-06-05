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
    from blueball.ai.episodes import compute_level_par
    path = _levels_dir() / "tutorial_hill.json"
    # tutorial_hill has a goal, no keys, no collectibles -> par = width + 200
    meta = _load_meta(path)
    assert compute_level_par(path) == pytest.approx(meta.total_width + 200.0)


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
