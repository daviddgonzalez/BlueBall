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
