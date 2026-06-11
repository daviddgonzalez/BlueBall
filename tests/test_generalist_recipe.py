"""Track B generalist recipe: mixed_episodes constructor + static ability grant."""

from __future__ import annotations

import numpy as np

from blueball.abilities import Ability
from blueball.ai import trainer
from blueball.ai.episodes import (
    compute_level_par,
    mixed_episodes,
    resolve_level_paths,
    static_episodes,
)
from blueball.ai.genome import random_genome


def test_mixed_episodes_composition():
    eps = mixed_episodes(
        infinite_seeds=[1, 2],
        level_names=["tutorial_hill", "maze"],
        gym_seeds=[4242],
        world_seed=7,
        max_steps=100,
        abilities=("double_jump",),
    )
    kinds = [e.kind for e in eps]
    assert kinds == ["infinite", "infinite", "static", "static", "gym"]

    # static + gym episodes carry the granted abilities
    for e in eps:
        if e.kind in ("static", "gym"):
            assert e.abilities == ("double_jump",)

    # first static episode is normalized by its level par
    static_eps = [e for e in eps if e.kind == "static"]
    tutorial_path = resolve_level_paths(["tutorial_hill"])[0]
    assert static_eps[0].norm == compute_level_par(tutorial_path)

    # infinite/gym keep norm 1.0; static keeps par norm
    for e in eps:
        if e.kind in ("infinite", "gym"):
            assert e.norm == 1.0


def test_mixed_episodes_infinite_carries_abilities():
    eps = mixed_episodes(
        infinite_seeds=[1],
        level_names=["tutorial_hill"],
        gym_seeds=[],
        world_seed=1,
        max_steps=50,
        abilities=("double_jump",),
    )
    inf = [e for e in eps if e.kind == "infinite"]
    assert inf[0].abilities == ("double_jump",)


def test_static_episodes_backward_compat_no_abilities():
    eps = static_episodes(resolve_level_paths(["tutorial_hill"]), 1, 100)
    assert eps[0].abilities == ()


def test_static_evaluate_grants_episode_abilities(monkeypatch):
    captured = {}

    real_player = trainer.Player

    def spy_player(*args, **kwargs):
        captured["abilities"] = kwargs.get("abilities")
        return real_player(*args, **kwargs)

    monkeypatch.setattr(trainer, "Player", spy_player)

    tutorial_path = resolve_level_paths(["tutorial_hill"])[0]
    genome = random_genome(np.random.default_rng(0))
    trainer.evaluate((0, genome, 1, tutorial_path, 60, ("double_jump",)))

    assert Ability.DOUBLE_JUMP in captured["abilities"]
