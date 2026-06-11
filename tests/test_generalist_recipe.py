"""Track B generalist recipe: mixed_episodes constructor + static ability grant."""

from __future__ import annotations

import numpy as np

from blueball import cli
from blueball.abilities import Ability
from blueball.ai import trainer
from blueball.ai.episodes import (
    compute_level_par,
    mixed_episodes,
    resolve_level_paths,
    static_episodes,
)
from blueball.ai.genome import random_genome
from blueball.ai.ftnn import GENOME_SIZE
from blueball.ai.persistence import run_dir_name


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


def test_run_dir_name_generalist():
    name = run_dir_name(world_seed=1, timestamp="T", num_levels=5, generalist=True)
    assert name == "genL5_w1_T"


def test_per_kind_scores_grouping(monkeypatch):
    # Stub the three evaluators so the test is fast and the grouping is exact.
    def fake_infinite(args):
        idx, genome, seed, world_seed, max_steps = args
        return idx, float(seed)  # score == seed so we can check the average

    def fake_gym(args):
        idx, genome, seed, world_seed, max_steps, abilities = args
        return idx, float(seed)

    def fake_static(args):
        idx, genome, world_seed, level_path, max_steps, abilities = args
        return idx, 42.0

    monkeypatch.setattr(trainer, "evaluate_infinite", fake_infinite)
    monkeypatch.setattr(trainer, "evaluate_gym", fake_gym)
    monkeypatch.setattr(trainer, "evaluate", fake_static)

    eps = mixed_episodes(
        infinite_seeds=[10, 30],          # avg = 20.0
        level_names=["tutorial_hill"],
        gym_seeds=[4, 8],                 # avg = 6.0
        world_seed=1,
        max_steps=50,
        abilities=("double_jump",),
    )
    genome = random_genome(np.random.default_rng(0))
    scores = cli._per_kind_scores(genome, eps)

    assert scores["infinite"] == 20.0
    assert scores["gym"] == 6.0
    assert scores["static:tutorial_hill"] == 42.0
    assert set(scores) == {"infinite", "gym", "static:tutorial_hill"}


def test_warm_start_places_genome_first():
    seed = np.zeros(GENOME_SIZE)
    seed[0] = 1.0

    captured = {}

    def on_gen(gen, best, population):
        if gen == 0 and "pop0" not in captured:
            captured["pop0"] = np.asarray(population[0]).copy()

    trainer.train(
        pop_size=4,
        generations=1,
        infinite_seed=1,
        ga_seed=0,
        max_steps=60,
        init_genome=seed,
        on_generation=on_gen,
    )

    assert np.array_equal(captured["pop0"], seed)


def test_warm_start_none_is_unchanged():
    common = dict(
        pop_size=4,
        generations=2,
        infinite_seed=1,
        ga_seed=0,
        max_steps=60,
    )
    res_default = trainer.train(**common)
    res_none = trainer.train(init_genome=None, **common)

    assert res_default.history[-1]["best"] == res_none.history[-1]["best"]


def test_warm_start_wrong_length_raises():
    try:
        trainer.train(
            pop_size=4,
            generations=1,
            infinite_seed=1,
            ga_seed=0,
            max_steps=60,
            init_genome=np.zeros(3),
        )
    except ValueError as e:
        assert "GENOME_SIZE" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong-length init_genome")


def test_warm_start_reproduces_mover_score():
    mover = random_genome(np.random.default_rng(7))
    _, mover_score = trainer.evaluate_infinite((0, mover, 1234, 1, 300))

    res = trainer.train(
        pop_size=6,
        generations=1,
        infinite_seed=1234,
        ga_seed=0,
        max_steps=300,
        init_genome=mover,
    )

    assert res.history[0]["best"] >= mover_score - 1e-6


def test_warm_start_seed_slot_is_float32():
    # Genomes are float32 everywhere; the warm-start must not leak float64 into
    # the seeded population slot (which would propagate to best_genome and the
    # persisted .npy on the success path).
    seed = np.zeros(GENOME_SIZE, dtype=np.float64)
    seed[0] = 1.0

    captured = {}

    def on_gen(gen, best, population):
        if gen == 0 and "pop0" not in captured:
            captured["pop0"] = np.asarray(population[0])

    trainer.train(
        pop_size=4,
        generations=1,
        infinite_seed=1,
        ga_seed=0,
        max_steps=60,
        init_genome=seed,
        on_generation=on_gen,
    )

    assert captured["pop0"].dtype == np.float32
