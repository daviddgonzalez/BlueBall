"""Track B generalist recipe: mixed_episodes constructor + static ability grant.

Backward-compat note: Task 0 (`starting_abilities`) INTENTIONALLY grants the
`maze` level double-jump as a deliberate foundation change, so `maze` is NOT
asserted to be byte-identical to its pre-Track-B behavior. The backward-compat
guarantees below are pinned on the parts B's episodes/warm-start/CLI additions
must NOT perturb: the infinite/gym single-mode trainer, non-maze static
evaluation (e.g. `tutorial_hill`, which declares no starting_abilities), and
the invariance of the new optional `init_genome` default (None == omitted).
"""

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
from blueball.cli import build_parser


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

    # cross-kind normalization: infinite/gym get par divisors (NOT 1.0) so the
    # min objective balances all three kinds instead of degenerating to the worst
    # static level. All three kinds' pars are a comparable order of magnitude.
    from blueball import config
    inf_eps = [e for e in eps if e.kind == "infinite"]
    gym_eps = [e for e in eps if e.kind == "gym"]
    assert all(e.norm == config.GENERALIST_INFINITE_PAR for e in inf_eps)
    assert all(e.norm == config.GENERALIST_GYM_PAR for e in gym_eps)
    assert inf_eps[0].norm > 100.0 and gym_eps[0].norm > 100.0
    assert static_eps[0].norm > 100.0  # par-normalized static is the same scale


def test_mixed_episodes_marks_specialist_levels_min_exempt():
    # maze + vertical_climb are trained as standalone specialists, so the
    # generalist's `min` objective must exempt them (they don't dominate the
    # worst-case). Every other episode stays non-exempt.
    from pathlib import Path
    from blueball import config

    eps = mixed_episodes(
        infinite_seeds=[1],
        level_names=list(config.GENERALIST_LEVELS),
        gym_seeds=[4242],
        world_seed=1,
        max_steps=100,
        abilities=("double_jump",),
    )
    exempt_stems = {
        Path(e.level_path).stem
        for e in eps if e.kind == "static" and e.min_exempt
    }
    assert exempt_stems == set(config.GENERALIST_MIN_EXEMPT_LEVELS)

    # Nothing outside the exempt set is flagged (infinite, gym, other statics).
    for e in eps:
        is_exempt_level = (e.kind == "static"
                           and Path(e.level_path).stem in config.GENERALIST_MIN_EXEMPT_LEVELS)
        assert e.min_exempt == is_exempt_level


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
        idx, genome, seed, world_seed, max_steps, abilities = args
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


def test_train_generalist_writes_artifacts(tmp_path, monkeypatch):
    # Point the genomes root at a temp dir. `cli._run_dir` does a fresh local
    # `from .ai.persistence import GENOMES_ROOT` at call time, so patching the
    # module attribute is sufficient.
    import blueball.ai.persistence as persistence

    monkeypatch.setattr(persistence, "GENOMES_ROOT", tmp_path)

    args = build_parser().parse_args(
        ["train", "generalist", "--pop", "4", "--gens", "1",
         "--max-steps", "150", "--workers", "1"])
    rc = args.func(args)
    assert rc == 0

    run_dirs = list(tmp_path.glob("genL5_*"))
    assert len(run_dirs) == 1, f"expected one genL5_* run dir, got {run_dirs}"
    run_dir = run_dirs[0]

    for artifact in ("final_best.npy", "per_kind_scores.json", "run.json"):
        assert (run_dir / artifact).exists(), f"missing {artifact} in {run_dir}"


def test_optional_params_dont_change_single_mode():
    # The new optional `init_genome` param must default to omitted behavior:
    # train() with no init_genome == train(init_genome=None) == a repeat call.
    common = dict(
        infinite_seed=1234, ga_seed=0, pop_size=4, generations=2, max_steps=200)
    res_default = trainer.train(**common)
    res_none = trainer.train(init_genome=None, **common)
    res_repeat = trainer.train(**common)

    best = res_default.history[-1]["best"]
    assert res_none.history[-1]["best"] == best
    assert res_repeat.history[-1]["best"] == best


def test_non_maze_static_eval_is_deterministic():
    # tutorial_hill declares no starting_abilities, so Task 0/1 must not perturb
    # its static evaluation: two identical runs produce the same best.
    eps = static_episodes(resolve_level_paths(["tutorial_hill"]), 1, 150)
    a = trainer.train(episodes=eps, pop_size=4, generations=1, ga_seed=0)
    b = trainer.train(episodes=eps, pop_size=4, generations=1, ga_seed=0)
    assert a.history[-1]["best"] == b.history[-1]["best"]


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
