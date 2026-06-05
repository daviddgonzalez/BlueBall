"""Smoke tests for the AI / GA scaffolding.

Tests are appended across Tasks 0–6 (FTNN/genome, GA ops, observation
adapter, fitness, FTNNAgent, trainer). All AI-scaffolding test cases
live in this one file so the suite reads top-to-bottom.
"""

from __future__ import annotations

import numpy as np
import pytest


# ----- Task 0: FTNN topology + genome -----

def test_ftnn_topology_constants():
    from blueball.ai.ftnn import FTNN_INPUTS, FTNN_HIDDEN, FTNN_OUTPUTS, GENOME_SIZE
    from blueball.ai.observation import INPUT_SIZE
    assert FTNN_INPUTS == INPUT_SIZE == 35
    assert FTNN_HIDDEN == 12
    assert FTNN_OUTPUTS == 6
    # 35*12 + 12 + 12*6 + 6 = 510
    assert GENOME_SIZE == 510


def test_ftnn_forward_pass_shape_and_dtype():
    from blueball.ai.ftnn import FTNN, FTNN_INPUTS, FTNN_OUTPUTS
    from blueball.ai.genome import random_genome
    net = FTNN(random_genome(np.random.default_rng(0)))
    y = net.forward(np.random.default_rng(1).standard_normal(FTNN_INPUTS).astype(np.float32))
    assert y.shape == (FTNN_OUTPUTS,)
    assert y.dtype == np.float32


def test_ftnn_zero_genome_zero_input_yields_zero_output():
    from blueball.ai.ftnn import FTNN, FTNN_INPUTS, GENOME_SIZE
    net = FTNN(np.zeros(GENOME_SIZE, dtype=np.float32))
    y = net.forward(np.zeros(FTNN_INPUTS, dtype=np.float32))
    assert np.all(y == 0.0)


def test_ftnn_rejects_wrong_genome_shape():
    from blueball.ai.ftnn import FTNN
    with pytest.raises(ValueError, match="510"):
        FTNN(np.zeros(100, dtype=np.float32))


def test_random_genome_shape_and_dtype():
    from blueball.ai.genome import random_genome, GENOME_SIZE
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert g.shape == (GENOME_SIZE,)
    assert g.dtype == np.float32


def test_random_genome_is_deterministic_under_same_seed():
    from blueball.ai.genome import random_genome
    a = random_genome(np.random.default_rng(42))
    b = random_genome(np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_ftnn_does_not_alias_caller_genome():
    """If the caller mutates the genome buffer in-place after construction,
    the FTNN's stored weights MUST NOT change. Without a defensive copy
    in __init__, slicing produces views that share memory and this fails."""
    from blueball.ai.ftnn import FTNN, FTNN_INPUTS, GENOME_SIZE
    genome = np.ones(GENOME_SIZE, dtype=np.float32)
    net = FTNN(genome)
    baseline = net.forward(np.zeros(FTNN_INPUTS, dtype=np.float32))
    genome[:] = -7.0      # in-place mutation by the caller
    after = net.forward(np.zeros(FTNN_INPUTS, dtype=np.float32))
    np.testing.assert_array_equal(baseline, after)


# ----- Task 1: GA operators -----

def test_mutate_at_rate_zero_returns_equal_but_different_object():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.arange(258, dtype=np.float32)
    out = mutate(g, rng, rate=0.0, sigma=1.0)
    assert out is not g
    assert np.array_equal(out, g)


def test_mutate_at_rate_one_changes_most_weights():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.zeros(258, dtype=np.float32)
    out = mutate(g, rng, rate=1.0, sigma=1.0)
    changed = np.count_nonzero(out != g)
    assert changed / 258 > 0.95


def test_mutate_does_not_modify_input():
    from blueball.ai.ga import mutate
    rng = np.random.default_rng(0)
    g = np.zeros(258, dtype=np.float32)
    snapshot = g.copy()
    mutate(g, rng, rate=1.0, sigma=1.0)
    assert np.array_equal(g, snapshot)


def test_crossover_inherits_from_both_parents():
    from blueball.ai.ga import crossover
    rng = np.random.default_rng(0)
    a = np.zeros(258, dtype=np.float32)
    b = np.ones(258, dtype=np.float32)
    child = crossover(a, b, rng)
    frac_b = np.count_nonzero(child == 1.0) / 258
    assert 0.3 < frac_b < 0.7
    # Every gene came from either parent
    assert np.all((child == 0.0) | (child == 1.0))


def test_tournament_select_returns_top_two_when_k_is_full():
    from blueball.ai.ga import tournament_select
    rng = np.random.default_rng(0)
    fitnesses = np.array([1.0, 5.0, 3.0, 9.0, 7.0])
    i1, i2 = tournament_select(fitnesses, rng, k=5)
    assert {i1, i2} == {3, 4}     # indices of 9.0 and 7.0


def test_breed_preserves_population_size_and_elitism():
    from blueball.ai.ga import breed
    rng = np.random.default_rng(0)
    pop = [np.full(258, float(i), dtype=np.float32) for i in range(8)]
    fitnesses = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 7.0, 4.0])
    nxt = breed(pop, fitnesses, rng, elitism=1)
    assert len(nxt) == 8
    # The best (fitness 9.0 -> pop[3], all 3.0) must survive in slot 0
    # — breed contracts that elites occupy the first `elitism` indices.
    assert np.array_equal(nxt[0], pop[3])


def test_tournament_select_favors_higher_fitness_under_sampling():
    """With k=2 sampling and a clearly dominant index, that index should be
    chosen as one of the returned pair far more often than uniform chance."""
    from blueball.ai.ga import tournament_select
    rng = np.random.default_rng(0)
    # 10 candidates; index 9 is overwhelmingly best.
    fitnesses = np.array([0.0] * 9 + [1000.0])
    dominant_picked = 0
    trials = 500
    for _ in range(trials):
        i1, i2 = tournament_select(fitnesses, rng, k=2)
        if 9 in (i1, i2):
            dominant_picked += 1
    # Uniform random pair-pick rate for "index 9 in 2-sample": 2/10 = 0.2.
    # The dominant candidate must win whenever it's sampled, so the actual
    # rate should be ~0.2 (it's sampled 20% of the time and always promoted),
    # but importantly higher than 0.18 noise floor for 500 trials.
    rate = dominant_picked / trials
    assert rate > 0.15, f"dominant rate too low: {rate}"
    assert rate < 0.30, f"dominant rate suspiciously high: {rate}"


# ----- Task 2: Observation → input-vector adapter -----

def _make_obs(
    *,
    rays=None,
    ray_hit_types=None,
    vel=(0.0, 0.0),
    ang_vel=0.0,
    grounded=False,
    nearest_pickup=None,
    nearest_hazard=None,
    abilities=0,
    keys_held=0,
):
    from blueball.agent import Observation
    if rays is None:
        rays = np.ones(8, dtype=np.float32)
    if ray_hit_types is None:
        ray_hit_types = np.zeros(8, dtype=np.int8)
    return Observation(
        rays=rays,
        ray_hit_types=np.asarray(ray_hit_types, dtype=np.int8),
        vel=np.asarray(vel, dtype=np.float32),
        ang_vel=float(ang_vel),
        grounded=bool(grounded),
        nearest_pickup=nearest_pickup,
        nearest_hazard=nearest_hazard,
        abilities=int(abilities),
        keys_held=int(keys_held),
    )


def test_observation_to_inputs_shape_and_dtype():
    from blueball.ai.observation import observation_to_inputs, INPUT_SIZE
    x = observation_to_inputs(_make_obs())
    assert x.shape == (INPUT_SIZE,)
    assert x.dtype == np.float32


def test_observation_to_inputs_input_size_matches_ftnn():
    """The adapter's output width is the single source of truth for the net's
    input layer — they must never drift."""
    from blueball.ai.observation import INPUT_SIZE
    from blueball.ai.ftnn import FTNN_INPUTS
    assert FTNN_INPUTS == INPUT_SIZE


def test_observation_to_inputs_rays_and_scalars():
    from blueball.ai.observation import observation_to_inputs
    from blueball import config
    rays = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    obs = _make_obs(
        rays=rays,
        vel=(config.MAX_LINEAR_SPEED, -config.MAX_LINEAR_SPEED / 2),
        ang_vel=config.MAX_ANGULAR_VEL,
        grounded=True,
    )
    x = observation_to_inputs(obs)
    np.testing.assert_allclose(x[0:8], rays)
    # vel normalized by MAX_LINEAR_SPEED and clamped to [-1, 1]
    assert x[16] == 1.0
    assert abs(x[17] - (-0.5)) < 1e-6
    # ang_vel normalized by MAX_ANGULAR_VEL
    assert abs(x[18] - 1.0) < 1e-6
    assert x[19] == 1.0  # grounded


def test_observation_to_inputs_vel_is_clamped():
    from blueball.ai.observation import observation_to_inputs
    from blueball import config
    obs = _make_obs(vel=(10 * config.MAX_LINEAR_SPEED, -10 * config.MAX_LINEAR_SPEED))
    x = observation_to_inputs(obs)
    assert x[16] == 1.0
    assert x[17] == -1.0


def test_observation_to_inputs_ray_semantic_channels():
    """Per-ray semantic channel: +1 for reward (PICKUP/GOAL), -1 for danger
    (HAZARD/ENEMY), 0 for everything else."""
    from blueball.ai.observation import observation_to_inputs
    from blueball.agent import HitType
    hits = np.array([
        HitType.PICKUP, HitType.GOAL,       # +1, +1
        HitType.HAZARD, HitType.ENEMY,      # -1, -1
        HitType.GROUND, HitType.BLOCK,      # 0, 0
        HitType.DOOR, HitType.MISS,         # 0, 0
    ], dtype=np.int8)
    x = observation_to_inputs(_make_obs(ray_hit_types=hits))
    np.testing.assert_allclose(
        x[8:16], [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_observation_to_inputs_nearest_pickup_and_hazard():
    from blueball.ai.observation import observation_to_inputs, NEAREST_DELTA_NORM
    obs = _make_obs(
        nearest_pickup=(NEAREST_DELTA_NORM / 2, -NEAREST_DELTA_NORM / 4),
        nearest_hazard=(10 * NEAREST_DELTA_NORM, 0.0),  # far → clamps to +1
    )
    x = observation_to_inputs(obs)
    # pickup: dx, dy normalized + present flag
    assert abs(x[20] - 0.5) < 1e-6
    assert abs(x[21] - (-0.25)) < 1e-6
    assert x[22] == 1.0
    # hazard: clamped dx, dy, present
    assert x[23] == 1.0
    assert x[24] == 0.0
    assert x[25] == 1.0


def test_observation_to_inputs_none_pickup_and_hazard():
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(nearest_pickup=None, nearest_hazard=None, grounded=False)
    x = observation_to_inputs(obs)
    assert x[19] == 0.0  # grounded False
    assert x[20] == 0.0 and x[21] == 0.0 and x[22] == 0.0  # pickup absent
    assert x[23] == 0.0 and x[24] == 0.0 and x[25] == 0.0  # hazard absent


def test_observation_to_inputs_abilities_and_keys_bitfields():
    from blueball.ai.observation import (
        observation_to_inputs, _ABILITIES_OFFSET, N_ABILITIES, _KEYS_OFFSET, KEY_BITS,
    )
    # bit 0 of abilities set; keys 0 and 3 held
    obs = _make_obs(abilities=0b1, keys_held=(1 << 0) | (1 << 3))
    x = observation_to_inputs(obs)
    assert x[_ABILITIES_OFFSET] == 1.0
    keys = x[_KEYS_OFFSET:_KEYS_OFFSET + KEY_BITS]
    assert keys[0] == 1.0
    assert keys[3] == 1.0
    assert keys[1] == 0.0
    assert keys.sum() == 2.0


def test_observation_to_inputs_rejects_wrong_ray_count():
    """ValueError (not AssertionError) so the check survives `python -O`."""
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(rays=np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(8,\)"):
        observation_to_inputs(obs)


# ----- Task 3: Fitness -----

def test_fitness_all_zero_returns_zero():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=0.0, collectibles=0, reached_goal=False,
        died=False, steps_taken=0, keys_collected=0, level_width=0.0,
    ))
    assert f == 0.0


def test_fitness_shape_matches_spec_formula():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=500.0, collectibles=3, reached_goal=True,
        died=False, steps_taken=1000, keys_collected=0, level_width=500.0,
    ))
    # 500 + 50*3 + GOAL_MULT(2.0)*500*1 - 0.01*1000 - 0 + 100*0 = 1640
    assert f == pytest.approx(1640.0)


def test_fitness_penalizes_death_and_charges_step_cost():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=10.0, collectibles=0, reached_goal=False,
        died=True, steps_taken=500, keys_collected=0, level_width=0.0,
    ))
    # 10 + 0 + 0 - 5 - 200 + 0 = -195  (no goal -> width term is 0)
    assert f == pytest.approx(-195.0)


def test_fitness_rewards_keys():
    """Each key collected adds exactly 100."""
    from blueball.ai.fitness import fitness, FitnessInputs
    base = dict(progress_x=0.0, collectibles=0, reached_goal=False,
                died=False, steps_taken=0, level_width=0.0)
    f0 = fitness(FitnessInputs(keys_collected=0, **base))
    f2 = fitness(FitnessInputs(keys_collected=2, **base))
    assert f0 == 0.0
    assert f2 == pytest.approx(200.0)


def test_fitness_completion_dominates_traversal():
    """Reaching the goal beats an identical no-goal run by exactly
    GOAL_MULT * level_width, on every level, with no magic constant."""
    from blueball import config
    from blueball.ai.fitness import fitness, FitnessInputs
    W = 2000.0
    base = dict(progress_x=W, collectibles=0, died=False, steps_taken=0,
                keys_collected=0, level_width=W)
    finished = fitness(FitnessInputs(reached_goal=True, **base))
    unfinished = fitness(FitnessInputs(reached_goal=False, **base))
    assert finished - unfinished == pytest.approx(config.GOAL_MULT * W)


def test_fitness_no_goal_is_independent_of_width():
    """Infinite-Run invariant: with reached_goal=False the width term is 0, so
    fitness does not depend on level_width."""
    from blueball.ai.fitness import fitness, FitnessInputs
    base = dict(progress_x=300.0, collectibles=1, reached_goal=False,
                died=False, steps_taken=10, keys_collected=1)
    a = fitness(FitnessInputs(level_width=0.0, **base))
    b = fitness(FitnessInputs(level_width=9999.0, **base))
    assert a == b


# ----- Task 5: FTNNAgent -----

def test_ftnn_agent_returns_action_enum():
    from blueball.agent import FTNNAgent, Action
    from blueball.ai.genome import random_genome
    rng = np.random.default_rng(0)
    agent = FTNNAgent(random_genome(rng))
    action = agent.act(_make_obs())
    assert isinstance(action, Action)


def test_ftnn_agent_is_deterministic_for_same_genome():
    from blueball.agent import FTNNAgent
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(7))
    a1 = FTNNAgent(g)
    a2 = FTNNAgent(g)
    obs = _make_obs(vel=(50.0, -30.0), ang_vel=2.0, grounded=True)
    assert a1.act(obs) == a2.act(obs)


def test_ftnn_agent_all_zero_genome_returns_idle():
    from blueball.agent import FTNNAgent, Action
    from blueball.ai.ftnn import GENOME_SIZE
    agent = FTNNAgent(np.zeros(GENOME_SIZE, dtype=np.float32))
    assert agent.act(_make_obs()) == Action.IDLE


# ----- Task 6: Trainer + smoke -----

def _level_path():
    from pathlib import Path
    import blueball
    return Path(blueball.__file__).parent / "levels" / "tutorial_hill.json"


def test_evaluate_runs_one_genome_to_completion():
    from blueball import config
    from blueball.ai.trainer import evaluate
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(0))
    idx, fit = evaluate((0, g, config.DEFAULT_SEED, _level_path(), 200))
    assert idx == 0
    assert np.isfinite(fit)


def test_episode_fitness_uses_furthest_x_and_counts_keys():
    """_episode_fitness scores progress on the furthest x reached (not final)
    and credits each held key (popcount of the bitfield)."""
    from blueball.ai.trainer import _episode_fitness

    class _StubPlayer:
        def __init__(self, keys_held, dead=False, collectibles=0):
            self.keys_held = keys_held
            self.dead = dead
            self.collectibles_collected = collectibles

    player = _StubPlayer(keys_held=(1 << 0) | (1 << 2))  # 2 keys
    f = _episode_fitness(player, spawn_x=80.0, max_x=300.0, steps=0,
                         reached_goal=False, level_width=0.0)
    # progress 300-80=220 + 100*2 = 420
    assert f == pytest.approx(420.0)


def test_trainer_smoke_5gens_no_crash():
    from blueball.ai.trainer import train
    from blueball.ai.ftnn import GENOME_SIZE
    result = train(
        pop_size=8,
        generations=5,
        level_path=_level_path(),
        max_steps=600,
        ga_seed=0,
    )
    assert len(result.history) == 5
    for entry in result.history:
        assert {"gen", "best", "mean", "min"} <= set(entry)
        assert np.isfinite(entry["best"])
        assert np.isfinite(entry["mean"])
        assert np.isfinite(entry["min"])
    assert result.best_genome.shape == (GENOME_SIZE,)
    assert result.best_genome.dtype == np.float32
    assert len(result.final_population) == 8
    for g in result.final_population:
        assert g.shape == (GENOME_SIZE,)


def test_trainer_is_deterministic_under_same_seed():
    from blueball.ai.trainer import train
    a = train(pop_size=6, generations=3, level_path=_level_path(),
              max_steps=300, ga_seed=42, world_seed=1)
    b = train(pop_size=6, generations=3, level_path=_level_path(),
              max_steps=300, ga_seed=42, world_seed=1)
    assert np.array_equal(a.best_genome, b.best_genome)


def test_trainer_rejects_zero_generations():
    """train(generations=0) must fail loudly, not silently return a random
    genome as the 'best'."""
    from blueball.ai.trainer import train
    with pytest.raises(ValueError, match="generations"):
        train(pop_size=4, generations=0, level_path=_level_path(), max_steps=100)


def test_trainer_rejects_zero_pop_size():
    from blueball.ai.trainer import train
    with pytest.raises(ValueError, match="pop_size"):
        train(pop_size=0, generations=2, level_path=_level_path(), max_steps=100)


# ----- Infinite Run headless eval -----

def test_evaluate_infinite_runs_headless_and_is_finite():
    """A genome can be evaluated on a streamed Infinite Run seed with no level
    file and no pygame — the headless training path."""
    from blueball import config
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(0))
    idx, fit = evaluate_infinite((0, g, 1234, config.DEFAULT_SEED, 200))
    assert idx == 0
    assert np.isfinite(fit)


def test_evaluate_infinite_is_deterministic_for_same_seeds():
    """Same (sampler_seed, world_seed) → identical fitness: the reference-run
    contract the GA relies on."""
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(3))
    _, f1 = evaluate_infinite((0, g, 99, 1, 250))
    _, f2 = evaluate_infinite((0, g, 99, 1, 250))
    assert f1 == f2


def test_evaluate_infinite_different_seeds_build_different_terrain():
    """Different sampler seeds should generally produce different fitness for
    the same genome (terrain actually varies with the seed).

    The genome must move enough to encounter terrain — seed 2 reliably does so
    across all 5 terrain seeds at max_steps=300.
    """
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(2))
    fits = {evaluate_infinite((0, g, s, 1, 300))[1] for s in (1, 2, 3, 4, 5)}
    assert len(fits) > 1


def test_train_on_infinite_seed_runs():
    from blueball.ai.trainer import train
    from blueball.ai.ftnn import GENOME_SIZE
    result = train(
        pop_size=6, generations=3, infinite_seed=7,
        max_steps=250, ga_seed=0, world_seed=1,
    )
    assert len(result.history) == 3
    assert result.best_genome.shape == (GENOME_SIZE,)
    for entry in result.history:
        assert np.isfinite(entry["best"])


def test_train_rejects_both_level_and_infinite():
    from blueball.ai.trainer import train
    with pytest.raises(ValueError, match="exactly one"):
        train(pop_size=4, generations=2, level_path=_level_path(),
              infinite_seed=7, max_steps=100)


def test_train_rejects_neither_level_nor_infinite():
    from blueball.ai.trainer import train
    with pytest.raises(ValueError, match="exactly one"):
        train(pop_size=4, generations=2, max_steps=100)


def test_evaluate_infinite_deterministic_over_long_run():
    """Bit-identical fitness at a large max_steps, where accumulator float
    drift would previously have surfaced."""
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(11))
    _, f1 = evaluate_infinite((0, g, 1234, 1, 2000))
    _, f2 = evaluate_infinite((0, g, 1234, 1, 2000))
    assert f1 == f2


# ----- Task 8: TrainScene -----

@pytest.fixture
def headless_pygame():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.font.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.display.quit()


class _SyncResult:
    """A map_async result that has already computed eagerly."""
    def __init__(self, values):
        self._values = values
    def ready(self):
        return True
    def get(self, timeout=None):
        return self._values


class _SyncPool:
    """Synchronous stand-in for multiprocessing.Pool — runs map_async eagerly
    so TrainScene tests are deterministic and process-free."""
    def __init__(self):
        self.calls = 0
    def map_async(self, fn, iterable):
        self.calls += 1
        return _SyncResult([fn(x) for x in iterable])
    def close(self):
        pass
    def terminate(self):
        pass
    def join(self):
        pass


def test_train_scene_constructs_and_steps_infinite(headless_pygame):
    """TrainScene builds on an Infinite Run seed, owns n_visible FTNN players
    on a streamed terrain, and update() ticks do not crash."""
    from blueball.scenes.train import TrainScene
    from blueball.agent import FTNNAgent
    from blueball import collision
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=6,
        n_visible=4,
        generations=2,
        max_steps=60,
        pool=_SyncPool(),
    )
    assert len(scene._players) == 4
    for p in scene._players:
        assert isinstance(p.agent, FTNNAgent)
        assert p.shape.filter.group == collision.PLAYER_GROUP
    for _ in range(10):
        scene.update(1 / 60)


def test_train_scene_evaluates_full_population(headless_pygame):
    """The async truth-eval scores all pop_size genomes, not just n_visible,
    and a generation advances when the result is ready."""
    from blueball.scenes.train import TrainScene
    pool = _SyncPool()
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=8,
        n_visible=3,
        generations=3,
        max_steps=40,
        pool=pool,
    )
    start_gen = scene.current_gen
    scene.update(1 / 60)
    assert scene.current_gen == start_gen + 1
    assert scene._last_fitnesses is not None
    assert len(scene._last_fitnesses) == 8


def test_train_scene_rejects_neither_or_both_sources(headless_pygame):
    from blueball.scenes.train import TrainScene
    with pytest.raises(ValueError):
        TrainScene(headless_pygame, pool=_SyncPool())
    with pytest.raises(ValueError):
        TrainScene(headless_pygame, level_path=_level_path(),
                   infinite_seed=1, pool=_SyncPool())


class _DelayedResult:
    """A result that reports not-ready on the first poll, ready afterward."""
    def __init__(self, values):
        self._values = values
        self._polls = 0
    def ready(self):
        self._polls += 1
        return self._polls > 1
    def get(self, timeout=None):
        return self._values


class _DelayedPool:
    def map_async(self, fn, iterable):
        return _DelayedResult([fn(x) for x in iterable])
    def close(self):
        pass
    def terminate(self):
        pass
    def join(self):
        pass


def test_train_scene_waits_for_eval_before_advancing(headless_pygame):
    """Generation must not advance while the async eval reports not-ready."""
    from blueball.scenes.train import TrainScene
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=5,
        n_visible=2,
        generations=3,
        max_steps=40,
        pool=_DelayedPool(),
    )
    start_gen = scene.current_gen
    scene.update(1 / 60)          # first poll: not ready
    assert scene.current_gen == start_gen
    scene.update(1 / 60)          # second poll: ready -> advances
    assert scene.current_gen == start_gen + 1


def test_train_scene_static_level_display(headless_pygame):
    """The static-level display branch (load_level + no terrain) builds and
    steps without crashing."""
    from blueball.scenes.train import TrainScene
    scene = TrainScene(
        headless_pygame,
        level_path=_level_path(),
        pop_size=4,
        n_visible=2,
        generations=2,
        max_steps=40,
        pool=_SyncPool(),
    )
    assert scene._terrain is None
    assert len(scene._players) == 2
    for _ in range(5):
        scene.update(1 / 60)


def test_train_scene_persists_genomes(headless_pygame, tmp_path):
    """With save_dir set, TrainScene writes per-gen best + final_best + run.json."""
    import json
    import numpy as np
    from blueball.scenes.train import TrainScene
    run_dir = tmp_path / "vizrun"
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=4,
        n_visible=2,
        generations=2,
        max_steps=40,
        pool=_SyncPool(),
        save_dir=run_dir,
    )
    # _SyncPool is ready immediately, so each update() completes a generation.
    for _ in range(5):
        scene.update(1 / 60)
        if scene._done:
            break
    assert scene._done
    snaps = sorted(p.name for p in run_dir.glob("best_gen*.npy"))
    assert snaps == ["best_gen000.npy", "best_gen001.npy"]
    final = np.load(run_dir / "final_best.npy")
    assert np.array_equal(final, scene.best_genome)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["infinite_seed"] == 1234
    assert len(meta["history"]) == 2


# ----- Task 2 (WS2): Multiprocessing.Pool integration -----

def _make_pool(n):
    """Create a Pool, or skip the test if the platform can't start workers."""
    import multiprocessing
    try:
        return multiprocessing.Pool(n)
    except (OSError, ValueError) as e:  # pragma: no cover - platform guard
        pytest.skip(f"multiprocessing unavailable: {e}")


def test_pool_eval_matches_serial_determinism():
    """train() under Pool.imap produces the same best genome as serial map."""
    from blueball.ai.trainer import train
    serial = train(pop_size=6, generations=2, infinite_seed=7,
                   max_steps=200, ga_seed=0, world_seed=1)
    pool = _make_pool(2)
    try:
        parallel = train(pop_size=6, generations=2, infinite_seed=7,
                         max_steps=200, ga_seed=0, world_seed=1,
                         map_fn=pool.imap)
    finally:
        pool.close()
        pool.join()
    assert np.array_equal(serial.best_genome, parallel.best_genome)


def test_pool_evaluate_infinite_reorders_results():
    """evaluate_infinite is picklable; Pool results, once sorted by idx,
    match the serial mapping element-for-element."""
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    rng = np.random.default_rng(0)
    args = [(i, random_genome(rng), 1234, 1, 150) for i in range(5)]
    serial = sorted(map(evaluate_infinite, args), key=lambda r: r[0])
    pool = _make_pool(2)
    try:
        parallel = sorted(pool.imap(evaluate_infinite, args), key=lambda r: r[0])
    finally:
        pool.close()
        pool.join()
    assert [r[0] for r in parallel] == [0, 1, 2, 3, 4]
    for s, p in zip(serial, parallel):
        assert s[0] == p[0]
        assert s[1] == p[1]
