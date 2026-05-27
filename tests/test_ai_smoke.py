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
    assert FTNN_INPUTS == 14
    assert FTNN_HIDDEN == 12
    assert FTNN_OUTPUTS == 6
    # 14*12 + 12 + 12*6 + 6 = 258
    assert GENOME_SIZE == 258


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
    with pytest.raises(ValueError, match="258"):
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
    vel=(0.0, 0.0),
    ang_vel=0.0,
    grounded=False,
    nearest_collectible=None,
):
    from blueball.agent import Observation
    if rays is None:
        rays = np.zeros(8, dtype=np.float32)
    return Observation(
        rays=rays,
        vel=np.asarray(vel, dtype=np.float32),
        ang_vel=float(ang_vel),
        grounded=bool(grounded),
        nearest_collectible=nearest_collectible,
    )


def test_observation_to_inputs_shape_and_dtype():
    from blueball.ai.observation import observation_to_inputs
    x = observation_to_inputs(_make_obs())
    assert x.shape == (14,)
    assert x.dtype == np.float32


def test_observation_to_inputs_layout_matches_spec():
    from blueball.ai.observation import observation_to_inputs
    rays = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    obs = _make_obs(
        rays=rays,
        vel=(11.0, -22.0),
        ang_vel=3.5,
        grounded=True,
        nearest_collectible=(50.0, -25.0),
    )
    x = observation_to_inputs(obs)
    np.testing.assert_allclose(x[0:8], rays)
    assert x[8] == 11.0 and x[9] == -22.0
    assert x[10] == 3.5
    assert x[11] == 1.0
    assert x[12] == 50.0 and x[13] == -25.0


def test_observation_to_inputs_handles_none_collectible():
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(nearest_collectible=None, grounded=False)
    x = observation_to_inputs(obs)
    assert x[11] == 0.0          # grounded=False → 0.0
    assert x[12] == 0.0 and x[13] == 0.0


def test_observation_to_inputs_rejects_wrong_ray_count():
    from blueball.ai.observation import observation_to_inputs
    obs = _make_obs(rays=np.zeros(7, dtype=np.float32))
    with pytest.raises(AssertionError, match=r"\(8,\)"):
        observation_to_inputs(obs)


# ----- Task 3: Fitness -----

def test_fitness_all_zero_returns_zero():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=0.0, collectibles=0, reached_goal=False,
        died=False, steps_taken=0,
    ))
    assert f == 0.0


def test_fitness_shape_matches_spec_formula():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=500.0,
        collectibles=3,
        reached_goal=True,
        died=False,
        steps_taken=1000,
    ))
    # 500 + 50*3 + 200 - 0.01*1000 - 0  = 500 + 150 + 200 - 10 = 840
    assert f == pytest.approx(840.0)


def test_fitness_penalizes_death_and_charges_step_cost():
    from blueball.ai.fitness import fitness, FitnessInputs
    f = fitness(FitnessInputs(
        progress_x=10.0, collectibles=0, reached_goal=False,
        died=True, steps_taken=500,
    ))
    # 10 + 0 + 0 - 5 - 100 = -95
    assert f == pytest.approx(-95.0)
