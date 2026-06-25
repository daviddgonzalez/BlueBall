import numpy as np

from blueball.agent import HitType, Observation
from blueball.ai.observation import INPUT_SIZE, observation_to_inputs
from blueball.ai.ftnn import GENOME_SIZE


def _blank_obs(**overrides):
    base = dict(
        rays=np.ones(8, dtype=np.float32),
        ray_hit_types=np.zeros(8, dtype=np.int8),
        vel=np.zeros(2, dtype=np.float32),
        ang_vel=0.0,
        grounded=False,
        nearest_pickup=None,
        nearest_hazard=None,
        abilities=0,
        keys_held=0,
    )
    base.update(overrides)
    return Observation(**base)


def test_input_and_genome_sizes():
    assert INPUT_SIZE == 37
    assert GENOME_SIZE == 534


def test_jump_state_packed_at_tail():
    obs = _blank_obs(air_jumps_remaining=1.0, can_jump_now=True)
    x = observation_to_inputs(obs)
    assert x.shape == (37,)
    assert x[35] == 1.0   # air_jumps_remaining (normalized)
    assert x[36] == 1.0   # can_jump_now flag


def test_jump_state_defaults_zero():
    x = observation_to_inputs(_blank_obs())
    assert x[35] == 0.0
    assert x[36] == 0.0
