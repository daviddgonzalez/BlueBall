import numpy as np
import pytest

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


def test_legacy_migration_zero_pads_new_inputs():
    from blueball.ai.ftnn import FTNN, LEGACY_GENOME_SIZE, GENOME_SIZE, migrate_genome
    rng = np.random.default_rng(1)
    legacy = rng.standard_normal(LEGACY_GENOME_SIZE).astype(np.float32)
    migrated = migrate_genome(legacy)
    assert migrated.shape == (GENOME_SIZE,)
    net = FTNN(legacy)  # FTNN must accept a legacy genome directly
    x = rng.standard_normal(37).astype(np.float32)
    y0 = net.forward(x.copy())
    x[35] = 5.0
    x[36] = -3.0  # perturb ONLY the two new inputs
    y1 = net.forward(x)
    assert np.allclose(y0, y1)  # zero weights -> new inputs have no effect
    assert migrated.dtype == np.float32
    legacy64 = rng.standard_normal(LEGACY_GENOME_SIZE)  # float64
    assert migrate_genome(legacy64).dtype == np.float32


def test_current_genome_passes_through_and_bad_size_raises():
    from blueball.ai.ftnn import GENOME_SIZE, migrate_genome
    rng = np.random.default_rng(2)
    cur = rng.standard_normal(GENOME_SIZE).astype(np.float32)
    assert migrate_genome(cur) is cur
    with pytest.raises(ValueError):
        migrate_genome(rng.standard_normal(999).astype(np.float32))
