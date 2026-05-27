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
