import pytest

from blueball.levels.chunks.base import CHUNK_REGISTRY
# Importing the chunks package registers every chunk type
from blueball.levels import chunks  # noqa: F401
from blueball.levels.sampler import ChunkSampler


def test_sampler_is_deterministic_per_seed():
    a = list(ChunkSampler(seed=42, target_chunks=50))
    b = list(ChunkSampler(seed=42, target_chunks=50))
    assert a == b


def test_sampler_ends_with_goal():
    seq = list(ChunkSampler(seed=1, target_chunks=20))
    assert seq[-1] == {"type": "goal"}
    assert sum(1 for s in seq if s["type"] == "goal") == 1


def test_sampler_emits_only_sampler_included_chunks():
    seq = list(ChunkSampler(seed=1, target_chunks=50))
    for entry in seq:
        t = entry["type"]
        if t in ("goal", "checkpoint"):
            continue
        assert CHUNK_REGISTRY[t].sampler_include is True


def test_sampler_difficulty_ramps_with_progress():
    seq = [s for s in ChunkSampler(seed=7, target_chunks=200) if s["type"] not in ("goal", "checkpoint")]
    q = len(seq) // 4
    first_q = seq[:q]
    last_q = seq[-q:]
    avg_first = sum(CHUNK_REGISTRY[s["type"]].difficulty for s in first_q) / len(first_q)
    avg_last = sum(CHUNK_REGISTRY[s["type"]].difficulty for s in last_q) / len(last_q)
    assert avg_last > avg_first


def test_sampler_inserts_checkpoints_every_n_steps():
    seq = list(ChunkSampler(seed=2, target_chunks=100, checkpoint_every=20))
    # Find indices of checkpoints
    checkpoint_indices = [i for i, s in enumerate(seq) if s["type"] == "checkpoint"]
    # First checkpoint should be at index ~20 (after 20 emits)
    assert len(checkpoint_indices) >= 4
