"""Tests for trained-genome persistence to disk."""

import json
from pathlib import Path

import numpy as np
import pytest


def test_run_dir_name_encodes_seeds_for_infinite():
    from blueball.ai.persistence import run_dir_name
    name = run_dir_name(infinite_seed=1234, world_seed=1, timestamp="20260602-1903")
    assert name == "inf1234_w1_20260602-1903"


def test_run_dir_name_encodes_level_name():
    from blueball.ai.persistence import run_dir_name
    name = run_dir_name(level_name="tutorial_hill", world_seed=7, timestamp="T")
    assert name == "tutorial_hill_w7_T"


def test_writer_saves_generation_genome_roundtrips(tmp_path):
    from blueball.ai.persistence import TrainingRunWriter
    w = TrainingRunWriter(tmp_path / "run")
    g = np.arange(10, dtype=np.float32)
    path = w.save_generation(3, g)
    assert path.name == "best_gen003.npy"
    loaded = np.load(path)
    assert np.array_equal(loaded, g)


def test_writer_finalize_writes_final_and_run_json(tmp_path):
    from blueball.ai.persistence import TrainingRunWriter
    w = TrainingRunWriter(tmp_path / "run")
    best = np.ones(5, dtype=np.float32)
    meta = {"ga_seed": 0, "infinite_seed": 1234, "history": [{"gen": 0, "best": 1.0}]}
    w.finalize(best, meta)

    final = np.load(tmp_path / "run" / "final_best.npy")
    assert np.array_equal(final, best)
    run_json = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run_json["ga_seed"] == 0
    assert run_json["infinite_seed"] == 1234
    assert run_json["history"][0]["best"] == 1.0


def test_train_persists_when_save_dir_given(tmp_path):
    """train(save_dir=...) writes a per-generation best + final_best + run.json."""
    from blueball.ai.trainer import train
    run_dir = tmp_path / "run"
    result = train(
        pop_size=4, generations=3, infinite_seed=7,
        max_steps=120, ga_seed=0, world_seed=1, save_dir=run_dir,
    )
    # One snapshot per generation.
    snaps = sorted(p.name for p in run_dir.glob("best_gen*.npy"))
    assert snaps == ["best_gen000.npy", "best_gen001.npy", "best_gen002.npy"]
    # Final best matches the returned best genome.
    final = np.load(run_dir / "final_best.npy")
    assert np.array_equal(final, result.best_genome)
    # run.json captures the seeds + full history.
    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["infinite_seed"] == 7
    assert run_json["ga_seed"] == 0
    assert run_json["world_seed"] == 1
    assert len(run_json["history"]) == 3


def test_train_without_save_dir_writes_nothing(tmp_path):
    from blueball.ai.trainer import train
    train(pop_size=4, generations=2, infinite_seed=7, max_steps=80, save_dir=None)
    assert list(tmp_path.iterdir()) == []
