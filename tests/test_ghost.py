import numpy as np
import pytest

from blueball import config
from blueball.scenes.ghost import GhostRunner, record_ghost_track


def test_ghost_runner_advances_one_row_per_phys_dt():
    track = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float32)
    g = GhostRunner(track)
    assert g.pose() == (0.0, 0.0, 0.0)
    g.update(config.PHYS_DT)
    assert g.pose() == (10.0, 0.0, 0.0)
    g.update(config.PHYS_DT * 2)
    assert g.pose() == (30.0, 0.0, 0.0)


def test_ghost_runner_freezes_on_last_pose():
    track = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    g = GhostRunner(track)
    g.update(config.PHYS_DT * 100)
    assert g.pose() == (10.0, 0.0, 0.0)
    assert g.done is True


def test_ghost_runner_rejects_empty_track():
    with pytest.raises(ValueError):
        GhostRunner(np.zeros((0, 3), dtype=np.float32))


def test_record_ghost_track_nonempty_and_faithful():
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.genome import random_genome
    from blueball.scenes.playback import PlaybackSim

    genome = random_genome(np.random.default_rng(0))
    path = resolve_level_paths(["tutorial_hill"])[0]
    track = record_ghost_track(genome, path, world_seed=1, max_steps=400,
                               abilities=("double_jump",))
    assert track.ndim == 2 and track.shape[1] == 3 and len(track) > 0

    sim = PlaybackSim(genome, mode="static", level_path=path, world_seed=1,
                      max_steps=400, abilities=("double_jump",))
    while not sim.done:
        sim.step_once()
    assert track[-1][0] == pytest.approx(sim.player.body.position.x, abs=1e-3)
    assert len(track) == sim.steps
