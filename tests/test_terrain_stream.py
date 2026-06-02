"""Tests for the pygame-free Infinite Run terrain streamer.

TerrainStream is the shared chunk-streaming state machine used by both
PlayScene (the live game) and the headless GA trainer, so the agent trains
on exactly the terrain a human would see for a given seed.
"""

import pymunk

from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.streaming import TerrainStream, MAX_GROUND_ELEV
from blueball.world import World


def _world():
    from blueball.collision import register
    w = World()
    register(w.space, world_ref=w)
    return w


def test_first_chunk_guarantees_ground_at_spawn():
    """The sampler can emit a floating chunk first; TerrainStream must lay a
    guaranteed Flat at x=0 so the player doesn't fall into the void."""
    w = _world()
    TerrainStream(w, sampler_seed=1)
    ground_segments = [
        s for s in w.space.shapes
        if isinstance(s, pymunk.Segment) and s.a.y == GROUND_Y and s.b.y == GROUND_Y
        and s.a.x <= 0 <= s.b.x
    ]
    assert ground_segments, "expected a ground segment spanning x=0 at GROUND_Y"


def test_initial_build_populates_chunks_ahead():
    w = _world()
    ts = TerrainStream(w, sampler_seed=1)
    assert ts.built_chunks, "initial build should materialize chunks"
    assert ts.build_x > 0.0


def test_sampler_excludes_checkpoints():
    w = _world()
    ts = TerrainStream(w, sampler_seed=1)
    assert ts.sampler.emit_checkpoints is False


def test_maintain_builds_ahead_and_culls_behind():
    w = _world()
    ts = TerrainStream(w, sampler_seed=2)
    initial_build_x = ts.build_x
    initial_count = len(ts.built_chunks)
    # Advance the "player" well past the current build cursor.
    ts.maintain(ts.build_x + 100)
    assert ts.build_x > initial_build_x
    assert len(ts.built_chunks) >= initial_count

    # Now jump very far ahead: chunks fully behind must be culled.
    far_x = ts.build_x + 5000
    ts.maintain(far_x)
    for info in ts.built_chunks:
        assert info["x_end"] >= far_x - ts.load_behind


def test_base_y_stays_within_elevation_band():
    w = _world()
    ts = TerrainStream(w, sampler_seed=3)
    for _ in range(50):
        ts.maintain(ts.build_x + 50)
        assert GROUND_Y - MAX_GROUND_ELEV - 1e-6 <= ts.base_y <= GROUND_Y + 1e-6


def test_same_seed_builds_identical_terrain():
    """Two streams with the same seed lay chunks at the same x positions."""
    a = TerrainStream(_world(), sampler_seed=42)
    b = TerrainStream(_world(), sampler_seed=42)
    for _ in range(20):
        a.maintain(a.build_x + 100)
        b.maintain(b.build_x + 100)
    a_bounds = [(c["x_start"], c["x_end"]) for c in a.built_chunks]
    b_bounds = [(c["x_start"], c["x_end"]) for c in b.built_chunks]
    assert a_bounds == b_bounds
