import pytest
import pymunk

from blueball.world import World
from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
from blueball.levels.chunks.flat import Flat
from blueball.levels.chunks.gap import Gap
from blueball.levels.chunks.spike_pit import SpikePit
from blueball.levels.chunks.patrol_platform import PatrolPlatform
from blueball.levels.chunks.stairs import StairsUp, StairsDown
from blueball.levels.chunks.bump import Bump
from blueball.levels.chunks.goal import GoalChunk


def test_registry_has_all_v1_chunks():
    for name in (
        "flat",
        "gap",
        "spike_pit",
        "patrol_platform",
        "stairs_up",
        "stairs_down",
        "bump",
        "goal",
        "falling_hazard",
    ):
        assert name in CHUNK_REGISTRY


def test_flat_adds_one_segment_and_reports_width():
    w = World()
    width = Flat(width_tiles=8).build(w, x_offset=100)
    assert width == 8 * TILE
    # Exactly one static segment was added
    static_segments = [
        s for s in w.space.shapes
        if isinstance(s, pymunk.Segment) and s.body is w.space.static_body
    ]
    assert len(static_segments) == 1


def test_gap_adds_no_geometry():
    w = World()
    width = Gap(width_tiles=3).build(w, x_offset=200)
    assert width == 3 * TILE
    # No new shapes added
    assert len(list(w.space.shapes)) == 0


def test_spike_pit_adds_n_spike_entities():
    w = World()
    width = SpikePit(width_tiles=4, spikes=4).build(w, x_offset=0)
    assert width == 4 * TILE
    from blueball.entities.spike import Spike
    spikes = [e for e in w.entities if isinstance(e, Spike)]
    assert len(spikes) == 4


def test_patrol_platform_adds_one_patroller():
    w = World()
    PatrolPlatform(length_tiles=6, patroller_speed=80).build(w, x_offset=0)
    from blueball.entities.patroller import Patroller
    patrollers = [e for e in w.entities if isinstance(e, Patroller)]
    assert len(patrollers) == 1
    assert patrollers[0].speed == 80


def test_goal_chunk_adds_one_goal_entity():
    w = World()
    width = GoalChunk().build(w, x_offset=500)
    from blueball.entities.goal import Goal
    goals = [e for e in w.entities if isinstance(e, Goal)]
    assert len(goals) == 1
