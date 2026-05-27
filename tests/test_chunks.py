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
from blueball.abilities import Ability
from blueball.entities.ability_pickup import AbilityPickup
from blueball.entities.boost_pad import BoostPad


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


def test_ability_pickup_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "ability_pickup" in CHUNK_REGISTRY


def test_ability_pickup_chunk_adds_one_pickup_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    w = World()
    chunk = CHUNK_REGISTRY["ability_pickup"](width_tiles=2, ability="double_jump")
    width = chunk.build(w, x_offset=0.0)
    from blueball.levels.chunks.base import TILE
    assert width == 2 * TILE
    pickups = [e for e in w.entities if isinstance(e, AbilityPickup)]
    assert len(pickups) == 1
    assert pickups[0].ability == Ability.DOUBLE_JUMP


def test_ability_pickup_chunk_rejects_unknown_ability():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    with pytest.raises(ValueError):
        CHUNK_REGISTRY["ability_pickup"](width_tiles=2, ability="frobnicate")


def test_boost_pad_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "boost_pad" in CHUNK_REGISTRY


def test_boost_pad_chunk_adds_one_boost_pad_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    w = World()
    chunk = CHUNK_REGISTRY["boost_pad"](width_tiles=3, multiplier=2.5)
    width = chunk.build(w, x_offset=100.0)
    assert width == 3 * TILE
    pads = [e for e in w.entities if isinstance(e, BoostPad)]
    assert len(pads) == 1
    assert pads[0].multiplier == 2.5
    # Pad width should match the segment span
    assert pads[0].width == 3 * TILE


def test_goal_chunk_uses_base_random_params_default():
    assert Flat.difficulty == 0
    assert Flat.sampler_include is True
    # random_params base default returns {}; GoalChunk never overrides it so it
    # still returns {} here (sampler_include=False keeps it out of sampling anyway).
    import random as _rng
    assert GoalChunk.random_params(_rng.Random(0)) == {}


def test_existing_chunks_difficulty_assigned():
    assert Flat.difficulty == 0
    assert Gap.difficulty == 1
    assert SpikePit.difficulty == 2
    from blueball.levels.chunks.falling_hazard import FallingHazardChunk
    assert FallingHazardChunk.difficulty == 3


def test_goal_and_ability_pickup_excluded_from_sampler():
    from blueball.levels.chunks.ability_pickup import AbilityPickupChunk
    assert GoalChunk.sampler_include is False
    assert AbilityPickupChunk.sampler_include is False


def test_flat_random_params_returns_width_in_range():
    import random as _rng
    params = Flat.random_params(_rng.Random(0))
    assert 2 <= params["width_tiles"] <= 5
