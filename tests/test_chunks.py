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


def test_platform_chunk_adds_one_floating_segment():
    from blueball.levels.chunks.platform import Platform
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    width = Platform(width_tiles=4, y_offset=96).build(w, x_offset=100)
    assert width == 4 * TILE
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 1
    a, b = segs[0].a, segs[0].b
    assert a.y == GROUND_Y - 96
    assert b.y == GROUND_Y - 96


def test_platform_chunk_registry_and_attributes():
    from blueball.levels.chunks.platform import Platform
    assert "platform" in CHUNK_REGISTRY
    assert Platform.difficulty == 0
    assert Platform.sampler_include is True
    import random as _rng
    params = Platform.random_params(_rng.Random(0))
    assert 3 <= params["width_tiles"] <= 5
    assert params["y_offset"] in (64, 96, 128)


def test_ice_floor_chunk_uses_low_friction():
    from blueball.levels.chunks.ice_floor import IceFloor
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball import config
    w = World()
    IceFloor(width_tiles=3).build(w, x_offset=0)
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 1
    assert segs[0].friction == config.ICE_FLOOR_FRICTION
    assert segs[0].a.y == GROUND_Y


def test_ice_floor_attributes():
    from blueball.levels.chunks.ice_floor import IceFloor
    assert "ice_floor" in CHUNK_REGISTRY
    assert IceFloor.difficulty == 1


def test_one_way_platform_chunk_registered():
    from blueball.levels.chunks.one_way_platform import OneWayPlatformChunk
    assert "one_way_platform" in CHUNK_REGISTRY
    assert OneWayPlatformChunk.difficulty == 1


def test_checkpoint_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "checkpoint" in CHUNK_REGISTRY


def test_checkpoint_chunk_adds_one_checkpoint_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.checkpoint import Checkpoint
    w = World()
    chunk = CHUNK_REGISTRY["checkpoint"](width_tiles=2, id=1)
    width = chunk.build(w, x_offset=0.0)
    assert width == 2 * TILE
    checkpoints = [e for e in w.entities if isinstance(e, Checkpoint)]
    assert len(checkpoints) == 1
    assert checkpoints[0].id == 1


def test_checkpoint_chunk_sampler_include_false():
    from blueball.levels.chunks.checkpoint import CheckpointChunk
    assert CheckpointChunk.sampler_include is False
    assert CheckpointChunk.difficulty == 0


def test_crumbling_platform_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "crumbling_platform" in CHUNK_REGISTRY


def test_crumbling_platform_chunk_difficulty():
    from blueball.levels.chunks.crumbling_platform import CrumblingPlatformChunk
    assert CrumblingPlatformChunk.difficulty == 2


def test_crumbling_platform_chunk_adds_one_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.crumbling_platform import CrumblingPlatform
    w = World()
    chunk = CHUNK_REGISTRY["crumbling_platform"](width_tiles=4)
    width = chunk.build(w, x_offset=0.0)
    assert width == 4 * TILE
    platforms = [e for e in w.entities if isinstance(e, CrumblingPlatform)]
    assert len(platforms) == 1


def test_vertical_column_builds_alternating_platforms():
    from blueball.levels.chunks.vertical_column import VerticalColumn
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    width = VerticalColumn(width_tiles=6, steps=4, step_height=80, bottom_offset=96, platform_tiles=2).build(w, x_offset=0)
    assert width == 6 * TILE
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 4
    # Sorted by y descending (bottom to top in pymunk y-down): smallest y is highest.
    segs_sorted = sorted(segs, key=lambda s: -s.a.y)  # bottom first
    ys = [s.a.y for s in segs_sorted]
    assert ys == [GROUND_Y - 96, GROUND_Y - 176, GROUND_Y - 256, GROUND_Y - 336]
    # Even-index hug left (start at x=0); odd-index hug right (end at width)
    assert segs_sorted[0].a.x == 0
    assert segs_sorted[1].b.x == 6 * TILE
    assert segs_sorted[2].a.x == 0
    assert segs_sorted[3].b.x == 6 * TILE


def test_moving_platform_chunk_registered():
    from blueball.levels.chunks.moving_platform import MovingPlatformChunk
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "moving_platform" in CHUNK_REGISTRY
    assert MovingPlatformChunk.difficulty == 2


def test_moving_platform_chunk_builds_one_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.moving_platform import MovingPlatform
    w = World()
    chunk = CHUNK_REGISTRY["moving_platform"](width_tiles=4)
    width = chunk.build(w, x_offset=0.0)
    assert width == 4 * TILE
    platforms = [e for e in w.entities if isinstance(e, MovingPlatform)]
    assert len(platforms) == 1


# ---------------------------------------------------------------------------
# PushableBoxChunk
# ---------------------------------------------------------------------------

def test_pushable_box_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "pushable_box" in CHUNK_REGISTRY


def test_pushable_box_chunk_difficulty():
    from blueball.levels.chunks.pushable_box import PushableBoxChunk
    assert PushableBoxChunk.difficulty == 2


def test_pushable_box_chunk_builds_one_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.pushable_box import PushableBox
    w = World()
    chunk = CHUNK_REGISTRY["pushable_box"](width_tiles=2)
    width = chunk.build(w, x_offset=0.0)
    assert width == 2 * TILE
    boxes = [e for e in w.entities if isinstance(e, PushableBox)]
    assert len(boxes) == 1


def test_pushable_box_chunk_random_params():
    import random as _rng
    from blueball.levels.chunks.pushable_box import PushableBoxChunk
    params = PushableBoxChunk.random_params(_rng.Random(0))
    assert 2 <= params["width_tiles"] <= 3
    assert params["size_px"] in (28, 32, 40)
    assert 0.4 <= params["mass"] <= 0.8


# ---------------------------------------------------------------------------
# Key chunk
# ---------------------------------------------------------------------------

def test_key_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "key" in CHUNK_REGISTRY


def test_key_chunk_sampler_include_false_and_difficulty():
    from blueball.levels.chunks.key import KeyChunk
    assert KeyChunk.sampler_include is False
    assert KeyChunk.difficulty == 1


def test_key_chunk_builds_one_key_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.key import Key
    w = World()
    chunk = CHUNK_REGISTRY["key"](width_tiles=2, key_id=0)
    width = chunk.build(w, x_offset=0.0)
    assert width == 2 * TILE
    keys = [e for e in w.entities if isinstance(e, Key)]
    assert len(keys) == 1
    assert keys[0].key_id == 0


def test_key_chunk_key_id_stored():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    from blueball.entities.key import Key
    w = World()
    chunk = CHUNK_REGISTRY["key"](width_tiles=2, key_id=5)
    chunk.build(w, x_offset=0.0)
    keys = [e for e in w.entities if isinstance(e, Key)]
    assert keys[0].key_id == 5


# ---------------------------------------------------------------------------
# Door chunk
# ---------------------------------------------------------------------------

def test_door_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "door" in CHUNK_REGISTRY


def test_door_chunk_sampler_include_false_and_difficulty():
    from blueball.levels.chunks.door import DoorChunk
    assert DoorChunk.sampler_include is False
    assert DoorChunk.difficulty == 0


def test_door_chunk_builds_one_door_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.door import Door
    w = World()
    chunk = CHUNK_REGISTRY["door"](width_tiles=2, key_id=0)
    width = chunk.build(w, x_offset=0.0)
    assert width == 2 * TILE
    doors = [e for e in w.entities if isinstance(e, Door)]
    assert len(doors) == 1
    assert doors[0].key_id == 0


def test_door_chunk_key_id_stored():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    from blueball.entities.door import Door
    w = World()
    chunk = CHUNK_REGISTRY["door"](width_tiles=3, key_id=4)
    chunk.build(w, x_offset=0.0)
    doors = [e for e in w.entities if isinstance(e, Door)]
    assert doors[0].key_id == 4


# ---------------------------------------------------------------------------
# SwingingHazardChunk
# ---------------------------------------------------------------------------

def test_swinging_hazard_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "swinging_hazard" in CHUNK_REGISTRY


def test_swinging_hazard_chunk_difficulty():
    from blueball.levels.chunks.swinging_hazard import SwingingHazardChunk
    assert SwingingHazardChunk.difficulty == 3


def test_swinging_hazard_chunk_builds_one_entity():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.swinging_hazard import SwingingHazard
    w = World()
    chunk = CHUNK_REGISTRY["swinging_hazard"](width_tiles=3, rope_length=80, bob_mass=2.0, initial_angle_deg=15.0)
    width = chunk.build(w, x_offset=0.0)
    assert width == 3 * TILE
    hazards = [e for e in w.entities if isinstance(e, SwingingHazard)]
    assert len(hazards) == 1


def test_swinging_hazard_chunk_random_params():
    import random as _rng
    from blueball.levels.chunks.swinging_hazard import SwingingHazardChunk
    params = SwingingHazardChunk.random_params(_rng.Random(42))
    assert 3 <= params["width_tiles"] <= 6
    assert 60 <= params["rope_length"] <= 150
    assert 1.0 <= params["bob_mass"] <= 3.0
    assert -30 <= params["initial_angle_deg"] <= 30


# ---------------------------------------------------------------------------
# charger_platform
# ---------------------------------------------------------------------------

def test_charger_platform_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "charger_platform" in CHUNK_REGISTRY


def test_charger_platform_chunk_difficulty():
    from blueball.levels.chunks.charger_platform import ChargerPlatformChunk
    assert ChargerPlatformChunk.difficulty == 3


def test_charger_platform_chunk_builds_one_charger():
    from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
    from blueball.entities.charger import Charger
    w = World()
    chunk = CHUNK_REGISTRY["charger_platform"](length_tiles=8, facing="right")
    width = chunk.build(w, x_offset=0.0)
    assert width == 8 * TILE
    chargers = [e for e in w.entities if isinstance(e, Charger)]
    assert len(chargers) == 1


def test_charger_platform_chunk_random_params():
    import random as _rng
    from blueball.levels.chunks.charger_platform import ChargerPlatformChunk
    params = ChargerPlatformChunk.random_params(_rng.Random(42))
    assert 6 <= params["length_tiles"] <= 10
    assert params["facing"] in ("left", "right")


# ---------------------------------------------------------------------------
# spike_wall
# ---------------------------------------------------------------------------

def test_spike_wall_chunk_places_oriented_spikes():
    from blueball.levels.chunks.spike_wall import SpikeWall
    from blueball.entities.spike import Spike
    w = World()
    SpikeWall(width_tiles=4, spikes=4, orientation="down", ceiling_y_offset=160).build(w, x_offset=0)
    spikes = [e for e in w.entities if isinstance(e, Spike)]
    assert len(spikes) == 4
    for s in spikes:
        assert s.orientation == "down"


def test_spike_wall_chunk_in_registry():
    from blueball.levels.chunks.base import CHUNK_REGISTRY
    assert "spike_wall" in CHUNK_REGISTRY


def test_spike_wall_chunk_difficulty():
    from blueball.levels.chunks.spike_wall import SpikeWall
    assert SpikeWall.difficulty == 2


def test_spike_wall_chunk_ceiling_y_placement():
    from blueball.levels.chunks.spike_wall import SpikeWall
    from blueball.entities.spike import Spike
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    SpikeWall(width_tiles=3, spikes=2, orientation="down", ceiling_y_offset=140).build(w, x_offset=0)
    spikes = [e for e in w.entities if isinstance(e, Spike)]
    for s in spikes:
        assert s.position[1] == GROUND_Y - 140


def test_spike_wall_chunk_up_orientation_places_at_ground():
    from blueball.levels.chunks.spike_wall import SpikeWall
    from blueball.entities.spike import Spike
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    SpikeWall(width_tiles=3, spikes=2, orientation="up", ceiling_y_offset=160).build(w, x_offset=0)
    spikes = [e for e in w.entities if isinstance(e, Spike)]
    for s in spikes:
        assert s.position[1] == GROUND_Y


def test_spike_wall_chunk_has_flat_ground_segment():
    from blueball.levels.chunks.spike_wall import SpikeWall
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    SpikeWall(width_tiles=3, spikes=2, orientation="down", ceiling_y_offset=160).build(w, x_offset=0)
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment) and s.body is w.space.static_body]
    assert len(segs) == 1
    assert segs[0].a.y == GROUND_Y
    assert segs[0].b.y == GROUND_Y


def test_spike_wall_chunk_random_params():
    import random as _rng
    from blueball.levels.chunks.spike_wall import SpikeWall
    params = SpikeWall.random_params(_rng.Random(42))
    assert 2 <= params["width_tiles"] <= 4
    assert 2 <= params["spikes"] <= 4
    assert params["orientation"] in ("down", "left", "right")
    assert params["ceiling_y_offset"] in (128, 160, 200)
