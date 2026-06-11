import pytest

from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.abilities import Ability
from blueball.levels.segment_stream import SegmentStream

ALL = frozenset({Ability.DOUBLE_JUMP})


def _world():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    return w


def test_construction_builds_initial_segments_with_increasing_boundaries():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    assert len(s.segment_ends) >= 4
    assert s.segment_ends == sorted(s.segment_ends)
    assert len(set(s.segment_ends)) == len(s.segment_ends)  # strictly increasing
    assert len(w.entities) > 0


def test_maintain_builds_ahead():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    n0 = len(s.segment_ends)
    s.maintain(player_x=s.build_x)  # ask for terrain at the frontier
    assert len(s.segment_ends) > n0


def test_maintain_culls_units_fully_behind_the_player():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    s.maintain(player_x=6000.0)
    ends_snapshot = list(s.segment_ends)
    s.maintain(player_x=12000.0)
    cutoff = 12000.0 - s.load_behind
    assert all(u["x_end"] >= cutoff for u in s.built)            # nothing behind remains
    assert s.segment_ends[: len(ends_snapshot)] == ends_snapshot  # history preserved
    assert all(sh in w.space.shapes for sh in list(w._shape_to_entity))  # no stale links


def test_materialize_calls_weld_ground_seams(monkeypatch):
    """Each _materialize call must invoke weld_ground_seams so chunk joints are
    smoothed in gym runs, matching the behaviour in loader.py."""
    call_count = []

    import blueball.levels.seams as seams_mod
    original = seams_mod.weld_ground_seams

    def _spy(space):
        call_count.append(1)
        original(space)

    monkeypatch.setattr(seams_mod, "weld_ground_seams", _spy)

    w = _world()
    # Construction materializes 1 flat footing + GYM_INITIAL_SEGMENTS segments.
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    count_after_init = len(call_count)
    assert count_after_init >= 1  # at least the footing call

    # An explicit maintain call triggers at least one more materialise.
    s.maintain(player_x=s.build_x)
    assert len(call_count) > count_after_init


def test_cull_removes_entities_and_shapes_from_space():
    w = _world()
    s = SegmentStream(w, seed=5, granted_abilities=ALL)
    s.maintain(player_x=4000.0)
    # Capture the earliest still-live built unit's entities and shapes.
    early_unit = s.built[0]
    early_entities = set(early_unit["entities"])
    early_shapes = set(early_unit["shapes"])
    assert early_entities or early_shapes  # the unit actually has content to cull
    s.maintain(player_x=20000.0)  # leave that early unit far behind -> culled
    # Its entities and shapes are gone from the world and the pymunk space.
    assert all(e not in w.entities for e in early_entities)
    assert all(sh not in w.space.shapes for sh in early_shapes)
