"""Double-jump chunk family + the sampler's ability-gating.

These chunks are the only ones in the Infinite Run pool that *require* a double
jump. The reachability bounds below are derived from config so the "solvable by
construction" guarantees track the real physics:

  - a single jump peaks at ~103 px and clears a ~413 px (center-to-center) flat
    gap; a double jump roughly doubles both.

Each chunk is checked to be (a) clearable with a double jump and (b), for the
capstones, genuinely *beyond* a single jump — and the sampler is checked to keep
them out of single-jump runs entirely.
"""

import random

import pymunk

from blueball import config
from blueball.abilities import Ability
from blueball.world import World
from blueball.levels.chunks.base import CHUNK_REGISTRY, TILE
from blueball.levels.sampler import ChunkSampler
from blueball.levels.streaming import TerrainStream

# --- reachability envelope, MEASURED through the real physics --------------
# Driving the real Player with optimal jump timing clears far more than the naive
# v^2/2g projectile formulas predict — and the GA finds that optimum, so the
# chunks are sized against these measured ceilings, not the textbook ones.
SINGLE_MOUNT_MAX = 172   # tallest flush wall a single jump mounts
DOUBLE_MOUNT_MAX = 260   # tallest flush wall a double jump mounts
LEDGE_SINGLE_CEIL = 152  # tallest gap+cliff a single jump mounts (worst case, 3-tile gap)
SINGLE_GAP_REACH = 420   # widest equal-height flat gap a single jump clears
DOUBLE_GAP_REACH = 720   # ... that a double jump clears

DJ = frozenset({Ability.DOUBLE_JUMP})
NEW_TYPES = ("double_gap", "double_ledge", "double_ledge_high", "double_step")
BASE_Y = 600.0


# --------------------------------------------------------------------------- #
# registration + ability metadata
# --------------------------------------------------------------------------- #
def test_new_chunks_registered():
    for name in NEW_TYPES:
        assert name in CHUNK_REGISTRY, f"{name} not registered"


def test_new_chunks_require_double_jump_and_are_sampler_included():
    for name in NEW_TYPES:
        cls = CHUNK_REGISTRY[name]
        assert cls.requires_ability is Ability.DOUBLE_JUMP, name
        assert cls.sampler_include is True, name


# --------------------------------------------------------------------------- #
# sampler ability-gating (the regression guard for single-jump runs)
# --------------------------------------------------------------------------- #
def test_default_sampler_excludes_double_jump_chunks():
    # A single-jump run (no abilities) must never see a double-jump chunk.
    for seed in range(5):
        seq = list(ChunkSampler(seed=seed, target_chunks=400))
        emitted = {s["type"] for s in seq}
        assert emitted.isdisjoint(NEW_TYPES), emitted & set(NEW_TYPES)


def test_default_pool_has_no_ability_gated_chunks():
    pool = ChunkSampler(seed=1, target_chunks=10)._pool
    assert all(cls.requires_ability is None for _, cls in pool)


def test_sampler_with_double_jump_surfaces_the_new_chunks():
    # Across a handful of long runs, every new chunk type should appear.
    emitted: set[str] = set()
    for seed in range(6):
        emitted |= {s["type"] for s in ChunkSampler(seed=seed, target_chunks=400, abilities=DJ)}
    for name in NEW_TYPES:
        assert name in emitted, f"{name} never sampled with double jump granted"


def test_default_sampler_unchanged_by_ability_param():
    # Passing the default (empty) abilities is identical to passing nothing.
    a = list(ChunkSampler(seed=99, target_chunks=120))
    b = list(ChunkSampler(seed=99, target_chunks=120, abilities=frozenset()))
    assert a == b


# --------------------------------------------------------------------------- #
# geometry — built into a real World
# --------------------------------------------------------------------------- #
def _segments(world):
    return [s for s in world.space.shapes if isinstance(s, pymunk.Segment)]


def test_double_gap_is_a_pure_hole():
    from blueball.levels.chunks.double_gap import DoubleGap

    w = World()
    width = DoubleGap(width_tiles=16).build(w, x_offset=0, base_y=BASE_Y)
    assert width == 16 * TILE
    # A fall-death gap builds no geometry — the surrounding flats own the ground.
    assert _segments(w) == []
    assert DoubleGap.entry_dy == 0.0 and DoubleGap.exit_dy == 0.0


def test_double_ledge_lands_high_and_returns_to_base():
    from blueball.levels.chunks.double_ledge import DoubleLedge

    w = World()
    chunk = DoubleLedge(approach_tiles=2, gap_tiles=5, ledge_tiles=3, exit_tiles=2, height=104)
    width = chunk.build(w, x_offset=0, base_y=BASE_Y)
    assert width == (2 + 5 + 3 + 2) * TILE
    segs = _segments(w)
    ys = {round(min(s.a.y, s.b.y), 3) for s in segs}
    # An approach/exit at base, a landing ledge raised by `height`.
    assert BASE_Y in ys
    assert (BASE_Y - 104) in ys
    # The chunk enters and exits at base level so seams stay continuous.
    assert chunk.entry_dy == 0.0 and chunk.exit_dy == 0.0
    # The gap is an actual hole: no floor segment bridges it at base level.
    approach_end = 2 * TILE
    ledge_start = (2 + 5) * TILE
    spanning = [
        s for s in segs
        if abs(s.a.y - BASE_Y) < 1e-6 and abs(s.b.y - BASE_Y) < 1e-6
        and min(s.a.x, s.b.x) < ledge_start and max(s.a.x, s.b.x) > approach_end
    ]
    assert spanning == []


def test_double_ledge_cliff_blocks_the_low_bypass():
    # The raised ledge must be fronted by a cliff wall spanning base..top. Without
    # it, a low single-jump arc (peak ~103px) sails *under* the ledge and lands on
    # the base-level exit beyond, skipping the climb — the bug found in playtest.
    from blueball.levels.chunks.double_ledge import DoubleLedge, DoubleLedgeHigh

    for cls, height in ((DoubleLedge, 104), (DoubleLedgeHigh, 144)):
        w = World()
        chunk = cls(approach_tiles=2, gap_tiles=5, ledge_tiles=3, exit_tiles=2, height=height)
        chunk.build(w, x_offset=0, base_y=BASE_Y)
        ledge_l = (2 + 5) * TILE
        walls = [
            s for s in _segments(w)
            if abs(s.a.x - s.b.x) < 1e-6 and abs(s.a.x - ledge_l) < 1e-6
        ]
        assert walls, f"{cls.__name__}: no cliff wall at the ledge's near edge"
        lo, hi = sorted((walls[0].a.y, walls[0].b.y))
        assert abs(lo - (BASE_Y - height)) < 1e-6   # top at the ledge surface
        assert abs(hi - BASE_Y) < 1e-6              # down to base — no gap under the ledge


def test_double_step_builds_wall_and_platform():
    from blueball.levels.chunks.double_step import DoubleStep

    w = World()
    chunk = DoubleStep(approach_tiles=2, plat_tiles=3, exit_tiles=2, height=136)
    width = chunk.build(w, x_offset=0, base_y=BASE_Y)
    assert width == (2 + 3 + 2) * TILE
    segs = _segments(w)
    # A vertical wall segment spanning base_y .. base_y-height.
    walls = [s for s in segs if abs(s.a.x - s.b.x) < 1e-6]
    assert any(
        abs(abs(s.a.y - s.b.y) - 136) < 1e-6 for s in walls
    ), "no vertical wall of the right height"
    # A platform at the raised level.
    ys = {round(min(s.a.y, s.b.y), 3) for s in segs}
    assert (BASE_Y - 136) in ys
    assert BASE_Y in ys
    assert chunk.entry_dy == 0.0 and chunk.exit_dy == 0.0


# --------------------------------------------------------------------------- #
# reachability invariants — solvable by construction
# --------------------------------------------------------------------------- #
def test_double_gap_width_beyond_single_jump_within_double():
    from blueball.levels.chunks.double_gap import DoubleGap

    for s in range(200):
        width_px = DoubleGap.random_params(random.Random(s))["width_tiles"] * TILE
        assert width_px > SINGLE_GAP_REACH    # past the single-jump reach
        assert width_px < DOUBLE_GAP_REACH    # within the double-jump reach


def test_double_ledge_high_and_step_require_double_jump():
    from blueball.levels.chunks.double_ledge import DoubleLedgeHigh
    from blueball.levels.chunks.double_step import DoubleStep

    for cls in (DoubleLedgeHigh, DoubleStep):
        for s in range(200):
            height = cls.random_params(random.Random(s))["height"]
            # Past even the flush single-jump ceiling (the gap+cliff one is lower),
            # so optimal single-jump timing can't cheese it ...
            assert height > SINGLE_MOUNT_MAX, f"{cls.__name__} height {height} single-mountable"
            # ... and inside the double-jump reach.
            assert height < DOUBLE_MOUNT_MAX, f"{cls.__name__} height {height} too tall"


def test_double_ledge_requires_double_jump():
    # Every diff-2 instance sits above the (gap-widened) single-jump cliff ceiling,
    # so there's no single-jump-feasible instance for the GA to opt out on.
    from blueball.levels.chunks.double_ledge import DoubleLedge

    for s in range(200):
        p = DoubleLedge.random_params(random.Random(s))
        assert p["height"] > LEDGE_SINGLE_CEIL    # > tallest single-jump cliff mount
        assert p["height"] < DOUBLE_MOUNT_MAX     # within the double-jump reach


def test_double_ledge_gaps_are_double_jump_solvable():
    from blueball.levels.chunks.double_ledge import DoubleLedge, DoubleLedgeHigh

    for cls in (DoubleLedge, DoubleLedgeHigh):
        for s in range(200):
            gap_px = cls.random_params(random.Random(s))["gap_tiles"] * TILE
            # Measured: the double jump clears the cliff for gaps up to 7 tiles.
            assert gap_px <= 7 * TILE


# --------------------------------------------------------------------------- #
# TerrainStream threads the granted abilities into its sampler
# --------------------------------------------------------------------------- #
def test_terrain_stream_threads_abilities_into_sampler():
    w = World()
    stream = TerrainStream(w, sampler_seed=1, abilities=DJ)
    assert Ability.DOUBLE_JUMP in stream.sampler.abilities


def test_terrain_stream_defaults_to_no_abilities():
    w = World()
    stream = TerrainStream(w, sampler_seed=1)
    assert stream.sampler.abilities == frozenset()


# --------------------------------------------------------------------------- #
# the showcase level chains all four chunks and is loadable
# --------------------------------------------------------------------------- #
def test_showcase_level_chains_all_chunks_and_loads():
    from blueball.debug.double_jump_play import build_showcase_level
    from blueball.levels.loader import load_level

    level = build_showcase_level()
    types = [c["type"] for c in level["chunks"]]
    for name in NEW_TYPES:
        assert name in types
    assert types[-1] == "goal"
    # Loads into a real World without error (geometry is valid).
    load_level(level, World())

