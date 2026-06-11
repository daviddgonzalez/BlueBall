"""double_ledge — cross a gap and land on a *raised* ledge.

The gap in front of the cliff eats into a single jump's height budget, so the
taller the leap the less height a single jump has left: a perfectly-timed single
jump (measured through the real physics) mounts a cliff fronted by a 5-tile gap
up to only ~136 px, and a 7-tile gap up to ~120 px — while the double jump
reaches ~264 px regardless of gap. Both rungs sit above the single ceiling, so
every instance genuinely needs the double (a single-jump-feasible instance would
just teach the GA to skip the skill). The difficulty knob is precision, not
reachability:

  - `DoubleLedge` (difficulty 2): 5-6 tile gap, ledge raised 160-184 px, wide
    landing — the gentle rung.
  - `DoubleLedgeHigh` (difficulty 3): 6-7 tile gap, ledge raised 188-220 px,
    narrow landing — demanding but inside the double-jump envelope.

Layout: approach ledge (base) | gap | cliff up to a raised ledge | step down to an
exit ledge (base). The cliff fronting the ledge is essential: without it a low
single-jump arc sails under the thin ledge and lands on the exit, skipping the
climb. Enters and exits at base level so streamed seams stay continuous; the gap
is a real hole (fall-death) the surrounding ledges frame.
"""

from __future__ import annotations

import pymunk

from ...abilities import Ability
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


def _seg(world, a, b, friction: float = 1.0) -> None:
    s = pymunk.Segment(world.space.static_body, a, b, 5)
    s.friction = friction
    world.space.add(s)


class _GapLedge(Chunk):
    """Shared build for the gap-onto-raised-ledge family. Subclasses set
    `difficulty` and `random_params`."""

    requires_ability = Ability.DOUBLE_JUMP

    def __init__(
        self,
        approach_tiles: int = 2,
        gap_tiles: int = 5,
        ledge_tiles: int = 4,
        exit_tiles: int = 2,
        height: int = 172,
    ) -> None:
        self.approach_tiles = approach_tiles
        self.gap_tiles = gap_tiles
        self.ledge_tiles = ledge_tiles
        self.exit_tiles = exit_tiles
        self.height = height

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        ax = self.approach_tiles * TILE
        gx = self.gap_tiles * TILE
        lx = self.ledge_tiles * TILE
        ex = self.exit_tiles * TILE
        top = base_y - self.height

        approach_r = x_offset + ax
        ledge_l = approach_r + gx
        ledge_r = ledge_l + lx
        exit_r = ledge_r + ex

        _seg(world, (x_offset, base_y), (approach_r, base_y))   # approach ledge
        # Cliff face fronting the ledge, full height base..top. Without it a low
        # single-jump arc sails under the thin ledge and lands on the exit beyond,
        # skipping the climb — so this wall is what forces the double jump.
        _seg(world, (ledge_l, base_y), (ledge_l, top))          # cliff face
        _seg(world, (ledge_l, top), (ledge_r, top))             # raised landing ledge
        _seg(world, (ledge_r, base_y), (exit_r, base_y))        # exit ledge (a step down)
        return ax + gx + lx + ex


@register_chunk("double_ledge")
class DoubleLedge(_GapLedge):
    difficulty: int = 2

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"gap_tiles": rng.randint(5, 6), "ledge_tiles": 4, "height": rng.randint(160, 184)}


@register_chunk("double_ledge_high")
class DoubleLedgeHigh(_GapLedge):
    difficulty: int = 3

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"gap_tiles": rng.randint(6, 7), "ledge_tiles": 3, "height": rng.randint(188, 220)}
