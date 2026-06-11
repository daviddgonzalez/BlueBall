"""lava_gap chunk — a boxless lava pit too wide to clear without a boost.

Shares its pit geometry with box_lava_gap via build_lava_pit().
"""
from __future__ import annotations

import pymunk

from ...entities.lava import Lava
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

_DEPTH = 72
_PIT_FLOOR_FRICTION = 0.1  # low friction so a firm push carries the box across the pit to mid


def build_lava_pit(world, x_offset, base_y, approach_tiles, pit_tiles,
                   exit_tiles, depth, floor_friction=_PIT_FLOOR_FRICTION):
    """Build approach/exit ledges, the two pit walls, and a low-friction floor.
    Returns (pit_left, pit_right, floor_y, total_width_px). No lava, no box."""
    ax = approach_tiles * TILE
    px = pit_tiles * TILE
    ex = exit_tiles * TILE
    total = ax + px + ex
    pit_left = x_offset + ax
    pit_right = pit_left + px
    floor_y = base_y + depth

    def seg(a, b, friction=1.0):
        s = pymunk.Segment(world.space.static_body, a, b, 5)
        s.friction = friction
        world.space.add(s)

    seg((x_offset, base_y), (pit_left, base_y))            # approach ledge
    seg((pit_right, base_y), (x_offset + total, base_y))   # exit ledge
    seg((pit_left, base_y), (pit_left, floor_y))           # near wall
    seg((pit_right, base_y), (pit_right, floor_y))         # far wall
    seg((pit_left, floor_y), (pit_right, floor_y), floor_friction)
    return pit_left, pit_right, floor_y, total


@register_chunk("lava_gap")
class LavaGapChunk(Chunk):
    sampler_include: bool = False  # only used inside BoostGapSegment
    difficulty: int = 3

    def __init__(self, approach_tiles: int = 2, pit_tiles: int = 26,
                 exit_tiles: int = 2, depth: int = _DEPTH) -> None:
        self.approach_tiles = approach_tiles
        self.pit_tiles = pit_tiles
        self.exit_tiles = exit_tiles
        self.depth = depth

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        pit_left, pit_right, floor_y, total = build_lava_pit(
            world, x_offset, base_y, self.approach_tiles, self.pit_tiles,
            self.exit_tiles, self.depth)
        px = self.pit_tiles * TILE
        # Lava fills the pit from ledge level down: any fall is lethal.
        world.add_entity(Lava(
            world,
            position=((pit_left + pit_right) / 2, base_y + self.depth / 2),
            width=px,
            rise_speed=0.0,
            height=self.depth,
        ))
        return total
