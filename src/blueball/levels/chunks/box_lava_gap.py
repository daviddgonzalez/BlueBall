"""box_lava_gap chunk — a long, shallow lava pit crossed by shoving a box in
as a mid-pit stepping stone.

The pit is too long to clear in one jump. The player pushes the PushableBox off
the near ledge; it drops to the solid pit floor (lava is a player-only sensor,
so the box is unharmed) and, with momentum across the low-friction floor,
settles near the middle, turning one long jump into two short ones:
    near ledge -> box top -> far ledge.
"""

from __future__ import annotations

import pymunk

from ...entities.lava import Lava
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

_PIT_DEPTH = 72       # pit floor sits this far below the ledges (px)
_BOX_SIZE = 40        # box edge length (px)
_BOX_MASS = 0.6
_LAVA_BELOW_BOX = 8   # lava surface sits this far below the resting box's top
_PIT_FLOOR_FRICTION = 0.1  # low friction so a firm push carries the box to mid


@register_chunk("box_lava_gap")
class BoxLavaGap(Chunk):
    sampler_include: bool = True
    difficulty: int = 3

    def __init__(
        self,
        approach_tiles: int = 2,
        pit_tiles: int = 6,
        exit_tiles: int = 2,
        depth: int = _PIT_DEPTH,
        box_size: int = _BOX_SIZE,
        box_mass: float = _BOX_MASS,
    ) -> None:
        self.approach_tiles = approach_tiles
        self.pit_tiles = pit_tiles
        self.exit_tiles = exit_tiles
        self.depth = depth
        self.box_size = box_size
        self.box_mass = box_mass

    @classmethod
    def random_params(cls, rng) -> dict:
        # Vary only the pit length (the difficulty knob); keep depth/box fixed
        # so the box-as-step geometry stays solvable.
        return {"pit_tiles": rng.randint(5, 7)}

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        ax = self.approach_tiles * TILE
        px = self.pit_tiles * TILE
        ex = self.exit_tiles * TILE
        total = ax + px + ex
        pit_left = x_offset + ax
        pit_right = pit_left + px
        floor_y = base_y + self.depth

        def seg(a, b, friction=1.0):
            s = pymunk.Segment(world.space.static_body, a, b, 5)
            s.friction = friction
            world.space.add(s)

        seg((x_offset, base_y), (pit_left, base_y))           # approach ledge
        seg((pit_right, base_y), (x_offset + total, base_y))  # exit ledge
        seg((pit_left, base_y), (pit_left, floor_y))          # near wall
        seg((pit_right, base_y), (pit_right, floor_y))        # far wall
        seg((pit_left, floor_y), (pit_right, floor_y), _PIT_FLOOR_FRICTION)  # floor

        box_rest_top = floor_y - self.box_size
        lava_surface = box_rest_top + _LAVA_BELOW_BOX
        world.add_entity(Lava(
            world,
            position=(pit_left + px / 2, lava_surface),
            width=px,
            rise_speed=0.0,
            height=self.depth,
        ))

        # Box starts on the approach ledge at the pit edge, ready to shove in.
        world.add_entity(PushableBox(
            world,
            position=(pit_left - self.box_size / 2 - 2, base_y - self.box_size / 2 - 1),
            size=self.box_size,
            mass=self.box_mass,
        ))
        return total
