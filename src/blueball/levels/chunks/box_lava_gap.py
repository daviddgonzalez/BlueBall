"""box_lava_gap chunk — a long, shallow lava pit crossed by shoving a box in
as a mid-pit stepping stone.

The pit is too long to clear in one jump. The player pushes the PushableBox off
the near ledge; it drops to the solid pit floor (lava is a player-only sensor,
so the box is unharmed) and, with momentum across the low-friction floor,
settles near the middle, turning one long jump into two short ones:
    near ledge -> box top -> far ledge.
"""

from __future__ import annotations

from ...entities.lava import Lava
from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y
from .lava_gap import build_lava_pit

_PIT_DEPTH = 72       # pit floor sits this far below the ledges (px)
_BOX_SIZE = 64        # box edge length (px) — large so it reads as a real plug/step
_BOX_MASS = 0.6
_LAVA_BELOW_BOX = 8   # lava surface sits this far below the resting box's top


@register_chunk("box_lava_gap")
class BoxLavaGap(Chunk):
    sampler_include: bool = True
    difficulty: int = 3

    def __init__(
        self,
        approach_tiles: int = 3,  # wide enough to seat the large box on the near ledge
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
        pit_left, pit_right, floor_y, total = build_lava_pit(
            world, x_offset, base_y, self.approach_tiles, self.pit_tiles,
            self.exit_tiles, self.depth)
        px = self.pit_tiles * TILE
        box_rest_top = floor_y - self.box_size
        lava_surface = box_rest_top + _LAVA_BELOW_BOX
        world.add_entity(Lava(
            world, position=(pit_left + px / 2, lava_surface),
            width=px, rise_speed=0.0, height=self.depth))
        world.add_entity(PushableBox(
            world,
            position=(pit_left - self.box_size / 2 - 2, base_y - self.box_size / 2 - 1),
            size=self.box_size, mass=self.box_mass))
        return total
