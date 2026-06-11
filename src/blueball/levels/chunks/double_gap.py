"""double_gap — a flat gap too wide to clear with a single jump.

A perfectly-timed single jump clears a ~420 px equal-height gap (measured through
the real physics); a double jump clears ~720 px. This gap is 15-18 tiles
(480-576 px): past the single-jump reach, well inside the double-jump reach. Like
`Gap` it builds no geometry — it is a hole the surrounding flats frame — so
falling in is a fall-death. Gated to double-jump runs by `requires_ability`.
"""

from __future__ import annotations

from ...abilities import Ability
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("double_gap")
class DoubleGap(Chunk):
    difficulty: int = 3
    requires_ability = Ability.DOUBLE_JUMP

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(15, 18)}

    def __init__(self, width_tiles: int = 16) -> None:
        self.width_tiles = width_tiles

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        return self.width_tiles * TILE
