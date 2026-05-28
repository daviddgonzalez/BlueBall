"""VerticalColumn — N stacked floating platforms inside the chunk's horizontal
slot. By default platforms alternate left/right hugs so the player must zig-zag
jump. An optional ``pattern`` overrides that with explicit per-layer horizontal
placements (and lets a layer carry more than one platform for route choice).
"""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("vertical_column")
class VerticalColumn(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 6,
        steps: int = 5,
        step_height: int = 80,
        bottom_offset: int = 96,
        platform_tiles: int = 2,
        pattern: list | None = None,
    ) -> None:
        self.width_tiles = width_tiles
        self.steps = steps
        self.step_height = step_height
        self.bottom_offset = bottom_offset
        self.platform_tiles = platform_tiles
        # Optional layout. Each entry maps to one layer (cycled if shorter than
        # `steps`) and is either a single tile-x offset or a list of them for
        # multiple platforms on that layer. None → legacy left/right alternation.
        self.pattern = pattern

    @classmethod
    def random_params(cls, rng) -> dict:
        params = {
            "width_tiles": 6,
            "steps": rng.randint(3, 6),
            "step_height": rng.choice([64, 80, 96]),
            "bottom_offset": 96,
            "platform_tiles": 2,
        }
        # Half the time, hand the column a varied multi-platform pattern instead
        # of plain left/right alternation — gives route choice and breaks the
        # monotony of identical zig-zag climbs in Infinite Run.
        if rng.random() < 0.5:
            params["pattern"] = rng.choice([
                [0, 4, 2, 4],
                [0, 3, 1, 4],
                [4, 0, [1, 4], 2],
                [2, 0, 4, [0, 4]],
                [0, 4, [0, 4], 2],
            ])
        return params

    def _layer_positions(self, i: int) -> list[int]:
        """Tile-x offsets of every platform on layer `i`."""
        if self.pattern is None:
            return [0 if i % 2 == 0 else self.width_tiles - self.platform_tiles]
        entry = self.pattern[i % len(self.pattern)]
        return entry if isinstance(entry, list) else [entry]

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        slot_w = self.width_tiles * TILE
        plat_w = self.platform_tiles * TILE
        max_tx = self.width_tiles - self.platform_tiles
        for i in range(self.steps):
            y = base_y - (self.bottom_offset + i * self.step_height)
            for tx in self._layer_positions(i):
                tx = max(0, min(max_tx, tx))
                ax = x_offset + tx * TILE
                seg = pymunk.Segment(world.space.static_body, (ax, y), (ax + plat_w, y), 5)
                seg.friction = 1.0
                world.space.add(seg)
        return slot_w
