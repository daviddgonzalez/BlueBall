"""StairsUp / StairsDown — staircase of N tiles, each `step_height` tall."""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


def _add_step(world, x0, x1, top_y):
    seg = pymunk.Segment(world.space.static_body, (x0, top_y), (x1, top_y), 5)
    seg.friction = 1.0
    world.space.add(seg)


@register_chunk("stairs_up")
class StairsUp(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"steps": rng.randint(2, 4), "step_height": rng.choice([24, 32, 40])}

    def __init__(self, steps: int = 3, step_height: int = 32) -> None:
        self.steps = steps
        self.step_height = step_height

    def build(self, world, x_offset: float) -> float:
        for i in range(self.steps):
            x0 = x_offset + i * TILE
            x1 = x0 + TILE
            top_y = GROUND_Y - (i + 1) * self.step_height
            _add_step(world, x0, x1, top_y)
            # Vertical riser
            seg = pymunk.Segment(world.space.static_body, (x0, top_y), (x0, top_y + self.step_height), 5)
            world.space.add(seg)
        return self.steps * TILE


@register_chunk("stairs_down")
class StairsDown(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"steps": rng.randint(2, 4), "step_height": rng.choice([24, 32, 40])}

    def __init__(self, steps: int = 3, step_height: int = 32) -> None:
        self.steps = steps
        self.step_height = step_height

    def build(self, world, x_offset: float) -> float:
        for i in range(self.steps):
            x0 = x_offset + i * TILE
            x1 = x0 + TILE
            top_y = GROUND_Y - (self.steps - i) * self.step_height
            _add_step(world, x0, x1, top_y)
            seg = pymunk.Segment(world.space.static_body, (x1, top_y), (x1, top_y + self.step_height), 5)
            world.space.add(seg)
        return self.steps * TILE
