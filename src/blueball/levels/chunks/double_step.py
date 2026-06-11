"""double_step — a wall too tall to mount with a single jump.

A perfectly-timed single jump mounts a flush wall up to ~172 px (measured through
the real physics — the naive v^2/2g apex of ~103 px badly underestimates it, and
the GA finds the optimum). This wall rises 192-224 px, past that ceiling and
inside the ~260 px the double jump reaches. Pure vertical skill — no gap. The
player jumps up the wall face, double-jumps, and drifts onto the platform, then
steps back down to base for the seam.

Layout: approach ledge (base) | vertical wall up to a platform | step back down
to an exit ledge (base). Gated to double-jump runs by `requires_ability`.
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


@register_chunk("double_step")
class DoubleStep(Chunk):
    difficulty: int = 3
    requires_ability = Ability.DOUBLE_JUMP

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"height": rng.randint(192, 224)}

    def __init__(
        self,
        approach_tiles: int = 2,
        plat_tiles: int = 3,
        exit_tiles: int = 2,
        height: int = 208,
    ) -> None:
        self.approach_tiles = approach_tiles
        self.plat_tiles = plat_tiles
        self.exit_tiles = exit_tiles
        self.height = height

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        ax = self.approach_tiles * TILE
        pwx = self.plat_tiles * TILE
        ex = self.exit_tiles * TILE
        top = base_y - self.height

        wall_x = x_offset + ax
        plat_r = wall_x + pwx
        exit_r = plat_r + ex

        _seg(world, (x_offset, base_y), (wall_x, base_y))   # approach ledge
        _seg(world, (wall_x, base_y), (wall_x, top))        # the wall to mount
        _seg(world, (wall_x, top), (plat_r, top))           # platform on top
        _seg(world, (plat_r, base_y), (exit_r, base_y))     # exit ledge (a step down)
        return ax + pwx + ex
