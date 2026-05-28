"""StairsUp / StairsDown — a staircase of N tiles, each `step_height` tall.

Hand-authored levels get the classic blocky steps (with vertical risers). The
Infinite Run sampler sets ``rounded=True`` (via random_params), replacing the
steps with one smoothstep ramp so a rolling ball flows instead of jolting.
"""

from __future__ import annotations

import math

import pymunk

from ... import config
from ._curve import smoothstep_ramp
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y

# Max average slope (rise per horizontal px) of a rounded ramp. A rounded ramp
# is widened beyond its nominal tile width if needed to stay this gentle, so:
#  (a) the ball never wedges in a valley between two ramps on a surface too
#      steep to count as "grounded" (a softlock), and
#  (b) the ball stays grounded on the *steepest* part of a ramp, so it can
#      recharge and perform its jump anywhere on the ramp.
# A smoothstep ramp's peak slope is 1.5x its average, so we cap the average at
# tan(tolerance - margin) / 1.5 to keep the steepest point within the player's
# grounded tolerance (with a few degrees of margin for contact/discretization).
_RAMP_MARGIN_DEG = 3.0
_MAX_RAMP_SLOPE = math.tan(
    math.radians(config.GROUNDED_NORMAL_TOLERANCE_DEG - _RAMP_MARGIN_DEG)
) / 1.5


def _rounded_ramp_width(steps: int, rise: int) -> int:
    """Horizontal run for a rounded ramp: at least the nominal tile width, but
    wide enough that the average slope stays within _MAX_RAMP_SLOPE."""
    return max(steps * TILE, math.ceil(rise / _MAX_RAMP_SLOPE))


def _add_step(world, x0, x1, top_y):
    seg = pymunk.Segment(world.space.static_body, (x0, top_y), (x1, top_y), 5)
    seg.friction = 1.0
    world.space.add(seg)


@register_chunk("stairs_up")
class StairsUp(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"steps": rng.randint(2, 4), "step_height": rng.choice([24, 32, 40]), "rounded": True}

    def __init__(self, steps: int = 3, step_height: int = 32, rounded: bool = False) -> None:
        self.steps = steps
        self.step_height = step_height
        self.rounded = rounded
        # Climbs: left edge at base, right edge `rise` higher (up = -y).
        self.entry_dy = 0.0
        self.exit_dy = -(steps * step_height)

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        if self.rounded:
            rise = self.steps * self.step_height
            width = _rounded_ramp_width(self.steps, rise)
            smoothstep_ramp(world, x_offset, base_y, x_offset + width, base_y - rise)
            return width
        width = self.steps * TILE
        for i in range(self.steps):
            x0 = x_offset + i * TILE
            x1 = x0 + TILE
            top_y = base_y - (i + 1) * self.step_height
            _add_step(world, x0, x1, top_y)
            # Vertical riser
            seg = pymunk.Segment(world.space.static_body, (x0, top_y), (x0, top_y + self.step_height), 5)
            world.space.add(seg)
        return width


@register_chunk("stairs_down")
class StairsDown(Chunk):
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"steps": rng.randint(2, 4), "step_height": rng.choice([24, 32, 40]), "rounded": True}

    def __init__(self, steps: int = 3, step_height: int = 32, rounded: bool = False) -> None:
        self.steps = steps
        self.step_height = step_height
        self.rounded = rounded
        # Descends: left edge `rise` higher than base, right edge at base.
        self.entry_dy = -(steps * step_height)
        self.exit_dy = 0.0

    def build(self, world, x_offset: float, base_y: float = GROUND_Y) -> float:
        if self.rounded:
            rise = self.steps * self.step_height
            width = _rounded_ramp_width(self.steps, rise)
            smoothstep_ramp(world, x_offset, base_y - rise, x_offset + width, base_y)
            return width
        width = self.steps * TILE
        for i in range(self.steps):
            x0 = x_offset + i * TILE
            x1 = x0 + TILE
            top_y = base_y - (self.steps - i) * self.step_height
            _add_step(world, x0, x1, top_y)
            seg = pymunk.Segment(world.space.static_body, (x1, top_y), (x1, top_y + self.step_height), 5)
            world.space.add(seg)
        return width
