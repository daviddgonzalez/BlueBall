"""Pixel theme — the first concrete art style. Sprites are added in Task 4."""

from __future__ import annotations

import math as _math

from ..sprites import SpriteDef

PALETTE = {
    "ball": (58, 138, 255),
    "ball_hi": (191, 224, 255),
    "ground": (63, 154, 85),
    "ground_top": (111, 217, 138),
    "spike": (226, 80, 63),
    "spike_hi": (255, 140, 110),
    "coin": (255, 210, 58),
    "coin_hi": (255, 244, 170),
    "sky_top": (207, 234, 255),
    "sky_bottom": (126, 199, 255),
    "hills_far": (96, 150, 120),
    "hills_near": (66, 124, 92),
    "goal": (220, 90, 80),
    "goal_hi": (255, 232, 130),
}

_BALL = SpriteDef(grid=[
    ".....bbbbbb.....",
    "...bbbbbbbbbb...",
    "..bbBBBbbbbbbb..",
    ".bbBBBBbbbbbbbb.",
    ".bbBBbbbbbbbbbb.",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbbbb",
    ".bbbbbbbbbbbbbb.",
    ".bbbbbbbbbbbbbb.",
    "..bbbbbbbbbbbb..",
    "...bbbbbbbbbb...",
    ".....bbbbbb.....",
], palette_key="ball")

_SPIKE = SpriteDef(grid=[
    ".......ss.......",
    "......ssss......",
    ".....ssssss.....",
    "....ssSSssss....",
    "...ssSSssssss...",
    "..ssssssssssss..",
    ".ssssssssssssss.",
    "ssssssssssssssss",
], palette_key="spike")

_COIN = SpriteDef(grid=[
    "..cccc..",
    ".cCCccc.",
    "cCCccccc",
    "cCcccccc",
    "cccccccc",
    "cccccccc",
    ".cccccc.",
    "..cccc..",
], palette_key="coin")

_GOAL = SpriteDef(grid=[
    "gg..........",
    "gg..........",
    "ggGGGGGG....",
    "ggGGGGGGGG..",
    "ggGGGGGGGGGg",
    "ggGGGGGGGGGg",
    "ggGGGGGGGG..",
    "ggGGGGGG....",
    "gg..........",
    "gg..........",
    "gg..........",
    "gg..........",
    "gg..........",
    "gg..........",
    "gg..........",
    "gg..........",
], palette_key="goal")


def _hill_strip(width: int, height: int, period: int, amp: int) -> list[str]:
    surface_y = [round(amp * (0.5 + 0.5 * _math.sin(2 * _math.pi * x / period)))
                 for x in range(width)]
    return ["".join("h" if y >= surface_y[x] else "." for x in range(width))
            for y in range(height)]


_HILLS_FAR = SpriteDef(_hill_strip(160, 110, period=80, amp=10), palette_key="hills_far")
_HILLS_NEAR = SpriteDef(_hill_strip(128, 130, period=48, amp=16), palette_key="hills_near")


def build():
    # Local imports avoid a circular import: theme.py registers this theme at
    # its own import time, so pixel.py must not import theme.py at module load.
    from types import MappingProxyType

    from ..theme import Theme, ParallaxLayer

    return Theme(
        palette=MappingProxyType(dict(PALETTE)),
        sprites={
            "ball": _BALL,
            "spike": _SPIKE,
            "coin": _COIN,
            "collectible": _COIN,   # collectibles render as coins
            "goal": _GOAL,
            "hills_far": _HILLS_FAR,
            "hills_near": _HILLS_NEAR,
        },
        parallax=[
            ParallaxLayer("hills_far", 0.3, y=150),
            ParallaxLayer("hills_near", 0.55, y=210),
        ],
        params=MappingProxyType({
            "squash_max": 0.35, "particle_cap": 300,
        }),
    )
