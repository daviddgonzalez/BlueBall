"""Pixel theme — the first concrete art style. Sprites are added in Task 4."""

from __future__ import annotations

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


def build():
    # Local imports avoid a circular import: theme.py registers this theme at
    # its own import time, so pixel.py must not import theme.py at module load.
    from types import MappingProxyType

    from ..theme import Theme

    return Theme(
        palette=MappingProxyType(dict(PALETTE)),
        sprites={
            "ball": _BALL,
            "spike": _SPIKE,
            "coin": _COIN,
            "collectible": _COIN,   # collectibles render as coins
            "goal": _GOAL,
        },
        params=MappingProxyType({
            "squash_max": 0.35, "particle_cap": 300,
        }),
    )
