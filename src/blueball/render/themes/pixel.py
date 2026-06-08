"""Pixel theme — the first concrete art style. Sprites are added in Task 4."""

from __future__ import annotations

import math as _math

from ..sprites import SpriteDef

PALETTE = {
    "ball": (58, 138, 255),
    "ball_hi": (191, 224, 255),
    "ground": (63, 154, 85),
    "ground_top": (111, 217, 138),
    "ground_edge": (40, 90, 50),
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
    # --- per-entity sprite palettes (Task 9) ---
    "patroller": (220, 100, 60),
    "patroller_hi": (255, 160, 110),
    "falling_hazard": (200, 70, 55),
    "falling_hazard_hi": (255, 130, 110),
    "platform": (120, 200, 120),
    "platform_hi": (170, 240, 170),
    "spring": (170, 170, 220),
    "spring_hi": (220, 220, 255),
    "checkpoint": (90, 220, 140),
    "checkpoint_hi": (200, 255, 220),
    "checkpoint_active": (255, 220, 80),
    "checkpoint_active_hi": (255, 248, 180),
    "crumbling": (180, 140, 100),
    "crumbling_hi": (220, 185, 140),
    "key": (255, 200, 60),
    "key_hi": (255, 240, 160),
    "door": (160, 100, 40),
    "door_hi": (220, 160, 80),
    "box": (160, 120, 80),
    "box_hi": (210, 170, 120),
    "swing_hazard": (210, 90, 90),
    "swing_hazard_hi": (255, 150, 150),
    "one_way": (80, 200, 160),
    "one_way_hi": (160, 245, 215),
    "charger": (200, 80, 80),
    "charger_hi": (255, 160, 160),
    "charger_charge": (255, 120, 120),
    "charger_charge_hi": (255, 210, 210),
    "lava": (240, 90, 30),
    "lava_hi": (255, 220, 100),
    "projectile": (255, 140, 40),
    "projectile_hi": (255, 235, 160),
    "cannon": (90, 90, 110),
    "cannon_hi": (150, 150, 180),
    "ability": (220, 220, 220),
    "ability_hi": (255, 255, 255),
    "ability_double_jump": (255, 220, 80),
    "ability_double_jump_hi": (255, 248, 190),
    "boost_pad": (80, 220, 240),
    "boost_pad_hi": (30, 150, 180),
    # HUD text
    "hud": (255, 255, 255),
    "hud_best": (255, 220, 80),
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


# ---------------------------------------------------------------------- #
# Per-entity sprites (Task 9). Rough but recognizable starters; aesthetic  #
# polish happens interactively later. 2-color format: lowercase=base,      #
# UPPERCASE=<key>_hi.                                                       #
# ---------------------------------------------------------------------- #

# Patroller: stout creature with two eyes (highlight = lit face/eyes).
_PATROLLER = SpriteDef(grid=[
    "..pppppppppp..",
    ".pppppppppppp.",
    "pppppppppppppp",
    "ppPPppppppPPpp",
    "ppPPppppppPPpp",
    "pppppppppppppp",
    "pppppppppppppp",
    "ppPPPPPPPPPPpp",
    "pppppppppppppp",
    "pppppppppppppp",
    ".pppppppppppp.",
    "p.pp.pp.pp.pp.",
], palette_key="patroller")

# Falling hazard: red anvil (flat top, narrow waist, flared foot).
_FALLING_HAZARD = SpriteDef(grid=[
    "FFFFFFFFFFFFF",
    "fffffffffffff",
    "fffffffffffff",
    ".fffffffffff.",
    "...fffffff...",
    ".....fff.....",
    ".....fff.....",
    "....fffff....",
    "..fffffffff..",
    "fffffffffffff",
], palette_key="falling_hazard")

# Horizontal moving platform: 32x8, scaled to length in renderer.
_PLATFORM = SpriteDef(grid=[
    "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP",
    "pppppppppppppppppppppppppppppppp",
    "pppppppppppppppppppppppppppppppp",
    "pppppppppppppppppppppppppppppppp",
], palette_key="platform")

# Spring pad: coiled base + plate (highlight = top plate).
_SPRING = SpriteDef(grid=[
    "SSSSSSSSSSSS",
    "SSSSSSSSSSSS",
    ".s.s.s.s.s..",
    "s.s.s.s.s.s.",
    ".s.s.s.s.s..",
    "ssssssssssss",
], palette_key="spring")

# Checkpoint: flag on a pole (two state-colored variants share this grid).
_CHECKPOINT = SpriteDef(grid=[
    "cCCCCCC...",
    "cCCCCCCC..",
    "cCCCCCCCC.",
    "cCCCCCCC..",
    "cCCCCCC...",
    "cc........",
    "cc........",
    "cc........",
    "cc........",
    "cc........",
    "ccc......c",
    "cccccccccc",
], palette_key="checkpoint")
_CHECKPOINT_ACTIVE = SpriteDef(_CHECKPOINT._grids[0], palette_key="checkpoint_active")

# Crumbling platform: cracked block (highlight = top crust).
_CRUMBLING = SpriteDef(grid=[
    "CCCCCCCCCCCCCCCC",
    "cccccccccccccccc",
    "cccc.cccccc.cccc",
    "ccc..ccc..c..ccc",
], palette_key="crumbling")

# Key: classic key with bow + bit (highlight = shine).
_KEY = SpriteDef(grid=[
    ".KKK....",
    "kKkKk...",
    "kK.Kk...",
    "kKkKk...",
    ".kkk....",
    "..kk....",
    "..kkk.k.",
    "..kkkkk.",
], palette_key="key")

# Door: paneled vertical door (16 tall, highlight = frame/handle).
_DOOR = SpriteDef(grid=[
    "DDDDDDDD",
    "dddddddd",
    "dDDDDDDd",
    "dDddddDd",
    "dDddddDd",
    "dDddDdDd",
    "dDddDdDd",
    "dDddddDd",
    "dDddddDd",
    "dDDDDDDd",
    "dddddddd",
    "dDDDDDDd",
    "dDddddDd",
    "dDddddDd",
    "dDDDDDDd",
    "dddddddd",
], palette_key="door")
# Door open: thin frame only (passable).
_DOOR_OPEN = SpriteDef(grid=[
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dd....dd",
    "dddddddd",
], palette_key="door")

# Pushable box: wooden crate (highlight = bevel/cross-brace).
_BOX = SpriteDef(grid=[
    "BBBBBBBBBBBB",
    "BbbbbbbbbbbB",
    "BbBbbbbbBbbB",
    "BbbBbbbBbbbB",
    "BbbbBbBbbbbB",
    "BbbbbBbbbbbB",
    "BbbbBbBbbbbB",
    "BbbBbbbBbbbB",
    "BbBbbbbbBbbB",
    "BbbbbbbbbbbB",
    "BbbbbbbbbbbB",
    "BBBBBBBBBBBB",
], palette_key="box")

# Swinging hazard bob: spiked ball.
_SWING_HAZARD = SpriteDef(grid=[
    "...s..s..s...",
    "..sssssssss..",
    ".sssSSsssss..",
    "ssssSSsssssss",
    "sssssssssssss",
    "ssssssssssss.",
    ".sssssssss...",
    "..s..s..s....",
], palette_key="swing_hazard")

# One-way platform: thin strip with down chevrons (highlight = surface).
_ONE_WAY = SpriteDef(grid=[
    "OOOOOOOOOOOOOOOO",
    "oooooooooooooooo",
    "o..oo..oo..oo..o",
    "...oo...oo...oo.",
], palette_key="one_way")

# Charger: angular dasher with a forward-pointing snout (highlight = eye).
_CHARGER = SpriteDef(grid=[
    "..hhhhhhhh..",
    ".hhhhhhhhhh.",
    "hhhhhhhhhhhh",
    "hhHHhhhhhhhh",
    "hhHHhhhhhhhh",
    "hhhhhhhhhhhh",
    "hhhhhhhhhhhh",
    "hhhhhhhhhhh.",
    ".hhhhhhhh...",
    "..hhhhhh....",
], palette_key="charger")
_CHARGER_CHARGE = SpriteDef(_CHARGER._grids[0], palette_key="charger_charge")

# Lava: molten surface block. 16x12, scaled to width/height. Top row = bright crust.
_LAVA = SpriteDef(grid=[
    "LLLLLLLLLLLLLLLL",
    "lLllLllLllLllLll",
    "llllllllllllllll",
    "llllllllllllllll",
    "llllllllllllllll",
    "llllllllllllllll",
    "llllllllllllllll",
    "llllllllllllllll",
], palette_key="lava")

# Projectile: fiery orb with bright core.
_PROJECTILE = SpriteDef(grid=[
    ".pppp.",
    "ppPPpp",
    "pPPPPp",
    "pPPPPp",
    "ppPPpp",
    ".pppp.",
], palette_key="projectile")

# Cannon: barrel pointing right (highlight = rim/muzzle). Renderer flips for left.
_CANNON = SpriteDef(grid=[
    "..cccccc...",
    ".cccccccCC.",
    "ccccccccCCC",
    "ccccccccCCC",
    "ccccccccCCC",
    "ccccccccCCC",
    "ccccccccCCC",
    ".cccccccCC.",
    "..cccccc...",
], palette_key="cannon")

# Ability pickup: diamond gem (two palette variants by ability).
_ABILITY = SpriteDef(grid=[
    "....aa....",
    "...aAAa...",
    "..aAAAAa..",
    ".aAAAAAAa.",
    "aAAAAAAAAa",
    ".aAAAAAAa.",
    "..aAAAAa..",
    "...aAAa...",
    "....aa....",
], palette_key="ability")
_ABILITY_DJ = SpriteDef(_ABILITY._grids[0], palette_key="ability_double_jump")

_BOOST_PAD = SpriteDef(grid=[
    "bbbbbbbbbbbbbbbb",
    "bBbbbBbbbBbbbBbb",
    "bbBbbbBbbbBbbbBb",
    "bBbbbBbbbBbbbBbb",
    "bbbbbbbbbbbbbbbb",
], palette_key="boost_pad")


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
            # --- Task 9 per-entity sprites ---
            "patroller": _PATROLLER,
            "falling_hazard": _FALLING_HAZARD,
            "platform": _PLATFORM,
            "spring": _SPRING,
            "checkpoint": _CHECKPOINT,
            "checkpoint_active": _CHECKPOINT_ACTIVE,
            "crumbling": _CRUMBLING,
            "key": _KEY,
            "door": _DOOR,
            "door_open": _DOOR_OPEN,
            "box": _BOX,
            "swing_hazard": _SWING_HAZARD,
            "one_way": _ONE_WAY,
            "charger": _CHARGER,
            "charger_charge": _CHARGER_CHARGE,
            "lava": _LAVA,
            "projectile": _PROJECTILE,
            "cannon": _CANNON,
            "ability": _ABILITY,
            "ability_double_jump": _ABILITY_DJ,
            "boost_pad": _BOOST_PAD,
        },
        parallax=[
            ParallaxLayer("hills_far", 0.3, y=150),
            ParallaxLayer("hills_near", 0.55, y=210),
        ],
        params=MappingProxyType({
            "squash_max": 0.35, "particle_cap": 300,
        }),
    )
