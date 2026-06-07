"""Pixel theme — the first concrete art style. Sprites are added in Task 4."""

from __future__ import annotations

PALETTE = {
    "ball": (58, 138, 255),
    "ball_hi": (191, 224, 255),
    "ground": (63, 154, 85),
    "ground_top": (111, 217, 138),
    "spike": (226, 80, 63),
    "coin": (255, 210, 58),
    "sky_top": (207, 234, 255),
    "sky_bottom": (126, 199, 255),
}


def build():
    # Local imports avoid a circular import: theme.py registers this theme at
    # its own import time, so pixel.py must not import theme.py at module load.
    from types import MappingProxyType

    from ..theme import Theme

    return Theme(
        palette=MappingProxyType(dict(PALETTE)),
        params=MappingProxyType({
            "squash_max": 0.35, "shake_decay": 8.0, "particle_cap": 300,
        }),
    )
