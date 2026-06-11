"""Hands-on playable showcase of the double-jump chunk family, for tuning by feel.

The four double-jump chunks are gated out of normal Infinite Run unless the run
is granted DOUBLE_JUMP, so you can't stumble onto them in `python main.py play`.
This boots a hand-built level that chains all four — with double jump
force-granted — in the normal PlayScene (same controls as `play`) so you can try
each one yourself and see whether the gap widths / ledge heights feel right.

    python main.py play-doublejump

Order is easy → hard: the gentle ledge first, then the demanding ledge, the wall
mount, and the wide fall-death gap. Heights are set past the *measured* single-
jump ceiling so each genuinely needs the double. The flats between obstacles are
long (12 tiles) on purpose: a single jump reaches ~420 px, so a shorter flat
would let you hop straight from one raised platform to the next at altitude,
skipping the climb — the bypass found in playtest.
"""

from __future__ import annotations

from ..abilities import Ability
from ..levels.chunks.flat import GROUND_Y

_BG = "#10131a"
_GROUND = "#5a6a7a"
_SPAWN = [80, GROUND_Y - 30]


def build_showcase_level() -> dict:
    """Level data chaining all four double-jump chunks with flat run-ups and a
    goal. Dimensions are pinned to representative hard instances (not random) so
    the showcase is reproducible."""
    return {
        "name": "Double-Jump Showcase",
        "background": _BG,
        "ground": _GROUND,
        "spawn": list(_SPAWN),
        "chunks": [
            {"type": "flat", "width_tiles": 8},
            # Gentle rung: a 5-tile gap caps the single-jump cliff mount at ~136px,
            # so a 176px ledge needs the double.
            {"type": "double_ledge", "gap_tiles": 5, "height": 176},
            {"type": "flat", "width_tiles": 12},
            # Demanding: wider gap, taller ledge, narrower landing.
            {"type": "double_ledge_high", "gap_tiles": 7, "height": 208},
            {"type": "flat", "width_tiles": 12},
            # Wall mount — flush, single-jump ceiling ~172px, so 208px needs two.
            {"type": "double_step", "height": 208},
            {"type": "flat", "width_tiles": 12},
            # Wide fall-death gap — past the ~420px single-jump reach.
            {"type": "double_gap", "width_tiles": 17},
            {"type": "flat", "width_tiles": 4},
            {"type": "goal", "width_tiles": 2},
        ],
    }


def play_showcase() -> int:
    import pygame

    from .. import config
    from ..scenes.play import PlayScene

    level_data = build_showcase_level()

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption(f"Blue Ball — {level_data['name']}")
    clock = pygame.time.Clock()
    scene = PlayScene(
        screen, level_data=level_data, extra_abilities={Ability.DOUBLE_JUMP}
    )
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
    pygame.quit()
    return 0
