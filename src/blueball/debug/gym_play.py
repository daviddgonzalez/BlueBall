"""Hands-on playable completion-gym segments, for tuning by feel.

Probes can tell you whether a *scripted* agent solves or cheeses a segment, but
not how it feels to a person. This boots a single gym segment in the normal
PlayScene (same human controls as `python main.py play`) so you can try to solve
it — and try to cheese it — yourself.

    python main.py play-gym box-lava               # default tuning
    python main.py play-gym box-lava --pit 22 --depth 72
    python main.py play-gym boost-gap --gap 28

box-lava is laid out exactly as BoxLavaSegment will be (boost pad removed, a
plain Flat spacer in its place so the pit edge stays at x=256). boost-gap is
Flat | boost_pad | lava_gap | goal — the boost is the only way across.
"""

from __future__ import annotations

from ..levels.chunks.flat import GROUND_Y

_BG = "#10131a"
_GROUND = "#a17a3a"
_SPAWN = [80, GROUND_Y - 30]


def _box_lava_level(pit_tiles: int = 18, depth: int = 72) -> dict:
    return {
        "name": f"Box-Lava  pit={pit_tiles}  depth={depth}",
        "background": _BG,
        "ground": _GROUND,
        "spawn": list(_SPAWN),
        "chunks": [
            {"type": "flat", "width_tiles": 2},
            {"type": "flat", "width_tiles": 3},  # plain spacer where the boost pad was
            {"type": "box_lava_gap", "pit_tiles": pit_tiles, "depth": depth},
            {"type": "goal", "width_tiles": 2},
        ],
    }


def _boost_gap_level(gap_tiles: int = 28) -> dict:
    # A long flat run-up BEFORE the pad: the player must NOT spawn on the pad,
    # and needs ~140px to reach full speed so the boosted leap is at full tilt.
    return {
        "name": f"Boost-Gap  gap={gap_tiles}",
        "background": _BG,
        "ground": _GROUND,
        "spawn": list(_SPAWN),
        "chunks": [
            {"type": "flat", "width_tiles": 8},
            {"type": "boost_pad", "width_tiles": 3, "multiplier": 2.0},
            {"type": "lava_gap", "pit_tiles": gap_tiles},
            {"type": "goal", "width_tiles": 2},
        ],
    }


_SEGMENTS = {
    "box-lava": _box_lava_level,
    "boost-gap": _boost_gap_level,
}


def build_level(name: str, **params) -> dict:
    """Return the level_data dict for a named segment (params override defaults).
    None-valued params are dropped so the builder defaults apply."""
    if name not in _SEGMENTS:
        raise SystemExit(f"unknown segment {name!r}; choose from {', '.join(_SEGMENTS)}")
    clean = {k: v for k, v in params.items() if v is not None}
    return _SEGMENTS[name](**clean)


def play_segment(name: str, **params) -> int:
    import pygame

    from .. import config
    from ..scenes.play import PlayScene

    level_data = build_level(name, **params)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption(f"Blue Ball — {level_data['name']}")
    clock = pygame.time.Clock()
    scene = PlayScene(screen, level_data=level_data)
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
    pygame.quit()
    return 0
