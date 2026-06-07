"""Theme — data the renderer reads to draw a given art style. Switchable via
config.ACTIVE_THEME. Themes register themselves at import time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .. import config

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ParallaxLayer:
    """A baked background strip + a scroll factor (0 = static, 1 = world-locked)."""
    sprite_key: str
    factor: float
    y: int = 0


@dataclass(frozen=True)
class Theme:
    palette: Mapping[str, Color]
    sprites: dict[str, object] = field(default_factory=dict)   # name -> SpriteDef (Task 4)
    parallax: list[ParallaxLayer] = field(default_factory=list)
    pixel_scale: int = config.PIXEL_SCALE
    params: Mapping[str, float | int] = field(default_factory=dict)


_REGISTRY: dict[str, Theme] = {}


def register_theme(name: str, theme: Theme) -> None:
    _REGISTRY[name] = theme


def get_theme(name: str) -> Theme:
    return _REGISTRY[name]


def get_active_theme() -> Theme:
    name = config.ACTIVE_THEME
    if name not in _REGISTRY:
        raise KeyError(
            f"Active theme {name!r} is not registered. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


# Import side effect: register built-in themes. Placed at the bottom so the
# names above (Theme, register_theme) already exist when pixel.py imports them.
from .themes import pixel as _pixel  # noqa: E402
register_theme("pixel", _pixel.build())
