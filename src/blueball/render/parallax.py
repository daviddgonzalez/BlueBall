"""Parallax background: tiled strips scrolled by a per-layer factor."""

from __future__ import annotations


def layer_offset(camera_x: float, factor: float, tile_w: int) -> float:
    """Horizontal blit start offset, wrapped into (-tile_w, 0]."""
    return -((camera_x * factor) % tile_w)
