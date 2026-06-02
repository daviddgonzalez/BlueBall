"""Shared smoothstep polyline helper for rounded terrain.

pymunk has no native curved shapes, so a "curve" is a chain of short straight
segments. The smoothstep profile s(t) = t^2 (3 - 2t) has zero slope at both
ends, so a ramp blends seamlessly into adjacent flat ground — and two ramps
sharing an endpoint (e.g. up then down) meet with a rounded, jolt-free peak.
"""

from __future__ import annotations

import pymunk

# Target horizontal length of each approximating segment, in world px.
_PX_PER_SEGMENT = 8.0


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def smoothstep_ramp(
    world,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    friction: float = 1.0,
) -> None:
    """Add static segments from (x0, y0) to (x1, y1) following a smoothstep
    profile in y. x advances linearly; y eases so the slope is flat at both
    endpoints."""
    width = abs(x1 - x0)
    n = max(2, round(width / _PX_PER_SEGMENT))
    prev = (x0, y0)
    for k in range(1, n + 1):
        t = k / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * _smoothstep(t)
        seg = pymunk.Segment(world.space.static_body, prev, (x, y), 5)
        seg.friction = friction
        world.space.add(seg)
        prev = (x, y)
