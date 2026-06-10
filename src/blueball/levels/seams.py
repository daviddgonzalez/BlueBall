"""Weld connected ground segments so a fast ball doesn't catch on the rounded
endcaps where two collinear `pymunk.Segment`s meet — the "seam hop".

Each chunk lays its own ground segment(s); at a joint the two radius-5 endcaps
form a tiny ridge that flicks a fast ball airborne (measured ~3px at top speed).
`Segment.set_neighbors` tells a segment the surface continues smoothly into its
neighbour, which suppresses the spurious endcap contact. We only weld
near-collinear continuations, never real corners (e.g. a floor meeting a pit
wall), so edges still behave like edges.
"""

from __future__ import annotations

import pymunk

# cos of the max angle between two segments still treated as a smooth
# continuation rather than a corner. cos(45°) ≈ 0.707.
_COLLINEAR_COS = 0.7


def _dir(seg: pymunk.Segment):
    d = seg.b - seg.a
    return d.normalized() if d.length > 1e-9 else None


def weld_ground_seams(space: pymunk.Space, eps: float = 1.0) -> int:
    """Set smooth neighbours on every pair of near-collinear static segments
    that share an endpoint. Returns the number of segments touched. Idempotent."""
    segs = [s for s in space.shapes
            if isinstance(s, pymunk.Segment)
            and s.body.body_type == pymunk.Body.STATIC]
    touched = 0
    for s in segs:
        s_dir = _dir(s)
        prev, nxt = s.a, s.b
        changed = False
        for o in segs:
            if o is s:
                continue
            o_dir = _dir(o)
            if s_dir is None or o_dir is None:
                continue
            if abs(s_dir.dot(o_dir)) < _COLLINEAR_COS:
                continue  # a real corner — leave the edge intact
            if (o.a - s.a).length < eps:
                prev, changed = o.b, True
            elif (o.b - s.a).length < eps:
                prev, changed = o.a, True
            if (o.a - s.b).length < eps:
                nxt, changed = o.b, True
            elif (o.b - s.b).length < eps:
                nxt, changed = o.a, True
        s.set_neighbors(prev, nxt)
        if changed:
            touched += 1
    return touched
