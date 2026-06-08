"""Renderer tests. The draw path has historically had zero coverage; these are
crash-smoke + interpolation-snapshot guards for the efficiency changes to
begin_frame (skip STATIC bodies) and draw_static_segments (viewport cull)."""

import pymunk

from blueball.render.renderer import Renderer
from blueball.world import World


def test_begin_frame_snapshots_pre_step_position_of_moving_body():
    """A DYNAMIC body's pre-step pose must be captured for interpolation."""
    w = World()
    body = pymunk.Body(mass=1.0, moment=10.0)
    body.position = (5.0, 7.0)
    shape = pymunk.Circle(body, 4.0)
    w.space.add(body, shape)

    r = Renderer(screen=None, camera=None)
    r.begin_frame(w)
    body.position = (99.0, 99.0)  # move AFTER the snapshot

    assert r._prev_pos[id(body)] == (5.0, 7.0)  # pre-move pose retained


def test_begin_frame_does_not_break_static_interpolation():
    """Static bodies never move, so _interp_body_pos must return their current
    position whether or not begin_frame snapshotted them (the fallback path)."""
    w = World()
    seg = pymunk.Segment(w.space.static_body, (0, 600), (400, 600), 5)
    w.space.add(seg)

    r = Renderer(screen=None, camera=None)
    r.begin_frame(w)
    # Interpolating the shared static body returns its (unchanging) position.
    sb = w.space.static_body
    assert r._interp_body_pos(sb, 0.5) == (sb.position.x, sb.position.y)


def test_draw_static_segments_culls_offscreen_keeps_onscreen(monkeypatch):
    """The viewport cull must skip fully-off-screen segments and still draw
    on-screen ones. Smoke tests can't catch a predicate inversion, so spy on
    pygame.draw.line and assert which segments actually got drawn."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    screen = pygame.display.set_mode((400, 300))

    class _IdentityCamera:
        def world_to_screen(self, p):
            return (p[0], p[1])

    r = Renderer(screen=screen, camera=_IdentityCamera())
    space = pymunk.Space()
    on_screen = pymunk.Segment(space.static_body, (10, 10), (200, 200), 3)
    off_screen = pymunk.Segment(space.static_body, (500, 50), (700, 50), 3)
    space.add(on_screen, off_screen)

    drawn = []
    monkeypatch.setattr(
        pygame.draw, "line",
        lambda surf, color, a, b, width=1: drawn.append((tuple(a), tuple(b))),
    )
    try:
        r.draw_static_segments(space)
    finally:
        pygame.display.quit()

    endpoints = {pt for seg in drawn for pt in seg}
    assert (10.0, 10.0) in endpoints and (200.0, 200.0) in endpoints  # on-screen drawn
    assert (500.0, 50.0) not in endpoints  # off-screen culled
    assert (700.0, 50.0) not in endpoints


class _ShiftCamera:
    """Identity projection plus a mutable (dx, dy) world offset."""
    def __init__(self):
        self.dx = self.dy = 0.0

    def world_to_screen(self, p):
        return (p[0] + self.dx, p[1] + self.dy)


def _spy_segments(monkeypatch, pygame, r, space):
    drawn = []
    monkeypatch.setattr(
        pygame.draw, "line",
        lambda surf, color, a, b, width=1: drawn.append((tuple(a), tuple(b))),
    )
    r.draw_static_segments(space)
    return drawn


def test_static_segments_reproject_when_camera_moves(monkeypatch):
    """Endpoints are cached in WORLD space, so a camera move on an unchanged
    space must still shift the drawn screen coords (guards a stale screen-space
    cache)."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    screen = pygame.display.set_mode((400, 300))
    cam = _ShiftCamera()
    r = Renderer(screen=screen, camera=cam)
    space = pymunk.Space()
    space.add(pymunk.Segment(space.static_body, (10, 10), (200, 200), 3))
    try:
        first = _spy_segments(monkeypatch, pygame, r, space)
        cam.dx = cam.dy = 25.0  # scroll the camera, geometry unchanged
        second = _spy_segments(monkeypatch, pygame, r, space)
    finally:
        pygame.display.quit()

    assert ((10.0, 10.0), (200.0, 200.0)) in first
    assert ((35.0, 35.0), (225.0, 225.0)) in second  # re-projected, not stale


def test_static_segments_cache_refreshes_when_geometry_changes(monkeypatch):
    """Adding a static segment (as streamed chunks do) must appear on the next
    draw — the cache cannot go permanently stale."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    screen = pygame.display.set_mode((400, 300))
    r = Renderer(screen=screen, camera=_ShiftCamera())
    space = pymunk.Space()
    space.add(pymunk.Segment(space.static_body, (10, 10), (50, 50), 3))
    try:
        first = _spy_segments(monkeypatch, pygame, r, space)
        space.add(pymunk.Segment(space.static_body, (60, 60), (120, 120), 3))
        second = _spy_segments(monkeypatch, pygame, r, space)
    finally:
        pygame.display.quit()

    first_pts = {pt for seg in first for pt in seg}
    second_pts = {pt for seg in second for pt in seg}
    assert (60.0, 60.0) not in first_pts          # not present before it existed
    assert (60.0, 60.0) in second_pts             # appears after geometry change
    assert (10.0, 10.0) in second_pts             # original still drawn
