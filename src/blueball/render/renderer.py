"""Renderer — draws the world by blitting theme sprites (with primitive
fallbacks for entities not yet themed)."""

from __future__ import annotations

import math

import pygame
import pymunk

from .. import config


def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, screen, camera) -> None:
        # `screen` accepts a RenderCore (production) or a raw Surface (legacy tests).
        from .core import RenderCore
        if isinstance(screen, RenderCore):
            self.core = screen
            self.screen = screen.surface
        else:
            self.core = None
            self.screen = screen
        self.camera = camera
        # Track previous positions per body for interpolation
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._prev_angle: dict[int, float] = {}
        self._hud_font = None  # lazily created (needs pygame.font init)

    def begin_frame(self, world) -> None:
        """Snapshot previous positions for next frame's interpolation.

        Only moving (non-STATIC) bodies are snapshotted: static bodies never
        move, so _interp_body_pos's fallback (return current position when the id
        is absent) yields the identical result, and skipping them avoids
        re-snapshotting the shared static_body every frame plus unbounded growth
        of the snapshot dicts as streamed-in static geometry churns.
        """
        for body in world.space.bodies:
            if body.body_type == pymunk.Body.STATIC:
                continue
            self._prev_pos[id(body)] = (body.position.x, body.position.y)
            self._prev_angle[id(body)] = body.angle

    def reset_interpolation(self) -> None:
        """Clear cached per-body previous positions/angles.

        Call whenever the renderer's target World is swapped (e.g. TrainScene's
        generation rollover). Without the reset, these dicts accumulate stale
        entries keyed by ``id(body)`` of bodies that are now garbage-collected;
        CPython can reuse those ids for freshly allocated bodies and
        ``_interp_body_pos`` would then read a dead body's last position for an
        unrelated new body — a one-frame teleport.
        """
        self._prev_pos.clear()
        self._prev_angle.clear()

    def draw_background(self, color: tuple[int, int, int]) -> None:
        self.screen.fill(color)

    def _w2s(self, world_xy):
        return self.camera.world_to_screen(world_xy)

    def _theme(self):
        from .theme import get_active_theme
        return get_active_theme()

    def _blit_sprite(self, world_xy, key, *, deg=0.0, frame=0, scale=None):
        theme = self._theme()
        surf = theme.sprites[key].frame(frame, theme.palette)
        if scale is not None:
            sx, sy = scale
            surf = pygame.transform.scale(
                surf, (max(1, round(surf.get_width() * sx)),
                       max(1, round(surf.get_height() * sy))))
        if deg:
            surf = pygame.transform.rotate(surf, deg)
        px, py = self.camera.world_to_screen(world_xy)
        ox, oy = self.core.shake_offset if self.core else (0.0, 0.0)
        self.screen.blit(surf, surf.get_rect(center=(round(px + ox), round(py + oy))))

    def _interp_body_pos(self, body: pymunk.Body, alpha: float) -> tuple[float, float]:
        prev = self._prev_pos.get(id(body), (body.position.x, body.position.y))
        return (_lerp(prev[0], body.position.x, alpha), _lerp(prev[1], body.position.y, alpha))

    def _interp_body_angle(self, body: pymunk.Body, alpha: float) -> float:
        prev = self._prev_angle.get(id(body), body.angle)
        return _lerp(prev, body.angle, alpha)

    def _fill_rect(self, cx, cy, hw, hh, fill, edge=None, edge_w=2) -> None:
        """Fill a world-space rect centered at (cx, cy) with half-extents (hw, hh),
        optionally stroking an outline in `edge`."""
        pts = [
            self._w2s((cx - hw, cy - hh)),
            self._w2s((cx + hw, cy - hh)),
            self._w2s((cx + hw, cy + hh)),
            self._w2s((cx - hw, cy + hh)),
        ]
        pygame.draw.polygon(self.screen, fill, pts)
        if edge is not None:
            pygame.draw.polygon(self.screen, edge, pts, edge_w)

    def _pulse(self, magnitude: float = 0.15) -> float:
        """Time-based scale factor oscillating around 1.0 for idle-pulse visuals."""
        t = pygame.time.get_ticks() / 1000.0
        return 1.0 + magnitude * math.sin(t * 2 * math.pi * config.COLLECTIBLE_PULSE_HZ)

    @staticmethod
    def _diamond_points(sx, sy, r):
        """Screen-space diamond (4 points) centered at (sx, sy) with radius r."""
        return [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)]

    def draw_ball(self, body: pymunk.Body, alpha: float) -> None:
        wx, wy = self._interp_body_pos(body, alpha)
        angle = self._interp_body_angle(body, alpha)
        self._blit_sprite((wx, wy), "ball", deg=-math.degrees(angle))

    def draw_spike(self, pos, width, height, orientation: str = "up"):
        """Draw a spike sprite oriented to the given direction."""
        deg = {"up": 0, "right": -90, "down": 180, "left": 90}[orientation]
        self._blit_sprite(pos, "spike", deg=deg)

    def draw_collectible(self, pos):
        self._blit_sprite(pos, "collectible")

    def draw_ability_pickup(self, pos, radius, ability: str) -> None:
        ability_pickup_default = (220, 220, 220)
        ability_pickup_colors = {"double_jump": (255, 220, 80)}
        x, y = pos
        sx, sy = self._w2s((x, y))
        r = int(radius * self._pulse())
        color = ability_pickup_colors.get(ability, ability_pickup_default)
        pygame.draw.polygon(self.screen, color, self._diamond_points(sx, sy, r))

    def draw_boost_pad(self, pos, width, direction: float = 1.0) -> None:
        # Flat cyan strip with a chevron pointing along the pad's launch arrow.
        x, y = pos
        hw = width / 2
        pad_h = config.BOOST_PAD_THICKNESS / 2
        self._fill_rect(x, y, hw, pad_h, (80, 220, 240))
        cx, cy = self._w2s((x, y))
        s = -1 if direction < 0 else 1
        pygame.draw.polygon(
            self.screen,
            (30, 150, 180),
            [(cx - 8 * s, cy - 6), (cx + 6 * s, cy), (cx - 8 * s, cy + 6)],
        )

    def draw_goal(self, pos, width, height):
        self._blit_sprite(pos, "goal")

    def draw_patroller(self, body, size, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        hw, hh = size[0] / 2, size[1] / 2
        self._fill_rect(wx, wy, hw, hh, (220, 100, 60))

    def draw_falling_hazard(self, body, radius, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        sx, sy = self._w2s((wx, wy))
        pygame.draw.circle(self.screen, self._theme().palette["spike"], (int(sx), int(sy)), radius)

    def draw_static_segments(self, space: pymunk.Space, color=None) -> None:
        """Draw every static pymunk.Segment as a thick line plus a darker top edge."""
        if color is None:
            color = self._theme().palette["ground"]
        sw, sh = self.screen.get_size()
        for shape in space.shapes:
            if isinstance(shape, pymunk.Segment) and shape.body is space.static_body:
                a = self._w2s((shape.a.x, shape.a.y))
                b = self._w2s((shape.b.x, shape.b.y))
                # Skip segments whose screen-space bounding box is entirely off
                # the viewport — they contribute no on-screen pixels. Uses min/max
                # of both endpoints so a long segment crossing the screen with
                # both endpoints off-screen is still drawn.
                if (max(a[0], b[0]) < 0 or min(a[0], b[0]) > sw
                        or max(a[1], b[1]) < 0 or min(a[1], b[1]) > sh):
                    continue
                pygame.draw.line(self.screen, color, a, b, 6)
                pygame.draw.line(self.screen, (40, 90, 50), a, b, 2)

    # ------------------------------------------------------------------ #
    # Phase 3 entity renderers                                            #
    # ------------------------------------------------------------------ #

    def draw_moving_platform(self, body: pymunk.Body, alpha: float, length: float) -> None:
        """Interpolated horizontal platform rect."""
        wx, wy = self._interp_body_pos(body, alpha)
        hw = length / 2
        thickness = 10
        self._fill_rect(wx, wy, hw, thickness / 2, (120, 200, 120), edge=(60, 140, 60))

    def draw_spring(self, pos, width: float, t: float) -> None:
        """Flat rect spring pad, color (170, 170, 220)."""
        x, y = pos
        hw = width / 2
        half_thick = 8
        # Slight squish animation when recently triggered (use t for gentle pulse)
        pulse_h = half_thick * (1.0 + 0.15 * abs(math.sin(t * 6)))
        self._fill_rect(x, y, hw, pulse_h, (170, 170, 220), edge=(100, 100, 180))

    def draw_checkpoint(self, pos, radius: int, t: float, active: bool) -> None:
        """Diamond shape; bright (255,220,80) if active else (90,220,140)."""
        x, y = pos
        sx, sy = self._w2s((x, y))
        color = (255, 220, 80) if active else (90, 220, 140)
        points = self._diamond_points(sx, sy, radius)
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

    def draw_crumbling_platform(self, pos, alpha: float, width: float, progress: float) -> None:
        """Platform rect that darkens as progress approaches 1.0."""
        x, y = pos
        hw = width / 2
        thickness = 10
        # Lerp from a warm tan to near-black as progress -> 1
        r = int(180 * (1.0 - progress))
        g = int(140 * (1.0 - progress))
        b = int(100 * (1.0 - progress))
        color = (max(0, r), max(0, g), max(0, b))
        self._fill_rect(x, y, hw, thickness / 2, color, edge=(60, 40, 20))

    def draw_key(self, pos, radius: int, key_id: int) -> None:
        """Small yellow circle with a keyhole notch."""
        x, y = pos
        sx, sy = self._w2s((x, y))
        pygame.draw.circle(self.screen, (255, 200, 60), (int(sx), int(sy)), radius)
        pygame.draw.circle(self.screen, (180, 120, 10), (int(sx), int(sy)), radius, 2)
        # Small inner circle as keyhole decoration
        inner_r = max(3, radius // 3)
        pygame.draw.circle(self.screen, (180, 120, 10), (int(sx), int(sy)), inner_r)

    def draw_door(self, pos, height: int, open_: bool) -> None:
        """Vertical bar; thinner outline-only when open."""
        x, y = pos
        if open_:
            # Outline only — door is passable
            a = self._w2s((x, y))
            b = self._w2s((x, y - height))
            pygame.draw.line(self.screen, (180, 140, 80), a, b, 2)
        else:
            thick = 8
            self._fill_rect(x, y - height / 2, thick / 2, height / 2,
                            (160, 100, 40), edge=(220, 160, 80))

    def draw_pushable_box(self, body: pymunk.Body, alpha: float, size: float) -> None:
        """Interpolated square box."""
        wx, wy = self._interp_body_pos(body, alpha)
        hs = size / 2
        self._fill_rect(wx, wy, hs, hs, (160, 120, 80), edge=(100, 70, 30))

    def draw_swinging_hazard(
        self,
        anchor_body: pymunk.Body,
        bob_body: pymunk.Body,
        bob_radius: float,
        alpha: float,
    ) -> None:
        """Line from anchor to bob, plus a spiky circle for the bob."""
        ax, ay = anchor_body.position.x, anchor_body.position.y
        bx, by = self._interp_body_pos(bob_body, alpha)
        sa = self._w2s((ax, ay))
        sb = self._w2s((bx, by))
        pygame.draw.line(self.screen, (160, 140, 100), sa, sb, 2)
        # Bob: filled circle + spike points around perimeter
        sbx, sby = int(sb[0]), int(sb[1])
        spike_color = self._theme().palette["spike"]
        pygame.draw.circle(self.screen, spike_color, (sbx, sby), int(bob_radius))
        pygame.draw.circle(self.screen, (240, 100, 100), (sbx, sby), int(bob_radius), 2)
        # 4 spike tips
        for angle_deg in (0, 90, 180, 270):
            a_rad = math.radians(angle_deg)
            tip_x = sbx + int((bob_radius + 5) * math.cos(a_rad))
            tip_y = sby + int((bob_radius + 5) * math.sin(a_rad))
            base1 = (sbx + int(3 * math.cos(a_rad + math.pi / 2)),
                     sby + int(3 * math.sin(a_rad + math.pi / 2)))
            base2 = (sbx + int(3 * math.cos(a_rad - math.pi / 2)),
                     sby + int(3 * math.sin(a_rad - math.pi / 2)))
            pygame.draw.polygon(self.screen, spike_color, [base1, base2, (tip_x, tip_y)])

    def draw_one_way_platform(self, pos, width: float) -> None:
        """Thin platform strip with a downward chevron arrow."""
        x, y = pos
        hw = width / 2
        # Platform line
        a = self._w2s((x - hw, y))
        b = self._w2s((x + hw, y))
        pygame.draw.line(self.screen, (80, 200, 160), a, b, 4)
        # Downward chevron at center
        cx, cy = self._w2s((x, y))
        arrow_pts = [(cx - 8, cy - 4), (cx, cy + 4), (cx + 8, cy - 4)]
        pygame.draw.lines(self.screen, (80, 200, 160), False, arrow_pts, 2)

    def draw_charger(self, body: pymunk.Body, alpha: float, state: str, radius: int = 12) -> None:
        """Interpolated circle; (255,120,120) when charging else (200,80,80)."""
        wx, wy = self._interp_body_pos(body, alpha)
        sx, sy = self._w2s((wx, wy))
        color = (255, 120, 120) if state == "charge" else (200, 80, 80)
        pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
        pygame.draw.circle(self.screen, (255, 200, 200), (int(sx), int(sy)), radius, 2)

    def draw_lava(self, body: pymunk.Body, alpha: float, width: float, height: float) -> None:
        """Wide orange rectangle. Top edge of the rectangle is the lava's
        deadly surface; body.position is the top-edge center.
        """
        wx, wy = self._interp_body_pos(body, alpha)
        top_left = self._w2s((wx - width / 2, wy))
        rect = pygame.Rect(int(top_left[0]), int(top_left[1]), int(width), int(height))
        pygame.draw.rect(self.screen, (240, 90, 30), rect)
        # Bright top edge for visibility
        pygame.draw.line(
            self.screen,
            (255, 220, 100),
            (int(top_left[0]), int(top_left[1])),
            (int(top_left[0] + width), int(top_left[1])),
            3,
        )

    def draw_projectile(self, body: pymunk.Body, alpha: float, radius: int = 10) -> None:
        """Interpolated fiery orb fired by a cannon."""
        wx, wy = self._interp_body_pos(body, alpha)
        sx, sy = self._w2s((wx, wy))
        pygame.draw.circle(self.screen, (255, 140, 40), (int(sx), int(sy)), radius)
        pygame.draw.circle(self.screen, (255, 230, 150), (int(sx), int(sy)), radius, 2)

    def draw_cannon(self, position, direction: str) -> None:
        """Wall-mounted barrel pointing in the firing direction. The cannon is
        fixed, so its position is drawn directly (no interpolation)."""
        sx, sy = self._w2s(position)
        barrel_w, barrel_h = 18, 12
        if direction == "right":
            rect = pygame.Rect(int(sx), int(sy - barrel_h / 2), barrel_w, barrel_h)
        else:
            rect = pygame.Rect(int(sx - barrel_w), int(sy - barrel_h / 2), barrel_w, barrel_h)
        pygame.draw.rect(self.screen, (90, 90, 110), rect)
        pygame.draw.circle(self.screen, (60, 60, 80), (int(sx), int(sy)), 9)
        pygame.draw.circle(self.screen, (120, 120, 150), (int(sx), int(sy)), 9, 2)

    def draw_score(self, score: int, best: int) -> None:
        """Top-left HUD: current run score and the persisted best."""
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont(None, 32)
        score_surf = self._hud_font.render(f"Score: {score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (16, 12))
        best_surf = self._hud_font.render(f"Best: {best}", True, (255, 220, 80))
        self.screen.blit(best_surf, (16, 44))
