"""Renderer — draws the world to a PyGame surface using flat primitives."""

from __future__ import annotations

import math

import pygame
import pymunk

from .. import config


_BALL_COLOR = (58, 138, 255)
_BALL_DARK = (26, 90, 204)
_SPIKE_COLOR = (204, 68, 68)
_PATROLLER_COLOR = (220, 100, 60)
_COLLECTIBLE_COLOR = (255, 220, 60)
_GOAL_COLOR = (255, 240, 120)
_GOAL_FLAG = (220, 60, 60)
_GROUND_COLOR = (59, 138, 74)
_GROUND_EDGE = (40, 90, 50)
_ABILITY_PICKUP_DEFAULT = (220, 220, 220)
_ABILITY_PICKUP_COLORS = {"double_jump": (255, 220, 80)}
_BOOST_PAD_COLOR = (80, 220, 240)   # cyan
_BOOST_PAD_EDGE = (30, 150, 180)    # deeper cyan


def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, screen: pygame.Surface, camera) -> None:
        self.screen = screen
        self.camera = camera
        # Track previous positions per body for interpolation
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._prev_angle: dict[int, float] = {}

    def begin_frame(self, world) -> None:
        """Snapshot previous positions for next frame's interpolation."""
        for body in world.space.bodies:
            self._prev_pos[id(body)] = (body.position.x, body.position.y)
            self._prev_angle[id(body)] = body.angle

    def draw_background(self, color: tuple[int, int, int]) -> None:
        self.screen.fill(color)

    def _w2s(self, world_xy):
        return self.camera.world_to_screen(world_xy)

    def _interp_body_pos(self, body: pymunk.Body, alpha: float) -> tuple[float, float]:
        prev = self._prev_pos.get(id(body), (body.position.x, body.position.y))
        return (_lerp(prev[0], body.position.x, alpha), _lerp(prev[1], body.position.y, alpha))

    def _interp_body_angle(self, body: pymunk.Body, alpha: float) -> float:
        prev = self._prev_angle.get(id(body), body.angle)
        return _lerp(prev, body.angle, alpha)

    def draw_ball(self, body: pymunk.Body, alpha: float) -> None:
        wx, wy = self._interp_body_pos(body, alpha)
        sx, sy = self._w2s((wx, wy))
        r = config.BALL_RADIUS
        pygame.draw.circle(self.screen, _BALL_COLOR, (int(sx), int(sy)), r)
        # Rotated darker arc to show spin
        angle = self._interp_body_angle(body, alpha)
        arc_x = sx + math.cos(angle) * r * 0.5
        arc_y = sy + math.sin(angle) * r * 0.5
        pygame.draw.circle(self.screen, _BALL_DARK, (int(arc_x), int(arc_y)), int(r * 0.4))

    def draw_spike(self, pos, width, height, orientation: str = "up"):
        """Draw a spike triangle oriented to the given direction."""
        x, y = pos
        hw = width / 2
        if orientation == "up":
            verts = [(x - hw, y), (x + hw, y), (x, y - height)]
        elif orientation == "down":
            verts = [(x - hw, y), (x + hw, y), (x, y + height)]
        elif orientation == "left":
            verts = [(x, y - hw), (x, y + hw), (x - height, y)]
        else:  # right
            verts = [(x, y - hw), (x, y + hw), (x + height, y)]
        points = [self._w2s(v) for v in verts]
        pygame.draw.polygon(self.screen, _SPIKE_COLOR, points)

    def draw_collectible(self, pos):
        x, y = pos
        sx, sy = self._w2s((x, y))
        pulse = 1.0 + 0.15 * math.sin(
            pygame.time.get_ticks() / 1000.0 * 2 * math.pi * config.COLLECTIBLE_PULSE_HZ
        )
        r = int(config.COLLECTIBLE_RADIUS * pulse)
        pygame.draw.circle(self.screen, _COLLECTIBLE_COLOR, (int(sx), int(sy)), r)

    def draw_ability_pickup(self, pos, radius, ability: str) -> None:
        x, y = pos
        sx, sy = self._w2s((x, y))
        pulse = 1.0 + 0.15 * math.sin(
            pygame.time.get_ticks() / 1000.0 * 2 * math.pi * config.COLLECTIBLE_PULSE_HZ
        )
        r = int(radius * pulse)
        color = _ABILITY_PICKUP_COLORS.get(ability, _ABILITY_PICKUP_DEFAULT)
        points = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)]
        pygame.draw.polygon(self.screen, color, points)

    def draw_boost_pad(self, pos, width) -> None:
        # Flat cyan strip with a forward-pointing chevron at the center.
        x, y = pos
        hw = width / 2
        pad_h = config.BOOST_PAD_THICKNESS / 2
        p1 = self._w2s((x - hw, y - pad_h))
        p2 = self._w2s((x + hw, y - pad_h))
        p3 = self._w2s((x + hw, y + pad_h))
        p4 = self._w2s((x - hw, y + pad_h))
        pygame.draw.polygon(self.screen, _BOOST_PAD_COLOR, [p1, p2, p3, p4])
        cx, cy = self._w2s((x, y))
        pygame.draw.polygon(
            self.screen,
            _BOOST_PAD_EDGE,
            [(cx - 8, cy - 6), (cx + 6, cy), (cx - 8, cy + 6)],
        )

    def draw_goal(self, pos, width, height):
        x, y = pos
        hw, hh = width / 2, height / 2
        p1 = self._w2s((x - hw, y - hh))
        p2 = self._w2s((x + hw, y - hh))
        p3 = self._w2s((x + hw, y + hh))
        p4 = self._w2s((x - hw, y + hh))
        pygame.draw.polygon(self.screen, _GOAL_COLOR, [p1, p2, p3, p4])
        # Flag bunting
        fp1 = self._w2s((x - hw, y - hh))
        fp2 = self._w2s((x - hw + 30, y - hh + 10))
        fp3 = self._w2s((x - hw, y - hh + 20))
        pygame.draw.polygon(self.screen, _GOAL_FLAG, [fp1, fp2, fp3])

    def draw_patroller(self, body, size, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        hw, hh = size[0] / 2, size[1] / 2
        p1 = self._w2s((wx - hw, wy - hh))
        p2 = self._w2s((wx + hw, wy - hh))
        p3 = self._w2s((wx + hw, wy + hh))
        p4 = self._w2s((wx - hw, wy + hh))
        pygame.draw.polygon(self.screen, _PATROLLER_COLOR, [p1, p2, p3, p4])

    def draw_falling_hazard(self, body, radius, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        sx, sy = self._w2s((wx, wy))
        pygame.draw.circle(self.screen, _SPIKE_COLOR, (int(sx), int(sy)), radius)

    def draw_static_segments(self, space: pymunk.Space, color=_GROUND_COLOR) -> None:
        """Draw every static pymunk.Segment as a thick line plus a darker top edge."""
        for shape in space.shapes:
            if isinstance(shape, pymunk.Segment) and shape.body is space.static_body:
                a = self._w2s((shape.a.x, shape.a.y))
                b = self._w2s((shape.b.x, shape.b.y))
                pygame.draw.line(self.screen, color, a, b, 6)
                pygame.draw.line(self.screen, _GROUND_EDGE, a, b, 2)

    # ------------------------------------------------------------------ #
    # Phase 3 entity renderers                                            #
    # ------------------------------------------------------------------ #

    def draw_moving_platform(self, body: pymunk.Body, alpha: float, length: float) -> None:
        """Interpolated horizontal platform rect."""
        wx, wy = self._interp_body_pos(body, alpha)
        hw = length / 2
        thickness = 10
        p1 = self._w2s((wx - hw, wy - thickness / 2))
        p2 = self._w2s((wx + hw, wy - thickness / 2))
        p3 = self._w2s((wx + hw, wy + thickness / 2))
        p4 = self._w2s((wx - hw, wy + thickness / 2))
        pygame.draw.polygon(self.screen, (120, 200, 120), [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, (60, 140, 60), [p1, p2, p3, p4], 2)

    def draw_spring(self, pos, width: float, t: float) -> None:
        """Flat rect spring pad, color (170, 170, 220)."""
        x, y = pos
        hw = width / 2
        half_thick = 8
        # Slight squish animation when recently triggered (use t for gentle pulse)
        pulse_h = half_thick * (1.0 + 0.15 * abs(math.sin(t * 6)))
        p1 = self._w2s((x - hw, y - pulse_h))
        p2 = self._w2s((x + hw, y - pulse_h))
        p3 = self._w2s((x + hw, y + pulse_h))
        p4 = self._w2s((x - hw, y + pulse_h))
        pygame.draw.polygon(self.screen, (170, 170, 220), [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, (100, 100, 180), [p1, p2, p3, p4], 2)

    def draw_checkpoint(self, pos, radius: int, t: float, active: bool) -> None:
        """Diamond shape; bright (255,220,80) if active else (90,220,140)."""
        x, y = pos
        sx, sy = self._w2s((x, y))
        color = (255, 220, 80) if active else (90, 220, 140)
        r = radius
        points = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)]
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
        p1 = self._w2s((x - hw, y - thickness / 2))
        p2 = self._w2s((x + hw, y - thickness / 2))
        p3 = self._w2s((x + hw, y + thickness / 2))
        p4 = self._w2s((x - hw, y + thickness / 2))
        pygame.draw.polygon(self.screen, color, [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, (60, 40, 20), [p1, p2, p3, p4], 2)

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
            p1 = self._w2s((x - thick / 2, y))
            p2 = self._w2s((x + thick / 2, y))
            p3 = self._w2s((x + thick / 2, y - height))
            p4 = self._w2s((x - thick / 2, y - height))
            pygame.draw.polygon(self.screen, (160, 100, 40), [p1, p2, p3, p4])
            pygame.draw.polygon(self.screen, (220, 160, 80), [p1, p2, p3, p4], 2)

    def draw_pushable_box(self, body: pymunk.Body, alpha: float, size: float) -> None:
        """Interpolated square box."""
        wx, wy = self._interp_body_pos(body, alpha)
        hs = size / 2
        p1 = self._w2s((wx - hs, wy - hs))
        p2 = self._w2s((wx + hs, wy - hs))
        p3 = self._w2s((wx + hs, wy + hs))
        p4 = self._w2s((wx - hs, wy + hs))
        pygame.draw.polygon(self.screen, (160, 120, 80), [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, (100, 70, 30), [p1, p2, p3, p4], 2)

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
        pygame.draw.circle(self.screen, _SPIKE_COLOR, (sbx, sby), int(bob_radius))
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
            pygame.draw.polygon(self.screen, _SPIKE_COLOR, [base1, base2, (tip_x, tip_y)])

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
