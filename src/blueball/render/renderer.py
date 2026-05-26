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

    def draw_spike(self, pos, width, height):
        x, y = pos
        half_w = width / 2
        p1 = self._w2s((x - half_w, y))
        p2 = self._w2s((x + half_w, y))
        p3 = self._w2s((x, y - height))
        pygame.draw.polygon(self.screen, _SPIKE_COLOR, [p1, p2, p3])

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
