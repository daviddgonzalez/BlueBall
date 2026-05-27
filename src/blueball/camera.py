"""Camera — converts world coordinates to screen coordinates.

`Camera` is the pure-math base; `FollowCamera` tracks a target with a
smoothed lerp; `FreeCamera` is driven by arrow keys + plus/minus for
developer use in TrainScene.
"""

from __future__ import annotations

import pygame

from . import config


class Camera:
    """A free camera; pure math, no PyGame dependencies for math itself.

    `scale` is a uniform zoom factor. world_to_screen multiplies the world
    offset by `scale` so that zoom flows through to the renderer with no
    renderer-side change. Primitive sizes drawn by the renderer stay in
    screen-space (debug-tool intent).
    """

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.position: tuple[float, float] = (0.0, 0.0)
        self.scale: float = 1.0

    def world_to_screen(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        wx, wy = world_xy
        cx, cy = self.position
        s = self.scale
        return ((wx - cx) * s + self.viewport_w / 2,
                (wy - cy) * s + self.viewport_h / 2)


class FollowCamera(Camera):
    """A camera that trails a target with a dead-zone and a lerp."""

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        super().__init__(viewport_w, viewport_h)
        self.dead_zone_w = config.CAMERA_DEAD_ZONE_W
        self.dead_zone_h = config.CAMERA_DEAD_ZONE_H
        self.lerp = config.CAMERA_LERP

    def update(self, target: tuple[float, float], dt: float) -> None:
        tx, ty = target
        cx, cy = self.position
        dx = tx - cx
        dy = ty - cy
        half_w = self.dead_zone_w / 2
        half_h = self.dead_zone_h / 2

        chase_x = 0.0
        chase_y = 0.0
        if dx > half_w:
            chase_x = dx - half_w
        elif dx < -half_w:
            chase_x = dx + half_w
        if dy > half_h:
            chase_y = dy - half_h
        elif dy < -half_h:
            chase_y = dy + half_h

        self.position = (cx + chase_x * self.lerp, cy + chase_y * self.lerp)


class FreeCamera(Camera):
    """A free camera driven by keyboard input. TrainScene uses this so the
    developer can pan around and zoom while the GA population trains.

    - Arrow keys pan in world units per second; pan speed is divided by
      the current scale so panning feels consistent at any zoom level.
    - `+` / `-` (top row or numpad) zoom in / out multiplicatively, clamped
      to [ZOOM_MIN, ZOOM_MAX].
    """

    PAN_SPEED = 500.0
    ZOOM_STEP = 1.1
    ZOOM_MIN = 0.1
    ZOOM_MAX = 4.0

    def handle_events(self, events) -> None:
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.scale = min(self.ZOOM_MAX, self.scale * self.ZOOM_STEP)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.scale = max(self.ZOOM_MIN, self.scale / self.ZOOM_STEP)

    def update(self, keys_pressed, dt: float) -> None:
        dx = 0.0
        dy = 0.0
        if keys_pressed[pygame.K_LEFT]:
            dx -= 1.0
        if keys_pressed[pygame.K_RIGHT]:
            dx += 1.0
        if keys_pressed[pygame.K_UP]:
            dy -= 1.0
        if keys_pressed[pygame.K_DOWN]:
            dy += 1.0
        if dx == 0.0 and dy == 0.0:
            return
        # Divide by scale so panning at a low zoom doesn't whip the camera around.
        step = self.PAN_SPEED * dt / max(self.scale, 1e-6)
        cx, cy = self.position
        self.position = (cx + dx * step, cy + dy * step)
