"""Camera — converts world coordinates to screen coordinates."""

from __future__ import annotations

from . import config


class Camera:
    """A free camera; pure math, no PyGame dependencies."""

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.position: tuple[float, float] = (0.0, 0.0)

    def world_to_screen(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        wx, wy = world_xy
        cx, cy = self.position
        return (wx - cx + self.viewport_w / 2, wy - cy + self.viewport_h / 2)


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
