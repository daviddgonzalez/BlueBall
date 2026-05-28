"""Cannon — a fixed wall emitter that fires Projectiles periodically along one
horizontal direction. Body-less: it is purely a timed spawner plus a drawn
barrel, so it never participates in physics itself.
"""

from __future__ import annotations

from .base import Entity
from .projectile import Projectile


class Cannon(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        direction: str = "right",
        interval_s: float = 2.0,
        speed: float = 220.0,
        pulse_period_s: float = 0.6,
        max_travel: float = 200.0,
        projectile_radius: int = 10,
        phase_s: float = 0.0,
    ) -> None:
        super().__init__()
        if direction not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right'; got {direction!r}")
        self._world_ref = world
        self.position = position
        self.direction = direction
        self._dir_sign = 1.0 if direction == "right" else -1.0
        self.interval_s = max(1e-3, interval_s)
        self.speed = speed
        self.pulse_period_s = pulse_period_s
        self.max_travel = max_travel
        self.projectile_radius = projectile_radius
        # phase_s staggers cannons so they don't all fire on the same tick.
        self._t = phase_s % self.interval_s

    def update(self, dt: float) -> None:
        self._t += dt
        while self._t >= self.interval_s:
            self._t -= self.interval_s
            self._fire()

    def _fire(self) -> None:
        self._world_ref.add_entity(Projectile(
            self._world_ref,
            position=self.position,
            direction=self._dir_sign,
            speed=self.speed,
            pulse_period_s=self.pulse_period_s,
            max_travel=self.max_travel,
            radius=self.projectile_radius,
        ))

    def draw(self, renderer, alpha: float) -> None:
        renderer.draw_cannon(self.position, self.direction)
