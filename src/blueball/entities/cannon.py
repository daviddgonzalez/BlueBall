"""Cannon — a fixed wall emitter that fires Projectiles periodically along one
horizontal direction. Body-less: it is purely a timed spawner plus a drawn
barrel, so it never participates in physics itself.
"""

from __future__ import annotations

import random

from .base import Entity
from .projectile import Projectile


class Cannon(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        direction: str = "right",
        interval_s: float = 2.0,
        interval_min_s: float | None = None,
        interval_max_s: float | None = None,
        speed: float = 220.0,
        pulse_period_s: float = 0.6,
        max_travel: float = 500.0,
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
        # When a min/max range is given, each gap between shots is a fresh
        # uniform draw in [min, max] (equally probable) instead of a fixed
        # cadence. Seeded per-cannon from the world seed + position so runs stay
        # deterministic without coupling to other systems' rng draws.
        self._randomized = interval_min_s is not None and interval_max_s is not None
        if self._randomized:
            self.interval_min_s = max(1e-3, float(interval_min_s))
            self.interval_max_s = max(self.interval_min_s, float(interval_max_s))
            self._rng = random.Random(
                f"{getattr(world, 'seed', 0)}:{round(position[0])}:{round(position[1])}"
            )
        self._interval = self._next_interval()
        # phase_s staggers cannons so they don't all fire on the same tick.
        self._t = phase_s % self._interval

    def _next_interval(self) -> float:
        if self._randomized:
            return self._rng.uniform(self.interval_min_s, self.interval_max_s)
        return self.interval_s

    def update(self, dt: float) -> None:
        self._t += dt
        while self._t >= self._interval:
            self._t -= self._interval
            self._fire()
            self._interval = self._next_interval()

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
