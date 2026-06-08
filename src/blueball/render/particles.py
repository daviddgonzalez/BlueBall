"""Lightweight particle pool. Capped so it never blows the frame budget."""

from __future__ import annotations

import collections
import math

# kind: (speed, life, size, gravity, color_key)
_PRESETS = {
    "dust":    (60.0, 0.4, 2, 40.0, "ground_top"),
    "sparkle": (90.0, 0.5, 2, -20.0, "coin"),
    "burst":   (160.0, 0.6, 2, 120.0, "spike"),
    "trail":   (40.0, 0.3, 2, 0.0, "ball_hi"),
}


class ParticleSystem:
    def __init__(self, cap: int = 300) -> None:
        self.cap = cap
        self._p = collections.deque(maxlen=cap)  # deque auto-drops oldest past cap

    def __len__(self) -> int:
        return len(self._p)

    def emit(self, kind: str, at, n: int = 8, seed_angle: float = 0.0) -> None:
        speed, life, size, grav, key = _PRESETS[kind]
        for i in range(n):
            a = seed_angle + (i / max(1, n)) * 2 * math.pi
            self._p.append([at[0], at[1],
                            math.cos(a) * speed, math.sin(a) * speed,
                            life, size, grav, key])

    def update(self, dt: float) -> None:
        for p in self._p:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += p[6] * dt      # gravity on vy
            p[4] -= dt             # life
        self._p = collections.deque((p for p in self._p if p[4] > 0), maxlen=self.cap)

    def draw(self, renderer) -> None:
        theme = renderer._theme()
        for p in self._p:
            renderer._blit_point((p[0], p[1]), theme.palette[p[7]], int(p[5]))
