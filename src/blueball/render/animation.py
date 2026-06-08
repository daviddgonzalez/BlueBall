"""Procedural animation helpers. Transform-based (resolution-independent) plus a
tiny frame cycler and a palette-cycle utility."""

from __future__ import annotations


def squash_stretch(vy: float, max_amount: float = 0.3, ref_speed: float = 400.0):
    """Return an (sx, sy) scale for vertical velocity vy (pymunk y-down: vy<0 is
    up). Stretches tall+thin while moving fast vertically; ~area-preserving."""
    amt = max(-1.0, min(1.0, -vy / ref_speed)) * max_amount  # up (vy<0) -> +stretch
    sy = 1.0 + amt
    sx = 1.0 / sy if sy != 0 else 1.0
    return (sx, sy)


class Anim:
    def __init__(self, n_frames: int, fps: float) -> None:
        self.n_frames = n_frames
        self.fps = fps

    def index(self, t_seconds: float) -> int:
        return int(t_seconds * self.fps) % self.n_frames


def palette_cycle(base: list, t_seconds: float, hz: float) -> int:
    """Index into a cyclic list (for lava shimmer / coin twinkle)."""
    return int(t_seconds * hz) % len(base)
