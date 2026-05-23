"""Entity base class. Every game object inherits from this."""

from __future__ import annotations

import abc

import pymunk


class Entity(abc.ABC):
    """Abstract base for all game objects."""

    def __init__(self) -> None:
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        self.alive: bool = True

    def update(self, dt: float) -> None:
        """Per-physics-tick logic. Default is a no-op; override for behavior."""

    @abc.abstractmethod
    def draw(self, renderer, alpha: float) -> None:
        """Render the entity. `alpha` is the interpolation fraction
        (0 = previous tick, 1 = current tick)."""
