"""Entity base class. Every game object inherits from this."""

from __future__ import annotations

import abc

import pymunk


class Entity(abc.ABC):
    """Abstract base for all game objects."""

    def __init__(self) -> None:
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        self.constraints: list[pymunk.Constraint] = []
        self.alive: bool = True

    def update(self, dt: float) -> None:
        """Per-physics-tick logic. Default is a no-op; override for behavior."""

    def _remove_from_space(self) -> None:
        """Remove this entity's constraints, shapes, and bodies from the physics
        space. Idempotent — anything already gone is skipped. Does not touch
        `alive` or drop the entity from `world.entities`; callers own their
        lifecycle flags. Requires the entity to have been added via
        `World.add_entity` (which sets `self._world`).
        """
        space = self._world.space
        for constraint in self.constraints:
            if constraint in space.constraints:
                space.remove(constraint)
        for shape in self.shapes:
            if shape in space.shapes:
                space.remove(shape)
            # Drop the shape->entity index entry so it can't leak (the index is
            # unbounded over an endless Infinite Run otherwise).
            self._world._shape_to_entity.pop(shape, None)
        for body in self.bodies:
            if body in space.bodies:
                space.remove(body)

    @abc.abstractmethod
    def draw(self, renderer, alpha: float) -> None:
        """Render the entity. `alpha` is the interpolation fraction
        (0 = previous tick, 1 = current tick)."""
