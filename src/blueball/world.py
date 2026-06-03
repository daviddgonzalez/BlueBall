"""World — owns the pymunk space and the fixed-timestep physics loop."""

from __future__ import annotations

import random

import pymunk

from . import config


class World:
    """A physics world. Holds the pymunk Space, a list of entities,
    and steps physics with a fixed-substep accumulator for determinism.
    """

    def __init__(self, seed: int = config.DEFAULT_SEED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = config.GRAVITY
        self._accumulator = 0.0
        self.entities: list = []
        self.level_complete = False

    def add_entity(self, entity) -> None:
        """Register an entity with the world. Adds the entity's bodies and shapes
        to the pymunk space and tracks the entity for per-tick updates.
        """
        entity._world = self
        for body in getattr(entity, "bodies", ()):
            self.space.add(body)
        for shape in getattr(entity, "shapes", ()):
            self.space.add(shape)
        for constraint in getattr(entity, "constraints", ()):
            self.space.add(constraint)
        self.entities.append(entity)

    def complete_level(self) -> None:
        self.level_complete = True

    def _run_one_substep(self) -> None:
        """Advance physics + entities by exactly one PHYS_DT substep."""
        self.space.step(config.PHYS_DT)
        for entity in self.entities:
            update = getattr(entity, "update", None)
            if update is not None:
                update(config.PHYS_DT)

    def substep(self) -> None:
        """Advance by exactly one fixed PHYS_DT substep, bypassing the
        real-time accumulator. Deterministic across hosts — N calls == N
        substeps with no float residual. Used by the headless trainer; the
        live game uses step(frame_dt) for real-time pacing.
        """
        self._run_one_substep()

    def step(self, frame_dt: float) -> int:
        """Advance the simulation by `frame_dt` real seconds.

        Internally runs zero or more fixed substeps of `config.PHYS_DT`.
        Returns the number of substeps actually executed (useful for tests
        and for debug overlays).
        """
        self._accumulator += frame_dt
        substeps = 0
        while self._accumulator >= config.PHYS_DT and substeps < config.MAX_ACCUMULATED_STEPS:
            self._run_one_substep()
            self._accumulator -= config.PHYS_DT
            substeps += 1

        if self._accumulator >= config.PHYS_DT:
            # Spiral-of-death guard: drop leftover time we couldn't run
            self._accumulator = 0.0
        return substeps

    @property
    def alpha(self) -> float:
        """Interpolation fraction the renderer should use between the previous and
        current physics state. 0.0 means "draw the previous tick"; 1.0 means
        "draw the most recent tick".
        """
        return self._accumulator / config.PHYS_DT
