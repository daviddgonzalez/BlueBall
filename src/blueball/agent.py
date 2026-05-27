"""Agent interface. v1 ships HumanAgent; AI agents arrive later.

NOTE: pygame is NOT imported at module level. HumanAgent imports pygame
lazily inside `act()` so that headless callers (the GA trainer, especially
under multiprocessing.Pool on hosts without DISPLAY/SDL) can import
`blueball.agent` without forcing pygame's C extension to load.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


@dataclass(frozen=True)
class Observation:
    rays: np.ndarray              # shape (8,), distances normalized 0..1
    vel: np.ndarray               # shape (2,), linear velocity (vx, vy)
    ang_vel: float
    grounded: bool
    nearest_collectible: Optional[tuple[float, float]]  # (dx, dy) or None


class Agent(abc.ABC):
    def reset(self, world) -> None:
        """Called at level start. Default no-op."""

    @abc.abstractmethod
    def act(self, observation: Observation) -> Action:
        ...


class HumanAgent(Agent):
    """Reads PyGame keyboard state and emits an Action.

    pygame is imported lazily inside ``act()`` rather than at module top so
    that headless callers (e.g. the GA trainer running under
    ``multiprocessing.Pool`` on a build server without DISPLAY) don't
    transitively load pygame's C extension when they import this module.
    """

    def act(self, observation: Observation) -> Action:
        import pygame  # lazy: see class docstring
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]

        if left and not right:
            return Action.LEFT_JUMP if jump else Action.LEFT
        if right and not left:
            return Action.RIGHT_JUMP if jump else Action.RIGHT
        if jump:
            return Action.JUMP
        return Action.IDLE


class FTNNAgent(Agent):
    """An Agent driven by a fixed-topology neural network (FTNN). Reads the
    observation, packs it into the 14-float input vector, runs it through
    the network, and returns the argmax Action.

    Imports of the `ai` package are lazy so that importing `agent` (which
    PlayScene and tests do) doesn't pull in the AI scaffolding transitively.
    """

    def __init__(self, genome: np.ndarray) -> None:
        from .ai.ftnn import FTNN
        from .ai.observation import observation_to_inputs

        self._net = FTNN(genome)
        self._to_inputs = observation_to_inputs

    def act(self, observation: Observation) -> Action:
        x = self._to_inputs(observation)
        y = self._net.forward(x)
        return Action(int(np.argmax(y)))
