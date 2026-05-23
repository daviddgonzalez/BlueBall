"""Agent interface. v1 ships HumanAgent; AI agents arrive later."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pygame


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
    """Reads PyGame keyboard state and emits an Action."""

    def act(self, observation: Observation) -> Action:
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
