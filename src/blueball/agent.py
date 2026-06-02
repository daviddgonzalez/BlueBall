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

from . import collision as _col


class Action(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


class HitType(IntEnum):
    MISS = 0
    GROUND = 1
    HAZARD = 2
    PICKUP = 3
    GOAL = 4
    ENEMY = 5
    BLOCK = 6
    DOOR = 7


_CT_TO_HITTYPE: dict[int, HitType] = {
    _col.CT_PLAYER: HitType.GROUND,  # never hit, but safe default
    _col.CT_SPIKE: HitType.HAZARD,
    _col.CT_PATROLLER: HitType.ENEMY,
    _col.CT_COLLECTIBLE: HitType.PICKUP,
    _col.CT_GOAL: HitType.GOAL,
    _col.CT_BOOST_PAD: HitType.PICKUP,
    _col.CT_ABILITY_PICKUP: HitType.PICKUP,
    _col.CT_ONE_WAY: HitType.GROUND,
    _col.CT_SPRING: HitType.PICKUP,
    _col.CT_PUSHABLE: HitType.BLOCK,
    _col.CT_SWINGING: HitType.HAZARD,
    _col.CT_CHARGER: HitType.ENEMY,
    _col.CT_CHECKPOINT: HitType.PICKUP,
    _col.CT_KEY: HitType.PICKUP,
    _col.CT_DOOR: HitType.DOOR,
}


@dataclass(frozen=True)
class Observation:
    rays: np.ndarray              # shape (8,), float32, in [0, 1]; 1.0 = miss
    ray_hit_types: np.ndarray     # shape (8,), int8 HitType values
    vel: np.ndarray               # shape (2,), float32
    ang_vel: float
    grounded: bool
    nearest_pickup: Optional[tuple[float, float]]
    nearest_hazard: Optional[tuple[float, float]]
    abilities: int                # bitfield, ability enum ordinal
    keys_held: int                # bitfield


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
    observation, packs it into the input vector (see ai/observation.py
    INPUT_SIZE), runs it through the network, and returns the argmax Action.

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
