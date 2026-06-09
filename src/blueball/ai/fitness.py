"""Fitness function for GA training. v1 spec's starting shape.

Tunable; this is the function most likely to be iterated during real
training. Lives in its own module so iteration touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float    # furthest x reached - spawn_x
    collectibles: int    # player.collectibles_collected
    reached_goal: bool   # player.reached_goal
    died: bool           # player.dead
    steps_taken: int     # the loop counter from the evaluator
    keys_collected: int  # popcount of player.keys_held
    level_width: float   # level total width; 0.0 for goalless (infinite) modes
    segments_cleared: int = 0  # gym: count of solved segments; 0 elsewhere


def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + config.GOAL_MULT * inputs.level_width * (1.0 if inputs.reached_goal else 0.0)
        + config.GYM_SEGMENT_BONUS * inputs.segments_cleared
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
    )
