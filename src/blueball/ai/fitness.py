"""Fitness function for GA training. v1 spec's starting shape.

Tunable; this is the function most likely to be iterated during real
training. Lives in its own module so iteration touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float    # player.body.position.x - spawn_x
    collectibles: int    # player.collectibles_collected
    reached_goal: bool   # player.reached_goal
    died: bool           # player.dead
    steps_taken: int     # the loop counter from evaluate()


def fitness(inputs: FitnessInputs) -> float:
    return (
        inputs.progress_x
        + 50.0  * inputs.collectibles
        + 200.0 * (1.0 if inputs.reached_goal else 0.0)
        -   0.01 * inputs.steps_taken
        - 100.0 * (1.0 if inputs.died else 0.0)
    )
