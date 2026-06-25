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
    climb_height: float = 0.0  # net upward progress (px); only the vertical
                               # curriculum populates it — 0.0 everywhere else
    purposeful_jumps: int = 0  # ground takeoffs taken at a real ledge/gap


def fitness(inputs: FitnessInputs) -> float:
    # Efficiency reward only in goal-terminal modes, where the episode ends on
    # goal so steps_taken distinguishes a fast finisher from a dawdler/bouncer.
    # Degenerate for goalless modes (every survivor uses all max_steps), so off.
    efficiency = 0.0
    if inputs.level_width > 0.0:
        efficiency = config.SPEED_W * inputs.progress_x / max(inputs.steps_taken, 1)
    return (
        inputs.progress_x
        + inputs.climb_height
        + 100.0 * inputs.keys_collected
        +  50.0 * inputs.collectibles
        + config.GOAL_MULT * inputs.level_width * (1.0 if inputs.reached_goal else 0.0)
        + config.GYM_SEGMENT_BONUS * inputs.segments_cleared
        + config.JUMP_GAP_BONUS * inputs.purposeful_jumps
        + efficiency
        -   0.01 * inputs.steps_taken
        - 200.0 * (1.0 if inputs.died else 0.0)
    )
