"""Reverse spawn-curriculum for training a single-level specialist on a hard
level (maze first).

Training-only scaffolding: the agent spawns partway through the level (near the
goal first) and the spawn recedes toward the true start as the population masters
each stage. At each spawn the agent is granted the keys for any gate behind it,
so doors are openable. The saved genome is an ordinary genome evaluated from the
real spawn. See docs/superpowers/specs/2026-06-06-maze-curriculum-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from ..agent import FTNNAgent
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..world import World
from .fitness import FitnessInputs, fitness

_KEY_NAME = "Key"
_GOAL_NAME = "Goal"

# How far before a key (px) the agent spawns, so it must move toward the key to
# collect it rather than collecting it on the spawn frame. Module constant for
# easy iteration; selection is comparative so the exact value isn't load-bearing.
SPAWN_MARGIN = 96.0


@dataclass(frozen=True)
class CurriculumStage:
    """One staged spawn. `granted_keys` is OR'd into player.keys_held at spawn."""

    spawn_xy: tuple[float, float]
    granted_keys: int
    label: str   # "near_goal" | "before_key<id>" | "start"


def granted_keys_before(keys: list[tuple[int, float]], spawn_x: float) -> int:
    """OR of (1 << key_id) for every (key_id, key_x) whose key_x < spawn_x.

    A door behind the spawn opens because its key's pickup is also behind the
    spawn (keys precede their doors), so its bit is granted. Strict `<`: a key at
    exactly spawn_x is *not* granted (the agent collects it itself)."""
    mask = 0
    for key_id, kx in keys:
        if kx < spawn_x:
            mask |= (1 << key_id)
    return mask


def build_spawn_curriculum(level: Union[str, Path, dict]) -> list[CurriculumStage]:
    """Derive the ordered (easiest -> hardest) reverse-curriculum stages from a
    level's real entity positions. Spawn x recedes start-ward across the list:

        near_goal      -> midpoint of (last key, goal); all keys granted
        before_key<id> -> SPAWN_MARGIN before each key, closest-to-start last,
                          granting keys whose pickup is strictly behind the spawn
        start          -> the level's true spawn (no keys granted)

    `near_goal` is emitted only when the level has both keys and a goal; if the
    goal is absent the list begins at the `before_key<id>` stages.
    """
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(level, world)

    keys: list[tuple[int, float]] = []
    goal_x: float | None = None
    for e in world.entities:
        cn = type(e).__name__
        if cn == _KEY_NAME:
            keys.append((int(e.key_id), float(e.position[0])))
        elif cn == _GOAL_NAME:
            goal_x = float(e.position[0])
    keys.sort(key=lambda k: k[1])  # ascending x (start -> goal)

    spawn_y = float(meta.spawn[1])
    start_x = float(meta.spawn[0])
    stages: list[CurriculumStage] = []

    # Easiest: near the goal, after the last key (all keys behind -> granted).
    if keys and goal_x is not None:
        near_goal_x = (keys[-1][1] + goal_x) / 2.0
        stages.append(CurriculumStage(
            spawn_xy=(near_goal_x, spawn_y),
            granted_keys=granted_keys_before(keys, near_goal_x),
            label="near_goal",
        ))

    # One stage just before each key, hardest (closest to start) emitted last.
    for key_id, kx in reversed(keys):
        sx = kx - SPAWN_MARGIN
        stages.append(CurriculumStage(
            spawn_xy=(sx, spawn_y),
            granted_keys=granted_keys_before(keys, sx),
            label=f"before_key{key_id}",
        ))

    # Hardest: the true start. Grants any key whose pickup precedes the spawn
    # (normally none — the rule is applied uniformly rather than hardcoding 0).
    stages.append(CurriculumStage(
        spawn_xy=(start_x, spawn_y),
        granted_keys=granted_keys_before(keys, start_x),
        label="start",
    ))
    return stages


def make_curriculum_player(world, genome, spawn_xy, granted_keys: int) -> Player:
    """Spawn a Player at `spawn_xy`, add it to `world`, and grant `granted_keys`
    (OR'd into keys_held) so doors behind the spawn are openable. Shared by the
    evaluator and tests."""
    player = Player(agent=FTNNAgent(genome),
                    spawn_xy=(float(spawn_xy[0]), float(spawn_xy[1])))
    world.add_entity(player)
    player.keys_held |= int(granted_keys)
    return player


def evaluate_curriculum(args: tuple) -> tuple[int, float, bool]:
    """One genome -> (idx, fitness, reached_goal) on a curriculum stage. Picklable
    in/out for multiprocessing.Pool. Args is
    (idx, genome, world_seed, level_path, max_steps, spawn_xy, granted_keys).

    Mirrors trainer.evaluate's drift-free substep loop, but spawns at the stage
    override with granted keys and additionally returns whether the goal was
    reached (the success signal the adaptive curriculum loop consumes)."""
    idx, genome, world_seed, level_path, max_steps, spawn_xy, granted_keys = args

    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    meta = load_level(level_path, world)

    spawn_x = float(spawn_xy[0])
    player = make_curriculum_player(world, genome, spawn_xy, granted_keys)

    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator residual,
        # so long headless runs are bit-identical across machines (see trainer).
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.dead or player.reached_goal:
            break

    # Granted keys are training scaffolding, not achievements: count only the
    # keys actually collected this episode (bits set that were NOT granted), so
    # the curriculum can't hand out free fitness.
    collected = bin(player.keys_held & ~int(granted_keys)).count("1")
    f = fitness(FitnessInputs(
        progress_x=float(max_x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(player.reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
        keys_collected=collected,
        level_width=float(meta.total_width),
    ))
    return idx, float(f), bool(player.reached_goal)
