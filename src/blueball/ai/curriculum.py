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

from ..collision import register as register_collisions
from ..levels.loader import load_level
from ..world import World

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
