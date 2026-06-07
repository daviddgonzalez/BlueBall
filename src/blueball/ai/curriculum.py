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
from typing import Callable, Union

import numpy as np

from .. import config
from ..agent import FTNNAgent
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..world import World
from .fitness import FitnessInputs, fitness
from .ftnn import GENOME_SIZE
from .ga import breed
from .genome import random_genome
from .trainer import TrainingResult

_KEY_NAME = "Key"
_GOAL_NAME = "Goal"

# How far before a key (px) the agent spawns, so it must move toward the key to
# collect it rather than collecting it on the spawn frame. Module constant for
# easy iteration; selection is comparative so the exact value isn't load-bearing.
SPAWN_MARGIN = 96.0

# How far left of the box's left face (px) the box-lava specialist spawns, so
# rolling right immediately contacts and pushes the box. Module constant for
# easy iteration; selection is comparative so the exact value isn't load-bearing.
BOX_LAVA_SPAWN_MARGIN = 12.0


@dataclass(frozen=True)
class CurriculumStage:
    """One staged spawn. `granted_keys` is OR'd into player.keys_held at spawn."""

    spawn_xy: tuple[float, float]
    granted_keys: int
    label: str   # "near_goal" | "before_key<id>" | "start" | "box_lava"


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


def build_box_lava_curriculum(level: Union[str, Path, dict]) -> list[CurriculumStage]:
    """Single-stage curriculum for the box-lava section: spawn just left of the
    PushableBox (so rolling right pushes it into the lava as a stepping stone)
    with every key behind the spawn granted (for the maze, both gates). Used to train a
    box-lava specialist.

    Returns exactly one CurriculumStage labelled "box_lava". Raises ValueError
    if the level has no PushableBox.
    """
    world = World(seed=0)
    register_collisions(world.space, world_ref=world)
    meta = load_level(level, world)

    box = next((e for e in world.entities
                if type(e).__name__ == "PushableBox"), None)
    if box is None:
        raise ValueError("build_box_lava_curriculum: level has no PushableBox")

    keys: list[tuple[int, float]] = [
        (int(e.key_id), float(e.position[0]))
        for e in world.entities if type(e).__name__ == _KEY_NAME
    ]

    box_x = float(box.body.position.x)
    spawn_x = box_x - box.size / 2.0 - BOX_LAVA_SPAWN_MARGIN
    spawn_y = float(meta.spawn[1])
    return [CurriculumStage(
        spawn_xy=(spawn_x, spawn_y),
        granted_keys=granted_keys_before(keys, spawn_x),
        label="box_lava",
    )]


def make_curriculum_player(world, genome, spawn_xy, granted_keys: int,
                           abilities=frozenset()) -> Player:
    """Spawn a Player at `spawn_xy`, add it to `world`, and grant `granted_keys`
    (OR'd into keys_held) so doors behind the spawn are openable. `abilities`
    are the level's assumed starting abilities (e.g. double jump). Passed as a
    fresh mutable set so the player's shared abilities set stays mutable. Shared
    by the evaluator and tests."""
    player = Player(agent=FTNNAgent(genome),
                    spawn_xy=(float(spawn_xy[0]), float(spawn_xy[1])),
                    abilities=set(abilities))
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
    player = make_curriculum_player(world, genome, spawn_xy, granted_keys,
                                    meta.starting_abilities)

    # Track the PushableBox's rightward displacement (the box-push reward
    # gradient). Levels with no box leave box_progress at 0.0 -> no behavior
    # change. Mirrors the player's max_x high-water mark (robust to knockback).
    box = next((e for e in world.entities
                if type(e).__name__ == "PushableBox"), None)
    box_start_x = float(box.body.position.x) if box is not None else None
    box_max_x = box_start_x

    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator residual,
        # so long headless runs are bit-identical across machines (see trainer).
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if box is not None and box.body.position.x > box_max_x:
            box_max_x = box.body.position.x
        if player.dead or player.reached_goal:
            break

    box_progress = max(0.0, box_max_x - box_start_x) if box is not None else 0.0

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
        box_progress=float(box_progress),
    ))
    return idx, float(f), bool(player.reached_goal)


def train_curriculum(
    *,
    level_path: Union[str, Path],
    pop_size: int,
    generations: int,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable = map,
    save_dir: Union[Path, str, None] = None,
    stages: Union[list[CurriculumStage], None] = None,
) -> "TrainingResult":
    """Adaptive reverse spawn-curriculum GA loop for one level.

    Holds a current stage index (starts at 0 = near_goal). Each generation the
    whole population is evaluated at the current stage's spawn + granted keys;
    selection uses raw fitness (the unchanged GA: breed/elitism/tournament). If
    the generation's best (elite) genome reached the goal from the current spawn,
    the stage recedes one step toward the true start. Deterministic given
    (ga_seed, world_seed): population, evaluations, and the stage trajectory are
    all reproducible. The saved best genome is curriculum-free at evaluation.
    Note: best_genome is the argmax of in-loop fitness, which spans stages with
    different spawn scaffolding, so its quality is only meaningful when
    re-evaluated from the true start after training (the CLI does this).
    """
    if pop_size < 1:
        raise ValueError(f"train_curriculum requires pop_size >= 1, got {pop_size}")
    if generations < 1:
        raise ValueError(f"train_curriculum requires generations >= 1, got {generations}")

    # Default None preserves today's behavior exactly; a custom one-element list
    # (e.g. build_box_lava_curriculum) trains a fixed-spawn specialist that never
    # recedes (it's already the last stage).
    stages = stages if stages is not None else build_spawn_curriculum(level_path)

    writer = None
    if save_dir is not None:
        from .persistence import TrainingRunWriter
        writer = TrainingRunWriter(save_dir)

    ga_rng = np.random.default_rng(ga_seed)
    population = [random_genome(ga_rng) for _ in range(pop_size)]
    history: list[dict] = []
    best_genome = population[0].copy()
    best_fitness = -np.inf

    stage_index = 0
    reached_gen: dict[int, int] = {0: 0}  # stage_index -> first gen current (stage 0 from gen 0)
    cleared_gen: dict[int, int] = {}      # stage_index -> first gen elite cleared it

    for gen in range(generations):
        stage = stages[stage_index]
        args_iter = [
            (i, population[i], world_seed, str(level_path), max_steps,
             stage.spawn_xy, stage.granted_keys)
            for i in range(pop_size)
        ]
        results = list(map_fn(evaluate_curriculum, args_iter))
        results.sort(key=lambda r: r[0])  # restore order (Pool may reorder)
        fitnesses = np.array([r[1] for r in results], dtype=np.float64)
        reached = [bool(r[2]) for r in results]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = float(fitnesses[gen_best_idx])
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[gen_best_idx].copy()

        elite_cleared = reached[gen_best_idx]
        history.append({
            "gen": gen,
            "stage": stage_index,
            "stage_label": stage.label,
            "best": gen_best,
            "mean": float(fitnesses.mean()),
            "best_reached_goal": elite_cleared,
        })

        if writer is not None:
            writer.save_generation(gen, best_genome)

        # Adaptive advancement: recede one stage once the elite clears this one.
        if elite_cleared:
            cleared_gen.setdefault(stage_index, gen)
            if stage_index < len(stages) - 1:
                stage_index += 1
                reached_gen.setdefault(stage_index, gen + 1)

        population = breed(
            population, fitnesses, ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )

    if writer is not None:
        trajectory = [
            {"stage": i, "label": stages[i].label,
             "reached_gen": reached_gen.get(i), "cleared_gen": cleared_gen.get(i)}
            for i in range(len(stages))
        ]
        last = len(stages) - 1
        writer.finalize(best_genome, {
            "mode": "curriculum",
            "level_path": str(level_path),
            "ga_seed": ga_seed,
            "world_seed": world_seed,
            "pop_size": pop_size,
            "generations": generations,
            "max_steps": max_steps,
            "genome_size": int(GENOME_SIZE),
            "best_fitness": float(best_fitness),
            "curriculum": {
                "stages": [s.label for s in stages],
                "trajectory": trajectory,
                "final_stage_index": stage_index,
                "final_stage_label": stages[stage_index].label,
                "cracked": stage_index == last and last in cleared_gen,
            },
            "history": history,
        })

    return TrainingResult(history=history, best_genome=best_genome,
                          final_population=population)
