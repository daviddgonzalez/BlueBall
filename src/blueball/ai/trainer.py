"""Headless GA trainer.

`evaluate((idx, genome, world_seed, level_path, max_steps))` is the worker
function; it builds a fresh headless World, registers collisions, loads
the level, spawns one Player(FTNNAgent(genome)) at the level's spawn,
steps physics at PHYS_DT up to max_steps (or until the player dies or
reaches the goal), and returns (idx, fitness).

`train(...)` is the generation loop; `map_fn` defaults to `map` (serial,
in-process). Real training callers pass `multiprocessing.Pool(...).imap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict]                       # per-gen stats
    best_genome: np.ndarray                   # shape (GENOME_SIZE,)
    final_population: list[np.ndarray]        # for follow-up runs / TrainScene re-entry


def evaluate(args: tuple) -> tuple[int, float]:
    """One genome -> one fitness. Picklable input/output so it works under
    multiprocessing.Pool. Args is (idx, genome, world_seed, level_path, max_steps).
    """
    idx, genome, world_seed, level_path, max_steps = args

    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    meta = load_level(level_path, world)

    spawn_x, spawn_y = float(meta.spawn[0]), float(meta.spawn[1])
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y))
    world.add_entity(player)

    steps = 0
    while steps < max_steps:
        # Step exactly one physics tick. world.step(PHYS_DT) would route
        # through the accumulator and execute one substep; calling the
        # pymunk space directly skips the accumulator bookkeeping in this
        # headless path.
        world.space.step(config.PHYS_DT)
        for entity in world.entities:
            upd = getattr(entity, "update", None)
            if upd is not None:
                upd(config.PHYS_DT)
        steps += 1
        if player.dead or player.reached_goal:
            break

    f = fitness(FitnessInputs(
        progress_x=float(player.body.position.x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(player.reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
    ))
    return idx, float(f)


def train(
    *,
    pop_size: int,
    generations: int,
    level_path: Path,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable[[Callable, Iterable], Iterable] = map,
    on_generation: Callable[[int, np.ndarray, list[np.ndarray]], None] | None = None,
) -> TrainingResult:
    """Run a GA training loop. Returns a TrainingResult.

    `map_fn` is the parallelism strategy: defaults to the builtin `map`
    (serial). For real training runs pass `multiprocessing.Pool(N).imap`.

    `ga_seed` controls all GA randomness (population init, mutation, crossover,
    tournament). `world_seed` controls physics. Two runs with the same
    `(ga_seed, world_seed)` produce byte-identical `best_genome`.
    """
    ga_rng = np.random.default_rng(ga_seed)
    population: list[np.ndarray] = [random_genome(ga_rng) for _ in range(pop_size)]
    history: list[dict] = []
    best_genome = population[0].copy()
    best_fitness = -np.inf

    for gen in range(generations):
        args_iter = [
            (i, population[i], world_seed, level_path, max_steps)
            for i in range(pop_size)
        ]
        results = list(map_fn(evaluate, args_iter))
        # Restore order: results may arrive out-of-order from a Pool.
        results.sort(key=lambda r: r[0])
        fitnesses = np.array([r[1] for r in results], dtype=np.float64)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = float(fitnesses[gen_best_idx])
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[gen_best_idx].copy()

        history.append({
            "gen": gen,
            "best": gen_best,
            "mean": float(fitnesses.mean()),
            "min": float(fitnesses.min()),
        })

        if on_generation is not None:
            on_generation(gen, best_genome, population)

        population = breed(
            population, fitnesses, ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )

    return TrainingResult(
        history=history,
        best_genome=best_genome,
        final_population=population,
    )
