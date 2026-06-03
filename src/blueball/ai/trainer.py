"""Headless GA trainer.

`evaluate((idx, genome, world_seed, level_path, max_steps))` is the worker
function; it builds a fresh headless World, registers collisions, loads
the level, spawns one Player(FTNNAgent(genome)) at the level's spawn,
steps physics at PHYS_DT up to max_steps (or until the player dies or
reaches the goal), and returns (idx, fitness).

`evaluate_infinite((idx, genome, sampler_seed, world_seed, max_steps))` is the
sibling worker for streamed Infinite Run: it drives a shared `TerrainStream`
(the same chunk pipeline PlayScene uses) so the agent trains on exactly the
terrain a human sees for that seed.

`train(...)` is the generation loop; pass `level_path` for a static level or
`infinite_seed` for Infinite Run. `map_fn` defaults to `map` (serial,
in-process). Real training callers pass `multiprocessing.Pool(...).imap`.

DETERMINISM: `evaluate` and `evaluate_infinite` call `world.substep()` once
per iteration — the drift-free path that runs exactly one PHYS_DT substep
with no accumulator residual. PHYS_DT = 1/120 is not exactly representable
in IEEE 754; using the real-time accumulator path (`world.step(PHYS_DT)`)
would let a tiny epsilon accumulate and occasionally fire a phantom extra
substep over thousands of iterations, breaking cross-machine determinism.
`substep()` bypasses the accumulator entirely so N calls == exactly N
substeps regardless of host float environment.
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
from ..levels.streaming import TerrainStream
from ..world import World
from .fitness import FitnessInputs, fitness
from .ftnn import GENOME_SIZE
from .ga import breed
from .genome import random_genome

# Spawn for streamed Infinite Run evaluation. Matches PlayScene's default
# Infinite Run spawn so the headless trainer and the visual TrainScene drop
# the ball where a human would start. The guaranteed Flat at x=0 gives it
# ground to land on.
INFINITE_SPAWN = (80.0, 540.0)


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict]                       # per-gen stats
    best_genome: np.ndarray                   # shape (GENOME_SIZE,)
    final_population: list[np.ndarray]        # for follow-up runs / TrainScene re-entry


def _episode_fitness(player, spawn_x: float, max_x: float, steps: int, reached_goal: bool) -> float:
    """Build the per-episode fitness from an evaluated player. Shared by both
    evaluators. `max_x` is the furthest x the player reached (>= spawn_x), so
    progress is robust to knockback / falling back before death."""
    return fitness(FitnessInputs(
        progress_x=float(max_x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=bool(reached_goal),
        died=bool(player.dead),
        steps_taken=steps,
        keys_collected=bin(player.keys_held).count("1"),
    ))


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

    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Use substep() — exactly one PHYS_DT step with no accumulator
        # residual, so long headless runs are bit-identical across machines.
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.dead or player.reached_goal:
            break

    f = _episode_fitness(player, spawn_x, max_x, steps,
                         reached_goal=bool(player.reached_goal))
    return idx, float(f)


def evaluate_infinite(args: tuple) -> tuple[int, float]:
    """One genome -> one fitness on a streamed Infinite Run. Picklable in/out
    for multiprocessing.Pool. Args is (idx, genome, sampler_seed, world_seed,
    max_steps).

    Streams chunks from the same `TerrainStream` the live game uses, so the
    agent is graded on exactly the terrain a human would see for that
    sampler_seed. Infinite Run has no goal, so fitness is driven by progress_x.
    """
    idx, genome, sampler_seed, world_seed, max_steps = args

    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    terrain = TerrainStream(world, int(sampler_seed))

    spawn_x, spawn_y = INFINITE_SPAWN
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y))
    world.add_entity(player)

    max_x = spawn_x
    steps = 0
    while steps < max_steps:
        # Extend/cull terrain ahead of the ball, then advance one substep.
        # Mirrors PlayScene.update's order (maintain then step).
        terrain.maintain(player.body.position.x)
        world.substep()
        steps += 1
        if player.body.position.x > max_x:
            max_x = player.body.position.x
        if player.dead:
            break

    f = _episode_fitness(player, spawn_x, max_x, steps, reached_goal=False)
    return idx, float(f)


def train(
    *,
    pop_size: int,
    generations: int,
    level_path: Path | None = None,
    infinite_seed: int | None = None,
    ga_seed: int = 0,
    world_seed: int = config.DEFAULT_SEED,
    max_steps: int = config.MAX_STEPS,
    map_fn: Callable[[Callable, Iterable], Iterable] = map,
    on_generation: Callable[[int, np.ndarray, list[np.ndarray]], None] | None = None,
    save_dir: Path | str | None = None,
) -> TrainingResult:
    """Run a GA training loop. Returns a TrainingResult.

    Provide exactly one of `level_path` (train on a static level) or
    `infinite_seed` (train on a streamed Infinite Run with that sampler seed).

    `map_fn` is the parallelism strategy: defaults to the builtin `map`
    (serial). For real training runs pass `multiprocessing.Pool(N).imap`.

    `ga_seed` controls all GA randomness (population init, mutation, crossover,
    tournament). `world_seed` controls physics. For Infinite Run, `infinite_seed`
    fixes the chunk layout. Two runs with the same seeds produce byte-identical
    `best_genome`.

    If `save_dir` is given, the running best genome is snapshotted there after
    every generation (best_gen<NNN>.npy) and the final best + a run.json
    metadata sidecar are written at the end. See `ai/persistence.py`.
    """
    if pop_size < 1:
        raise ValueError(f"train requires pop_size >= 1, got {pop_size}")
    if generations < 1:
        raise ValueError(f"train requires generations >= 1, got {generations}")
    if (level_path is None) == (infinite_seed is None):
        raise ValueError(
            "train requires exactly one of level_path or infinite_seed"
        )

    if infinite_seed is None:
        eval_fn = evaluate
        def make_args(i):
            return (i, population[i], world_seed, level_path, max_steps)
    else:
        eval_fn = evaluate_infinite
        def make_args(i):
            return (i, population[i], int(infinite_seed), world_seed, max_steps)

    writer = None
    if save_dir is not None:
        from .persistence import TrainingRunWriter
        writer = TrainingRunWriter(save_dir)

    ga_rng = np.random.default_rng(ga_seed)
    population: list[np.ndarray] = [random_genome(ga_rng) for _ in range(pop_size)]
    history: list[dict] = []
    best_genome = population[0].copy()
    best_fitness = -np.inf

    for gen in range(generations):
        args_iter = [make_args(i) for i in range(pop_size)]
        results = list(map_fn(eval_fn, args_iter))
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

        if writer is not None:
            writer.save_generation(gen, best_genome)

        if on_generation is not None:
            on_generation(gen, best_genome, population)

        population = breed(
            population, fitnesses, ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )

    if writer is not None:
        writer.finalize(best_genome, {
            "ga_seed": ga_seed,
            "world_seed": world_seed,
            "infinite_seed": infinite_seed,
            "level_path": str(level_path) if level_path is not None else None,
            "pop_size": pop_size,
            "generations": generations,
            "max_steps": max_steps,
            "genome_size": int(GENOME_SIZE),
            "best_fitness": float(best_fitness),
            "history": history,
        })

    return TrainingResult(
        history=history,
        best_genome=best_genome,
        final_population=population,
    )
