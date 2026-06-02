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

DETERMINISM CAVEAT: `evaluate` calls `world.step(config.PHYS_DT)` once per
iteration. PHYS_DT = 1/120 is not exactly representable in IEEE 754, so the
world's accumulator drifts by a tiny epsilon each step. Over thousands of
iterations the drift could cross PHYS_DT and fire an extra substep,
breaking exact determinism across float environments (different numpy/
python builds, x86 vs ARM). The smoke test passes consistently today, but
if `test_trainer_is_deterministic_under_same_seed` ever flakes on CI, this
is the place to look — a follow-up should switch the accumulator to an
integer substep counter or pre-quantize PHYS_DT.
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
# Infinite Run spawn so the headless trainer drops the ball where a human
# would start. The guaranteed Flat at x=0 gives it ground to land on.
_INFINITE_SPAWN = (80.0, 540.0)


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
        # Use World.step so the headless path stays in lockstep with the
        # live game. Passing exactly PHYS_DT means the accumulator runs
        # exactly one substep per call.
        world.step(config.PHYS_DT)
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

    spawn_x, spawn_y = _INFINITE_SPAWN
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y))
    world.add_entity(player)

    steps = 0
    while steps < max_steps:
        # Extend/cull terrain ahead of the ball, then advance one substep.
        # Mirrors PlayScene.update's order (maintain then step).
        terrain.maintain(player.body.position.x)
        world.step(config.PHYS_DT)
        steps += 1
        if player.dead:
            break

    f = fitness(FitnessInputs(
        progress_x=float(player.body.position.x - spawn_x),
        collectibles=int(player.collectibles_collected),
        reached_goal=False,  # Infinite Run has no goal
        died=bool(player.dead),
        steps_taken=steps,
    ))
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
