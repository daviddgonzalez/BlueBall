"""Climb-height fitness term — dense vertical shaping for the vertical-climb
specialist. The term defaults to 0.0 so every non-curriculum caller (static
`evaluate`, `evaluate_infinite`, `evaluate_gym`, and therefore the generalist
and maze runs) is byte-identical."""

import numpy as np

from blueball.ai.fitness import FitnessInputs, fitness


def _base(**over):
    kw = dict(progress_x=100.0, collectibles=0, reached_goal=False, died=False,
              steps_taken=0, keys_collected=0, level_width=0.0)
    kw.update(over)
    return FitnessInputs(**kw)


def test_climb_height_defaults_to_zero_backward_compatible():
    # Omitting climb_height leaves fitness exactly progress_x — proves the new
    # term cannot perturb existing (non-curriculum) callers.
    assert fitness(_base()) == 100.0


def test_climb_height_adds_one_to_one_with_horizontal_progress():
    f0 = fitness(_base())
    fh = fitness(_base(climb_height=250.0))
    assert fh - f0 == 250.0


def test_evaluate_curriculum_rewards_climb_height_on_vertical_climb():
    """On vertical_climb, evaluate_curriculum adds the ball's net upward progress
    (climb_height) to the score, so a rising agent strictly outscores the flat
    trainer.evaluate run on the IDENTICAL deterministic sim. Both use the level's
    true spawn, the same genome/seed/abilities/step budget, so they differ ONLY
    by the height term."""
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.trainer import evaluate
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World

    path = resolve_level_paths(["vertical_climb"])[0]
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    spawn = (float(meta.spawn[0]), float(meta.spawn[1]))
    abilities = tuple(a.value for a in meta.starting_abilities)
    # A tracked genome that provably rises ~497px off the spawn trampoline.
    genome = np.load("genomes/inf1234_w1_20260604-183539/final_best.npy")
    world_seed, max_steps = 1, 1500

    _, f_flat = evaluate((0, genome, world_seed, str(path), max_steps, abilities))
    _, f_curric, _ = evaluate_curriculum(
        (0, genome, world_seed, str(path), max_steps, spawn, 0))

    # The rising ball earns climb_height; the flat evaluator does not.
    assert f_curric > f_flat


def test_evaluate_curriculum_no_climb_reward_on_horizontal_level():
    """The height term must engage ONLY for real climbs (goal above spawn). The
    maze is horizontal (goal ~level with the spawn), so its curriculum fitness
    stays byte-identical to the flat trainer.evaluate — the maze specialist run
    is unperturbed. (Without the gate, the ball jumping onto high maze platforms
    would inject a spurious climb_height bonus.)"""
    from blueball.ai.curriculum import evaluate_curriculum
    from blueball.ai.episodes import resolve_level_paths
    from blueball.ai.trainer import evaluate
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.world import World

    path = resolve_level_paths(["maze"])[0]
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    spawn = (float(meta.spawn[0]), float(meta.spawn[1]))
    abilities = tuple(a.value for a in meta.starting_abilities)
    genome = np.load("genomes/inf1234_w1_20260604-183539/final_best.npy")
    world_seed, max_steps = 1, 3000

    _, f_flat = evaluate((0, genome, world_seed, str(path), max_steps, abilities))
    _, f_curric, _ = evaluate_curriculum(
        (0, genome, world_seed, str(path), max_steps, spawn, 0))

    # No climb shaping on a horizontal level → identical to the flat evaluator.
    assert f_curric == f_flat
