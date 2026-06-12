"""Tests for the watch-best playback tool (scenes/playback.py).

The playback tool loads a saved genome and plays it visually. Its core promise
is *faithfulness*: the live HUD fitness must equal what the headless trainer
would score the same genome (the whole point is to *see* a number you already
trust). So the heart of these tests is comparing PlaybackSim's final fitness to
the trainer's evaluate* functions for the same genome/seed/steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pygame
import pytest

import blueball
from blueball.ai.genome import random_genome
from blueball.scenes.playback import (
    PlaybackScene,
    PlaybackSim,
    load_genome,
    resolve_genome_path,
)

WORLD_SEED = 1


@pytest.fixture
def headless_surface():
    """A windowless pygame surface (dummy SDL driver) for scene smoke tests."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.font.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.event.clear()
    pygame.display.quit()


def _level_path(name: str = "tutorial_hill") -> str:
    return str(Path(blueball.__file__).parent / "levels" / f"{name}.json")


def _drive(sim: PlaybackSim) -> PlaybackSim:
    """Step a sim to completion (its own termination, no rendering)."""
    guard = 0
    while not sim.done:
        sim.step_once()
        guard += 1
        assert guard < 10_000, "sim failed to terminate"
    return sim


def test_resolve_genome_path_directory_resolves_to_final_best(tmp_path):
    run_dir = tmp_path / "inf1234_w1_run"
    run_dir.mkdir()
    np.save(run_dir / "final_best.npy", random_genome(np.random.default_rng(0)))
    assert resolve_genome_path(run_dir) == run_dir / "final_best.npy"


def test_resolve_genome_path_file_passes_through(tmp_path):
    genome_file = tmp_path / "some_genome.npy"
    np.save(genome_file, random_genome(np.random.default_rng(0)))
    assert resolve_genome_path(genome_file) == genome_file


def test_static_fitness_matches_headless_evaluate():
    """The whole point of the HUD: PlaybackSim's final fitness on a static level
    must equal what the trainer's headless `evaluate` scores the same genome."""
    from blueball.ai.trainer import evaluate

    genome = random_genome(np.random.default_rng(7))
    level = _level_path("tutorial_hill")
    max_steps = 400

    _, headless_fit = evaluate((0, genome, WORLD_SEED, level, max_steps))

    sim = PlaybackSim(genome, mode="static", level_path=level,
                      world_seed=WORLD_SEED, max_steps=max_steps)
    _drive(sim)
    assert sim.fitness == pytest.approx(headless_fit)


def test_infinite_fitness_matches_headless_evaluate_infinite():
    """Infinite Run playback must equal the trainer's streamed evaluate_infinite."""
    from blueball.ai.trainer import evaluate_infinite

    genome = random_genome(np.random.default_rng(7))
    seed, max_steps = 1234, 400

    _, headless_fit = evaluate_infinite((0, genome, seed, WORLD_SEED, max_steps))

    sim = PlaybackSim(genome, mode="infinite", seed=seed,
                      world_seed=WORLD_SEED, max_steps=max_steps)
    _drive(sim)
    assert sim.fitness == pytest.approx(headless_fit)


def test_infinite_fitness_matches_headless_evaluate_infinite_with_abilities():
    """Regression: infinite playback WITH double_jump must match evaluate_infinite's
    abilities run. The ability-gated double-jump chunks must appear in the REPLAY
    terrain too — previously only the player got the grant, so the sampler omitted
    them and the replay ran easier terrain (overstating distance/fitness)."""
    from blueball.ai.trainer import evaluate_infinite

    genome = random_genome(np.random.default_rng(3))
    seed, max_steps = 5, 1500
    abilities = ("double_jump",)

    _, headless_fit = evaluate_infinite(
        (0, genome, seed, WORLD_SEED, max_steps, abilities))

    sim = PlaybackSim(genome, mode="infinite", seed=seed, world_seed=WORLD_SEED,
                      max_steps=max_steps, abilities=abilities)
    _drive(sim)
    assert sim.fitness == pytest.approx(headless_fit)


def test_infinite_playback_grants_abilities_to_terrain_sampler():
    """Root-cause guard: the terrain sampler (not just the player) must receive the
    granted abilities, so ability-gated chunks surface — matching evaluate_infinite."""
    from blueball.abilities import Ability

    genome = random_genome(np.random.default_rng(3))
    sim = PlaybackSim(genome, mode="infinite", seed=5, world_seed=WORLD_SEED,
                      max_steps=120, abilities=("double_jump",))
    assert Ability.DOUBLE_JUMP in sim._stream.sampler.abilities


def test_gym_fitness_matches_headless_evaluate_gym():
    """Completion-Gym playback (with double_jump, as the gym is trained) must
    equal the trainer's evaluate_gym — including its segment-clear counting."""
    from blueball.ai.trainer import evaluate_gym

    genome = random_genome(np.random.default_rng(7))
    seed, max_steps = 4242, 600
    abilities = ("double_jump",)

    _, headless_fit = evaluate_gym(
        (0, genome, seed, WORLD_SEED, max_steps, abilities))

    sim = PlaybackSim(genome, mode="gym", seed=seed, world_seed=WORLD_SEED,
                      max_steps=max_steps, abilities=abilities)
    _drive(sim)
    assert sim.fitness == pytest.approx(headless_fit)


def test_scene_grants_requested_abilities(headless_surface):
    """The player replays with exactly the abilities handed to the sim — the
    double-jump invariant (a gym/maze genome must replay double-jumping)."""
    from blueball.abilities import Ability

    genome = random_genome(np.random.default_rng(7))
    sim = PlaybackSim(genome, mode="gym", seed=4242, world_seed=WORLD_SEED,
                      max_steps=120, abilities=("double_jump",))
    assert sim.player.abilities == {Ability.DOUBLE_JUMP}


def test_scene_advances_sim_and_draws(headless_surface):
    """update() steps the underlying sim; draw() exercises the whole render +
    HUD path without raising."""
    genome = random_genome(np.random.default_rng(7))
    sim = PlaybackSim(genome, mode="static", level_path=_level_path(),
                      world_seed=WORLD_SEED, max_steps=400)
    scene = PlaybackScene(headless_surface, sim)
    before = sim.steps
    scene.update(1 / 60)
    assert sim.steps > before
    scene.draw()  # world + HUD overlay must not raise


def test_scene_smoke_from_saved_genome(headless_surface, tmp_path):
    """The CI smoke path: load a genome from a run-dir on disk, construct the
    scene, and run a few frames headlessly (no real window)."""
    run_dir = tmp_path / "inf1234_w1_run"
    run_dir.mkdir()
    np.save(run_dir / "final_best.npy", random_genome(np.random.default_rng(1)))

    genome = load_genome(run_dir)
    sim = PlaybackSim(genome, mode="infinite", seed=1, world_seed=WORLD_SEED,
                      max_steps=120)
    scene = PlaybackScene(headless_surface, sim)
    for _ in range(5):
        scene.update(1 / 60)
        scene.draw()
    assert sim.steps > 0


# --- CLI wiring: sim construction + argument parsing ---------------------- #

def _saved_genome(tmp_path):
    gpath = tmp_path / "g.npy"
    np.save(gpath, random_genome(np.random.default_rng(0)))
    return str(gpath)


def test_build_playback_sim_gym_defaults_to_double_jump(tmp_path):
    """A --gym replay grants double_jump by default (the gym is trained with it),
    so the HUD matches the gym evaluator out of the box."""
    from blueball.abilities import Ability
    from blueball.scenes.playback import build_playback_sim

    sim = build_playback_sim(_saved_genome(tmp_path), gym=4242, world_seed=WORLD_SEED)
    assert sim.mode == "gym"
    assert sim.player.abilities == {Ability.DOUBLE_JUMP}


def test_build_playback_sim_defaults_to_static_level_no_abilities(tmp_path):
    """No mode flag -> a static level, granting no abilities (matching the
    headless static evaluator, which grants none)."""
    from blueball.scenes.playback import build_playback_sim

    sim = build_playback_sim(_saved_genome(tmp_path), world_seed=WORLD_SEED)
    assert sim.mode == "static"
    assert sim.player.abilities == set()
    assert sim.level_width > 0  # a real level was loaded


def test_build_playback_sim_abilities_override(tmp_path):
    """An explicit --abilities string overrides the per-mode default (e.g. to
    replay a double-jump-trained genome on a static level faithfully)."""
    from blueball.abilities import Ability
    from blueball.scenes.playback import build_playback_sim

    sim = build_playback_sim(_saved_genome(tmp_path), level="maze",
                             abilities="double_jump", world_seed=WORLD_SEED)
    assert sim.player.abilities == {Ability.DOUBLE_JUMP}


def test_watch_best_parser_infinite():
    from blueball.cli import build_parser, cmd_watch_best

    args = build_parser().parse_args(["watch-best", "mygen.npy", "--infinite", "7"])
    assert args.func is cmd_watch_best
    assert args.target == "mygen.npy"
    assert args.infinite == 7
    assert args.level is None and args.gym is None


def test_watch_best_parser_mode_flags_mutually_exclusive():
    from blueball.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["watch-best", "g", "--gym", "1", "--level", "maze"])
