"""Single entry point for Blue Ball.

One canonical command — `python main.py <subcommand>` — replaces the old pile of
top-level scripts (main.py / train_main.py / train_infinite.py / train_levels.py
/ train_maze_curriculum.py / train_completion_gym.py / probe_*.py).

    python main.py                 # play the game (default)
    python main.py play            # play the game (explicit)
    python main.py watch           # watch the GA train, live (pygame scene)

    python main.py train infinite  # headless GA on Infinite Run
    python main.py train levels     # headless GA across the static levels
    python main.py train maze       # headless reverse spawn-curriculum on maze
    python main.py train gym        # headless GA on the Completion Gym

    python main.py repro-boost      # reproduce the boost-pad bug (headless trace)
    python main.py repro-boost --play   # ...as a playable level instead

    python main.py play-gym box-lava    # play one gym segment by hand (tune by feel)
    python main.py play-gym boost-gap --gap 28

Every `train` subcommand keeps the flags the old scripts had. Run any of them
with `-h` for the full list.
"""

from __future__ import annotations

import argparse
import multiprocessing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config


# --------------------------------------------------------------------------- #
# shared training helpers
# --------------------------------------------------------------------------- #
def _add_common_train_args(p: argparse.ArgumentParser, *, max_steps_default: int) -> None:
    p.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    p.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    p.add_argument("--max-steps", type=int, default=max_steps_default)
    p.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    p.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count())


@contextmanager
def _pool(workers: int):
    """Yield (pool, map_fn). Serial map when workers<=1; closed on exit."""
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    try:
        yield pool, (pool.imap if pool is not None else map)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _run_dir(**kwargs) -> Path:
    from .ai.persistence import GENOMES_ROOT, run_dir_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(GENOMES_ROOT) / run_dir_name(timestamp=timestamp, **kwargs)


# --------------------------------------------------------------------------- #
# train subcommands
# --------------------------------------------------------------------------- #
def cmd_train_infinite(args) -> int:
    from .ai.episodes import generate_seeds, infinite_episodes
    from .ai.trainer import train

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else generate_seeds(args.infinite_seed, args.num_seeds))
    episodes = infinite_episodes(seeds, world_seed=args.world_seed, max_steps=args.max_steps)
    run_dir = _run_dir(infinite_seed=seeds[0], world_seed=args.world_seed, num_seeds=len(seeds))
    print(f"Training {args.pop}x{args.gens} on Infinite Run seeds={seeds} "
          f"world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}")
    with _pool(args.workers) as (_, map_fn):
        result = train(pop_size=args.pop, generations=args.gens, episodes=episodes,
                       ga_seed=args.ga_seed, world_seed=args.world_seed,
                       max_steps=args.max_steps, map_fn=map_fn, save_dir=run_dir)
    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


def cmd_train_levels(args) -> int:
    from .ai.episodes import available_levels, resolve_level_paths, static_episodes
    from .ai.trainer import train

    names = ([n.strip() for n in args.levels.split(",") if n.strip()] if args.levels
             else available_levels())
    try:
        level_paths = resolve_level_paths(names)
    except ValueError as e:
        raise SystemExit(str(e))
    episodes = static_episodes(level_paths, world_seed=args.world_seed, max_steps=args.max_steps)
    run_dir = _run_dir(world_seed=args.world_seed, num_levels=len(level_paths))
    print(f"Training {args.pop}x{args.gens} on levels={names} "
          f"world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}")
    with _pool(args.workers) as (_, map_fn):
        result = train(pop_size=args.pop, generations=args.gens, episodes=episodes,
                       aggregate="min", ga_seed=args.ga_seed, world_seed=args.world_seed,
                       max_steps=args.max_steps, map_fn=map_fn, save_dir=run_dir)
    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.3f} mean={final['mean']:.3f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


def cmd_train_maze(args) -> int:
    from .ai.curriculum import (build_spawn_curriculum, evaluate_curriculum,
                                train_curriculum)
    from .ai.episodes import resolve_level_paths

    try:
        level_path = resolve_level_paths([args.level])[0]
    except ValueError as e:
        raise SystemExit(str(e))
    run_dir = _run_dir(world_seed=args.world_seed, level_name=args.level, curriculum=True)
    # Built here for the stage-count print + the true-start verdict spawn below;
    # train_curriculum builds its own copy internally.
    stages = build_spawn_curriculum(level_path)
    print(f"Curriculum training {args.pop}x{args.gens} on {args.level} "
          f"({len(stages)} stages) world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}")
    with _pool(args.workers) as (_, map_fn):
        result = train_curriculum(level_path=level_path, pop_size=args.pop,
                                  generations=args.gens, ga_seed=args.ga_seed,
                                  world_seed=args.world_seed, max_steps=args.max_steps,
                                  map_fn=map_fn, save_dir=run_dir)
    # Verdict: evaluate the final best genome from the TRUE start (no scaffolding).
    start = stages[-1]
    _, fit, reached = evaluate_curriculum(
        (0, result.best_genome, args.world_seed, level_path, args.max_steps,
         start.spawn_xy, start.granted_keys))
    final = result.history[-1]
    print(f"Done. gen {final['gen']}: final stage '{final['stage_label']}'")
    print(f"Verdict @ true start: reached_goal={reached} fitness={fit:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


def cmd_train_gym(args) -> int:
    from .ai.episodes import generate_seeds, gym_episodes
    from .ai.trainer import train

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else generate_seeds(args.gym_seed, args.num_seeds))
    abilities = tuple(a.strip() for a in args.abilities.split(",") if a.strip())
    episodes = gym_episodes(seeds, world_seed=args.world_seed,
                            max_steps=args.max_steps, abilities=abilities)
    run_dir = _run_dir(gym_seed=seeds[0], world_seed=args.world_seed, num_seeds=len(seeds))
    print(f"Training {args.pop}x{args.gens} on Completion Gym seeds={seeds} "
          f"abilities={abilities or '(single jump)'} world={args.world_seed} "
          f"ga={args.ga_seed}\n  -> {run_dir}")
    with _pool(args.workers) as (_, map_fn):
        result = train(pop_size=args.pop, generations=args.gens, episodes=episodes,
                       ga_seed=args.ga_seed, world_seed=args.world_seed,
                       max_steps=args.max_steps, map_fn=map_fn, save_dir=run_dir)
    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


# --------------------------------------------------------------------------- #
# pygame subcommands
# --------------------------------------------------------------------------- #
def cmd_play(args) -> int:
    import os
    import pygame
    from .scenes.menu import MenuScene

    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball")
    clock = pygame.time.Clock()
    # Optional live FPS readout in the title bar (BLUEBALL_FPS=1). Off by default
    # so normal play is unaffected; lets you measure real window-present FPS,
    # which a headless profiler can't (display.flip is a no-op under dummy SDL).
    show_fps = bool(os.environ.get("BLUEBALL_FPS"))
    scene = MenuScene(screen)
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
        if show_fps:
            pygame.display.set_caption(f"Blue Ball — {clock.get_fps():4.1f} fps")
    pygame.quit()
    return 0


def cmd_watch(args) -> int:
    import pygame
    from .scenes.train import TrainScene

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    scene = TrainScene(screen, infinite_seed=config.INFINITE_RUN_SEED)
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
    pygame.quit()
    return 0


def cmd_repro_boost(args) -> int:
    from .debug.boost_repro import play_repro, trace_repro
    return play_repro() if args.play else trace_repro()


def cmd_play_gym(args) -> int:
    from .debug.gym_play import play_segment
    if args.segment == "box-lava":
        return play_segment("box-lava", pit_tiles=args.pit, depth=args.depth)
    return play_segment("boost-gap", gap_tiles=args.gap)


def cmd_play_doublejump(args) -> int:
    from .debug.double_jump_play import play_showcase
    return play_showcase()


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("play", help="play the game").set_defaults(func=cmd_play)
    sub.add_parser("watch", help="watch the GA train live").set_defaults(func=cmd_watch)

    p_repro = sub.add_parser("repro-boost", help="reproduce the boost-pad bug")
    p_repro.add_argument("--play", action="store_true",
                         help="boot a playable boost-pad level instead of the headless trace")
    p_repro.set_defaults(func=cmd_repro_boost)

    p_pg = sub.add_parser("play-gym", help="play a single completion-gym segment by hand")
    p_pg.add_argument("segment", choices=["box-lava", "boost-gap"],
                      help="which gym segment to play")
    p_pg.add_argument("--pit", type=int, default=None, help="box-lava: pit_tiles")
    p_pg.add_argument("--depth", type=int, default=None, help="box-lava: pit depth (px)")
    p_pg.add_argument("--gap", type=int, default=None, help="boost-gap: lava_gap pit_tiles")
    p_pg.set_defaults(func=cmd_play_gym)

    sub.add_parser(
        "play-doublejump",
        help="play a hand-built level showcasing the double-jump chunks",
    ).set_defaults(func=cmd_play_doublejump)

    p_train = sub.add_parser("train", help="headless GA training")
    tsub = p_train.add_subparsers(dest="mode", required=True)

    p_inf = tsub.add_parser("infinite", help="Infinite Run")
    _add_common_train_args(p_inf, max_steps_default=config.MAX_STEPS)
    p_inf.add_argument("--infinite-seed", type=int, default=config.INFINITE_RUN_SEED)
    p_inf.add_argument("--num-seeds", type=int, default=1,
                       help="train across N seeds derived from --infinite-seed")
    p_inf.add_argument("--seeds", type=str, default=None,
                       help="explicit comma-separated sampler seeds (overrides --num-seeds)")
    p_inf.set_defaults(func=cmd_train_infinite)

    p_lvl = tsub.add_parser("levels", help="static hand-built levels")
    _add_common_train_args(p_lvl, max_steps_default=config.MAX_STEPS)
    p_lvl.add_argument("--levels", type=str, default=None,
                       help="comma-separated level names (default: all)")
    p_lvl.set_defaults(func=cmd_train_levels)

    p_maze = tsub.add_parser("maze", help="reverse spawn-curriculum on a hard level")
    _add_common_train_args(p_maze, max_steps_default=config.MAX_STEPS)
    p_maze.add_argument("--level", type=str, default="maze")
    p_maze.set_defaults(func=cmd_train_maze)

    p_gym = tsub.add_parser("gym", help="Completion Gym")
    _add_common_train_args(p_gym, max_steps_default=config.GYM_MAX_STEPS)
    p_gym.add_argument("--gym-seed", type=int, default=config.GYM_SEED)
    p_gym.add_argument("--num-seeds", type=int, default=config.GYM_DEFAULT_NUM_SEEDS,
                       help="train across N gym seeds derived from --gym-seed")
    p_gym.add_argument("--seeds", type=str, default=None,
                       help="explicit comma-separated gym seeds (overrides --num-seeds)")
    p_gym.add_argument("--abilities", type=str, default="double_jump",
                       help="comma-separated granted abilities; '' for single jump")
    p_gym.set_defaults(func=cmd_train_gym)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", cmd_play)  # no subcommand -> play
    return func(args)
