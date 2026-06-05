"""Headless training on the hand-built static levels.

Trains one generalist agent across a set of levels (default: all of them),
scoring each genome on every level and selecting on the per-level-normalized
mean - lam*std. Each level's fitness is divided by its 'par' so a big level
does not dominate selection. No pygame / no display required.

    python train_levels.py                              # all levels
    python train_levels.py --levels maze                # single level
    python train_levels.py --levels tutorial_hill,speed_run --gens 50

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import (available_levels, resolve_level_paths,
                                  static_episodes)
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--levels", type=str, default=None,
                        help="comma-separated level names (default: all)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.levels:
        names = [n.strip() for n in args.levels.split(",") if n.strip()]
    else:
        names = available_levels()
    try:
        level_paths = resolve_level_paths(names)
    except ValueError as e:
        raise SystemExit(str(e))

    episodes = static_episodes(level_paths, world_seed=args.world_seed,
                               max_steps=args.max_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        world_seed=args.world_seed, timestamp=timestamp,
        num_levels=len(level_paths),
    )

    print(
        f"Training {args.pop}x{args.gens} on levels={names} "
        f"world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train(
            pop_size=args.pop,
            generations=args.gens,
            episodes=episodes,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.3f} mean={final['mean']:.3f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
