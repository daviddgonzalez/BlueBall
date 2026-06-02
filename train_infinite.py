"""Headless reference training on Infinite Run.

Runs the GA on the pinned reference seeds (config.GA_SEED + config.
INFINITE_RUN_SEED) and persists the result into a timestamped run folder
under genomes/. No pygame / no display required.

    python train_infinite.py                 # defaults from config
    python train_infinite.py --gens 50       # override generations

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--infinite-seed", type=int, default=config.INFINITE_RUN_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        infinite_seed=args.infinite_seed,
        world_seed=args.world_seed,
        timestamp=timestamp,
    )

    print(
        f"Training {args.pop}x{args.gens} on Infinite Run "
        f"seed={args.infinite_seed} world={args.world_seed} ga={args.ga_seed}\n"
        f"  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train(
            pop_size=args.pop,
            generations=args.gens,
            infinite_seed=args.infinite_seed,
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
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
