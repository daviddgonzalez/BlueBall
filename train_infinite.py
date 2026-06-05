"""Headless reference training on Infinite Run.

Runs the GA on the pinned reference seed (config.INFINITE_RUN_SEED) by default,
or across multiple seeds for generalization, and persists the result into a
timestamped run folder under genomes/. No pygame / no display required.

    python train_infinite.py                      # single reference seed
    python train_infinite.py --num-seeds 3        # base seed + 2 derived seeds
    python train_infinite.py --seeds 1234,777,9   # explicit seed set
    python train_infinite.py --gens 50            # override generations

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import generate_seeds, infinite_episodes
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
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="train across N seeds derived from --infinite-seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="explicit comma-separated sampler seeds (overrides --num-seeds)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = generate_seeds(args.infinite_seed, args.num_seeds)

    episodes = infinite_episodes(seeds, world_seed=args.world_seed,
                                 max_steps=args.max_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        infinite_seed=args.infinite_seed, world_seed=args.world_seed,
        timestamp=timestamp, num_seeds=len(seeds),
    )

    print(
        f"Training {args.pop}x{args.gens} on Infinite Run seeds={seeds} "
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
    print(f"Done. gen {final['gen']}: best={final['best']:.1f} mean={final['mean']:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
