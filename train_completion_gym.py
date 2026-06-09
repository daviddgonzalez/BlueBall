"""Headless training on the Completion Gym — an endless chain of solvable,
goal-terminated segments (keys, doors, boxes, lava) with difficulty ramping by
depth. Trains the completion mechanics that Infinite Run never exercises.

    python train_completion_gym.py                    # default multi-seed run
    python train_completion_gym.py --num-seeds 16     # more chains -> generalize
    python train_completion_gym.py --seeds 3,7,11     # explicit gym seeds
    python train_completion_gym.py --abilities ''     # single-jump gym (tier 0/1)
    python train_completion_gym.py --gens 80

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.episodes import generate_seeds, gym_episodes
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name
from blueball.ai.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.GYM_MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--gym-seed", type=int, default=config.GYM_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--num-seeds", type=int, default=config.GYM_DEFAULT_NUM_SEEDS,
                        help="train across N gym seeds derived from --gym-seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="explicit comma-separated gym seeds (overrides --num-seeds)")
    parser.add_argument("--abilities", type=str, default="double_jump",
                        help="comma-separated granted abilities; '' for single jump")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = generate_seeds(args.gym_seed, args.num_seeds)

    abilities = tuple(a.strip() for a in args.abilities.split(",") if a.strip())

    episodes = gym_episodes(seeds, world_seed=args.world_seed,
                            max_steps=args.max_steps, abilities=abilities)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        gym_seed=seeds[0], world_seed=args.world_seed,
        timestamp=timestamp, num_seeds=len(seeds),
    )

    print(
        f"Training {args.pop}x{args.gens} on Completion Gym seeds={seeds} "
        f"abilities={abilities or '(single jump)'} world={args.world_seed} ga={args.ga_seed}\n  -> {run_dir}"
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
