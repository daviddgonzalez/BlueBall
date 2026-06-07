"""Headless reverse spawn-curriculum training on a single hard level (maze).

Spawns the agent near the goal first and recedes the spawn toward the true start
as the population masters each stage (see ai/curriculum.py). Persists into a
timestamped run folder under genomes/. No pygame / no display required.

    python train_maze_curriculum.py                 # maze, spec defaults
    python train_maze_curriculum.py --gens 120      # override generations
    python train_maze_curriculum.py --level maze --pop 80 --gens 200

For parallel evaluation, this script uses multiprocessing.Pool by default.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

from blueball import config
from blueball.ai.curriculum import (build_box_lava_curriculum,
                                    build_spawn_curriculum, evaluate_curriculum,
                                    train_curriculum)
from blueball.ai.episodes import resolve_level_paths
from blueball.ai.persistence import GENOMES_ROOT, run_dir_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="maze")
    parser.add_argument("--pop", type=int, default=config.TRAIN_POP_SIZE)
    parser.add_argument("--gens", type=int, default=config.TRAIN_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--ga-seed", type=int, default=config.GA_SEED)
    parser.add_argument("--world-seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--box-lava", action="store_true",
                        help="train a box-lava specialist: single fixed stage "
                             "spawned just left of the PushableBox, all keys "
                             "granted (writes a mazeboxlavacurr_* run dir).")
    args = parser.parse_args()

    try:
        level_path = resolve_level_paths([args.level])[0]
    except ValueError as e:
        raise SystemExit(str(e))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # --box-lava trains a single fixed box-lava stage (run dir mazeboxlavacurr_*);
    # otherwise the full reverse spawn-curriculum (run dir mazecurr_*).
    if args.box_lava:
        stages = build_box_lava_curriculum(level_path)
        level_name = "mazeboxlava"
    else:
        stages = build_spawn_curriculum(level_path)
        level_name = args.level

    run_dir = Path(GENOMES_ROOT) / run_dir_name(
        world_seed=args.world_seed, timestamp=timestamp,
        level_name=level_name, curriculum=True,
    )
    print(
        f"Curriculum training {args.pop}x{args.gens} on {level_name} "
        f"({len(stages)} stages) world={args.world_seed} ga={args.ga_seed}\n"
        f"  -> {run_dir}"
    )

    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    try:
        result = train_curriculum(
            level_path=level_path,
            pop_size=args.pop,
            generations=args.gens,
            ga_seed=args.ga_seed,
            world_seed=args.world_seed,
            max_steps=args.max_steps,
            map_fn=pool.imap if pool is not None else map,
            save_dir=run_dir,
            stages=stages,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Verdict: re-evaluate the final best genome from the last stage's spawn
    # (the level's true start for the reverse curriculum; the box-lava spawn
    # for --box-lava). No new keys beyond that stage's granted scaffolding.
    start = stages[-1]
    _, fit, reached = evaluate_curriculum(
        (0, result.best_genome, args.world_seed, level_path, args.max_steps,
         start.spawn_xy, start.granted_keys))

    final = result.history[-1]
    print(f"Done. gen {final['gen']}: final stage '{final['stage_label']}'")
    print(f"Verdict @ '{start.label}': reached_goal={reached} fitness={fit:.1f}")
    print(f"Best genome + history written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
