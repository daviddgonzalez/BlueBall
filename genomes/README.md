# Trained genomes

GA-trained FTNN genomes, committed so golden agents travel with the repo.

Each training run writes one timestamped subfolder (never overwritten):

```
genomes/
  inf1234_w1_20260602-190345/
    run.json          # seeds, pop/gen counts, fitness history
    best_gen000.npy   # running best genome after each generation
    best_gen001.npy
    ...
    final_best.npy    # best genome of the whole run
```

Folder key: `inf<sampler_seed>` for Infinite Run (or the level name for a
static level), then `_w<world_seed>_<timestamp>`.

Produce a run with the pinned reference seeds:

```
python train_infinite.py            # config.GA_SEED + config.INFINITE_RUN_SEED
```

A genome is a flat float32 array of shape `(GENOME_SIZE,)`; load with
`numpy.load(...)` and pass to `blueball.agent.FTNNAgent` (or
`blueball.ai.ftnn.FTNN`). Keep committed runs intentional — prune
exploratory runs rather than committing every experiment.
</content>
