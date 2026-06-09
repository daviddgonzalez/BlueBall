import numpy as np

from blueball import config
from blueball.ai.episodes import gym_episodes
from blueball.ai.persistence import run_dir_name
from blueball.ai.trainer import train


def test_run_dir_name_gym_key():
    assert run_dir_name(gym_seed=7, world_seed=1, timestamp="T",
                        num_seeds=1).startswith("gym7_w1_")
    assert run_dir_name(gym_seed=7, world_seed=1, timestamp="T",
                        num_seeds=3).startswith("gym7x3_w1_")


def test_gym_training_runs_and_is_reproducible():
    eps = gym_episodes([7], world_seed=config.DEFAULT_SEED, max_steps=400,
                       abilities=("double_jump",))
    r1 = train(pop_size=4, generations=2, episodes=eps, ga_seed=0,
               world_seed=config.DEFAULT_SEED, map_fn=map)
    r2 = train(pop_size=4, generations=2, episodes=eps, ga_seed=0,
               world_seed=config.DEFAULT_SEED, map_fn=map)
    assert np.array_equal(r1.best_genome, r2.best_genome)
    assert np.isfinite(r1.history[-1]["best"])
