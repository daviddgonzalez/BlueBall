import bisect

import numpy as np
import pytest

from blueball import config
from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.ai.ftnn import GENOME_SIZE, FTNN_INPUTS, FTNN_HIDDEN, FTNN_OUTPUTS
from blueball.levels.segment_stream import SegmentStream
from blueball.ai.trainer import evaluate_gym, evaluate_episodes
from blueball.ai.episodes import gym_episodes


class _RightAgent(Agent):
    def act(self, observation):
        return Action.RIGHT


def _drive(stream, world, player, max_steps):
    """The exact bookkeeping loop evaluate_gym runs, exposed for assertions."""
    max_x = player.body.position.x
    cleared = 0
    cumulative_keys = 0
    prev = 0
    for _ in range(max_steps):
        stream.maintain(player.body.position.x)
        world.substep()
        max_x = max(max_x, player.body.position.x)
        cur = bin(player.keys_held).count("1")
        if cur > prev:
            cumulative_keys += cur - prev
        prev = cur
        n = bisect.bisect_right(stream.segment_ends, max_x)
        if n > cleared:
            cleared = n
            player.keys_held = 0
            prev = 0
        if player.dead:
            break
    return cleared, cumulative_keys


def test_chain_continues_past_first_goal_and_clears_keys():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    granted = frozenset()  # single jump -> only roll-solvable tier 0/1 segments
    stream = SegmentStream(w, seed=3, granted_abilities=granted)
    p = Player(agent=_RightAgent(), spawn_xy=config.GYM_SPAWN, abilities=set(granted))
    w.add_entity(p)
    cleared, cumulative_keys = _drive(stream, w, p, max_steps=8000)
    assert cleared >= 2                       # did NOT stop at the first goal
    assert cumulative_keys >= 1               # collected >=1 key across the chain
    assert bin(p.keys_held).count("1") <= 1   # keys cleared between segments


def test_evaluate_gym_smoke_returns_finite_fitness():
    genome = np.random.default_rng(0).standard_normal(GENOME_SIZE).astype(np.float32)
    idx, f = evaluate_gym((0, genome, 11, config.DEFAULT_SEED, 500, ("double_jump",)))
    assert idx == 0
    assert np.isfinite(f)


def test_evaluate_episodes_dispatches_gym():
    genome = np.random.default_rng(1).standard_normal(GENOME_SIZE).astype(np.float32)
    eps = gym_episodes([11], world_seed=config.DEFAULT_SEED, max_steps=400,
                       abilities=("double_jump",))
    idx, score = evaluate_episodes((0, genome, tuple(eps), 1.0, "mean_std"))
    assert idx == 0 and np.isfinite(score)


def _always_right_genome():
    """A genome whose FTNN always argmaxes to Action.RIGHT (output index 2):
    all weights/biases zero except the RIGHT output bias = 1, so forward() returns
    [0,0,1,0,0,0] for any input."""
    g = np.zeros(GENOME_SIZE, dtype=np.float32)
    b2_start = FTNN_INPUTS * FTNN_HIDDEN + FTNN_HIDDEN + FTNN_HIDDEN * FTNN_OUTPUTS
    g[b2_start + int(Action.RIGHT)] = 1.0
    return g


def test_evaluate_gym_real_path_banks_the_segment_bonus():
    # Sanity: the crafted genome really emits RIGHT.
    from blueball.agent import FTNNAgent, Observation
    import numpy as _np
    agent = FTNNAgent(_always_right_genome())
    obs = Observation(rays=_np.ones(8, _np.float32), ray_hit_types=_np.zeros(8, _np.int8),
                      vel=_np.zeros(2, _np.float32), ang_vel=0.0, grounded=True,
                      nearest_pickup=None, nearest_hazard=None, abilities=0, keys_held=0)
    assert agent.act(obs) == Action.RIGHT
    # Through the REAL evaluate_gym over a single-jump (tier 0/1) gym, an
    # always-right roller clears segments and banks >= one segment bonus.
    # Seed 3 observed fitness=122048.0 >> GYM_SEGMENT_BONUS=1200.0.
    idx, f = evaluate_gym((0, _always_right_genome(), 3, config.DEFAULT_SEED, 8000, ()))
    assert idx == 0
    assert f > config.GYM_SEGMENT_BONUS  # at least one segment cleared -> bonus flowed
