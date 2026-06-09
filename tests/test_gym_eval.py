import bisect

import numpy as np
import pytest

from blueball import config
from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.ai.ftnn import GENOME_SIZE
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
