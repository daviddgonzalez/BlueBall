"""Track D: granted abilities thread through the Infinite Run training path.

`evaluate_infinite` must hand the granted ability set to BOTH:
  - the ``TerrainStream`` chunk sampler, so double-jump-gated chunks
    (``double_gap`` / ``double_ledge`` / ``double_step``) become eligible, and
  - the ``Player``, so the mover can actually perform the maneuver.

Backward-compat: the legacy 5-tuple call and a default ``EpisodeSpec.abilities=()``
stay single-jump and byte-identical to pre-Track-D behaviour.
"""
import numpy as np

from blueball.abilities import Ability
from blueball.ai import trainer as tr
from blueball.ai.episodes import EpisodeSpec
from blueball.ai.ftnn import GENOME_SIZE
from blueball.ai.genome import random_genome


def _genome():
    return random_genome(np.random.default_rng(0))


def test_evaluate_infinite_5tuple_back_compat_equals_empty_abilities():
    """Legacy 5-tuple == explicit empty-abilities 6-tuple: both single jump."""
    g = _genome()
    _, f5 = tr.evaluate_infinite((0, g, 1234, 1, 80))
    _, f6 = tr.evaluate_infinite((0, g, 1234, 1, 80, ()))
    assert f5 == f6


def test_evaluate_infinite_threads_abilities_to_terrain_and_player(monkeypatch):
    captured = {}
    real_terrain, real_player = tr.TerrainStream, tr.Player

    def spy_terrain(world, seed, **kw):
        captured["terrain"] = kw.get("abilities", frozenset())
        return real_terrain(world, seed, **kw)

    def spy_player(*a, **kw):
        captured["player"] = kw.get("abilities")
        return real_player(*a, **kw)

    monkeypatch.setattr(tr, "TerrainStream", spy_terrain)
    monkeypatch.setattr(tr, "Player", spy_player)

    tr.evaluate_infinite((0, _genome(), 1234, 1, 30, ("double_jump",)))

    assert Ability.DOUBLE_JUMP in captured["terrain"]
    assert Ability.DOUBLE_JUMP in captured["player"]


def test_evaluate_episodes_infinite_honors_episode_abilities(monkeypatch):
    """An infinite EpisodeSpec carrying abilities reaches the terrain sampler."""
    captured = {}
    real_terrain = tr.TerrainStream

    def spy_terrain(world, seed, **kw):
        captured["terrain"] = kw.get("abilities", frozenset())
        return real_terrain(world, seed, **kw)

    monkeypatch.setattr(tr, "TerrainStream", spy_terrain)

    ep = EpisodeSpec(kind="infinite", seed=1234, level_path=None,
                     world_seed=1, max_steps=30, abilities=("double_jump",))
    tr.evaluate_episodes((0, _genome(), [ep], 0.0, "min"))

    assert Ability.DOUBLE_JUMP in captured["terrain"]


def test_train_infinite_cli_grants_double_jump_by_default():
    """The global invariant: `train infinite` grants double-jump unless opted out."""
    from blueball.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["train", "infinite"]).abilities == "double_jump"
    assert parser.parse_args(["train", "infinite", "--abilities", ""]).abilities == ""
