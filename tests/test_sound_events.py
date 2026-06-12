import pytest

from blueball.abilities import Ability
from blueball.agent import HumanAgent
from blueball.ai.episodes import resolve_level_paths
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.loader import load_level
from blueball.world import World


def test_world_emit_sound_queues():
    w = World()
    assert w.sound_events == []
    w.emit_sound("whoosh")
    assert w.sound_events == ["whoosh"]


def _emit_for(level_name, entity_type, sound):
    path = resolve_level_paths([level_name])[0]
    w = World()
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    player = Player(agent=HumanAgent(), spawn_xy=tuple(meta.spawn),
                    abilities={Ability.DOUBLE_JUMP})
    w.add_entity(player)
    ent = next(e for e in w.entities if type(e).__name__ == entity_type)
    epos = ent.position if hasattr(ent, "position") else (ent.body.position.x, ent.body.position.y)
    player.body.position = (float(epos[0]), float(epos[1]))
    w.substep()
    return w.sound_events


def test_goal_emits_fanfare():
    assert "fanfare" in _emit_for("tutorial_hill", "Goal", "fanfare")


def test_key_emits_key():
    assert "key" in _emit_for("maze", "Key", "key")


def test_spring_emits_spring():
    assert "spring" in _emit_for("lava_rising", "Spring", "spring")


def test_boost_pad_emits_whoosh():
    assert "whoosh" in _emit_for("maze", "BoostPad", "whoosh")
