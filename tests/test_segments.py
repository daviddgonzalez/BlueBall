from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import GoalSegment, KeyDoorGoalSegment
from blueball.abilities import Ability
from blueball.levels.segments import (
    BoxLavaSegment, KeyDoorBoxLavaSegment, SEGMENT_TEMPLATES,
)


class _DelayedJumpAgent(Agent):
    """Rolls right to build speed, then spams RIGHT_JUMP — the strongest vault
    attempt (full jump + air-jump arc launched near the pit edge)."""
    def __init__(self, delay: int) -> None:
        self._delay = delay
        self._t = 0

    def act(self, observation):
        self._t += 1
        return Action.RIGHT if self._t <= self._delay else Action.RIGHT_JUMP


def _remove_entity(world, entity):
    if entity in world.entities:
        world.entities.remove(entity)
    for shp in list(getattr(entity, "shapes", [])):
        if shp in world.space.shapes:
            world.space.remove(shp)
        world._shape_to_entity.pop(shp, None)
    for bod in list(getattr(entity, "bodies", [])):
        if bod in world.space.bodies:
            world.space.remove(bod)


def _fresh_world():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    return w


def _names(world):
    return [type(e).__name__ for e in world.entities]


class _RightAgent(Agent):
    """Always rolls right — solves any flat, jump-free segment."""
    def act(self, observation):
        return Action.RIGHT


def _rolls_to_goal(world, player, max_steps=3000):
    for _ in range(max_steps):
        world.substep()
        if player.reached_goal:
            return True
    return False


def test_goal_segment_builds_a_goal_with_positive_width():
    w = _fresh_world()
    width = GoalSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)
    assert GoalSegment.tier == 0
    assert GoalSegment.min_abilities == frozenset()


def test_keydoorgoal_builds_key_left_of_door_plus_goal():
    w = _fresh_world()
    width = KeyDoorGoalSegment().build(w, x_offset=0.0)
    names = _names(w)
    assert names.count("Key") == 1 and names.count("Door") == 1 and "Goal" in names
    key = next(e for e in w.entities if type(e).__name__ == "Key")
    door = next(e for e in w.entities if type(e).__name__ == "Door")
    assert key.position[0] < door.position[0]
    assert width > 0


def test_goal_segment_is_solvable_by_rolling_right():
    w = _fresh_world()
    GoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    assert _rolls_to_goal(w, p)


def test_keydoorgoal_is_solvable_by_rolling_right():
    w = _fresh_world()
    KeyDoorGoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    # Rolling right collects the low key, opens the door, reaches the goal.
    assert _rolls_to_goal(w, p)


def test_boxlava_segment_composition_and_requirements():
    w = _fresh_world()
    width = BoxLavaSegment(pit_tiles=6).build(w, x_offset=0.0)
    names = _names(w)
    assert "Lava" in names and "PushableBox" in names and "Goal" in names
    assert BoxLavaSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoxLavaSegment.min_abilities
    assert width > 0


def test_boxlava_random_varies_pit_width():
    import random
    widths = {BoxLavaSegment.random(random.Random(s)).pit_tiles for s in range(40)}
    assert len(widths) > 1  # not constant
    assert all(20 <= w <= 24 for w in widths)


def test_tier3_combo_composition():
    w = _fresh_world()
    KeyDoorBoxLavaSegment().build(w, x_offset=0.0)
    names = _names(w)
    for kind in ("Key", "Door", "Lava", "PushableBox", "Goal"):
        assert kind in names, kind
    assert KeyDoorBoxLavaSegment.tier == 3
    assert Ability.DOUBLE_JUMP in KeyDoorBoxLavaSegment.min_abilities


def test_all_four_tiers_registered():
    assert {t.tier for t in SEGMENT_TEMPLATES} == {0, 1, 2, 3}


def test_boxlava_pit_requires_the_box_not_vaultable():
    # The box must be NECESSARY: with the box removed, a DOUBLE_JUMP agent
    # cannot clear the pit by jumping. Guards against a narrow (vaultable) pit
    # that would train a jump instead of the box-push. Tests the low end (20)
    # of the gym's pit range across many jump-timing phases.
    for jump_delay in range(0, 64, 4):
        w = _fresh_world()
        BoxLavaSegment(pit_tiles=20).build(w, x_offset=0.0)
        box = next(e for e in w.entities if type(e).__name__ == "PushableBox")
        _remove_entity(w, box)
        p = Player(agent=_DelayedJumpAgent(jump_delay),
                   spawn_xy=(40.0, GROUND_Y - 30.0),
                   abilities={Ability.DOUBLE_JUMP})
        w.add_entity(p)
        reached = False
        for _ in range(1500):
            w.substep()
            if p.reached_goal:
                reached = True
                break
            if p.dead:
                break
        assert not reached, f"vaulted the box-removed pit at jump_delay={jump_delay}"
