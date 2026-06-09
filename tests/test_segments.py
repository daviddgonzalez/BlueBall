from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import GoalSegment, KeyDoorGoalSegment


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
