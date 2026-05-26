import pymunk
import pytest

from blueball import collision
from blueball.abilities import Ability
from blueball.agent import Action, Agent
from blueball.entities.player import Player
from blueball.world import World


class _Idle(Agent):
    def act(self, observation):
        return Action.IDLE


def _player_world():
    w = World()
    collision.register(w.space, world_ref=w)
    p = Player(agent=_Idle(), spawn_xy=(100, 100))
    w.add_entity(p)
    return w, p


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    return save_mod


def test_player_dies_on_spike_contact():
    w, p = _player_world()
    # Create a static spike directly under the player
    spike_body = w.space.static_body
    spike_shape = pymunk.Poly(spike_body, [(80, 130), (120, 130), (100, 110)])
    spike_shape.collision_type = collision.CT_SPIKE
    w.space.add(spike_shape)

    # Step until player falls into spike
    for _ in range(60):
        w.step(1 / 60)
        if p.dead:
            break
    assert p.dead


def test_goal_marks_level_complete():
    w, p = _player_world()
    goal_body = w.space.static_body
    goal_shape = pymunk.Poly(goal_body, [(80, 130), (120, 130), (120, 150), (80, 150)])
    goal_shape.sensor = True
    goal_shape.collision_type = collision.CT_GOAL
    w.space.add(goal_shape)

    for _ in range(60):
        w.step(1 / 60)
        if w.level_complete:
            break
    assert w.level_complete


def test_player_unlocks_ability_on_pickup_contact(tmp_save):
    from blueball.entities.ability_pickup import AbilityPickup
    w, p = _player_world()
    # Place the pickup directly on top of the player so contact is immediate.
    pickup = AbilityPickup(w, position=(100, 100), ability=Ability.DOUBLE_JUMP, radius=20)
    w.add_entity(pickup)

    for _ in range(5):
        w.step(1 / 60)
        if pickup._collected:
            break
    assert Ability.DOUBLE_JUMP in p.abilities
    assert pickup._collected is True
    assert pickup.shapes[0] not in w.space.shapes
    assert tmp_save.load() == {"double_jump"}
