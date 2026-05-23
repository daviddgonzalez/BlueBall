import pytest

from blueball import collision
from blueball.entities.base import Entity
from blueball.entities.collectible import Collectible
from blueball.entities.goal import Goal
from blueball.entities.spike import Spike
from blueball.world import World


class _StubEntity(Entity):
    def __init__(self):
        super().__init__()
        self.update_calls = 0

    def update(self, dt):
        self.update_calls += 1

    def draw(self, renderer, alpha):
        pass


def test_abstract_draw_required():
    class Bad(Entity):
        pass

    with pytest.raises(TypeError):
        Bad()


def test_entity_added_to_world_gets_update_called():
    w = World()
    e = _StubEntity()
    w.add_entity(e)
    w.step(1 / 60)  # runs 2 substeps at PHYS_HZ=120
    assert e.update_calls == 2


def test_spike_uses_correct_collision_type():
    w = World()
    s = Spike(w, position=(100, 200), width=32, height=24)
    w.add_entity(s)
    assert s.shapes[0].collision_type == collision.CT_SPIKE


def test_collectible_is_sensor_and_collects():
    w = World()
    c = Collectible(w, position=(50, 50))
    w.add_entity(c)
    assert c.shapes[0].sensor is True
    assert c.shapes[0].collision_type == collision.CT_COLLECTIBLE
    c.collect()
    assert c.alive is False
    assert c.shapes[0] not in w.space.shapes


def test_goal_is_sensor():
    w = World()
    g = Goal(w, position=(500, 100), width=40, height=80)
    w.add_entity(g)
    assert g.shapes[0].sensor is True
    assert g.shapes[0].collision_type == collision.CT_GOAL
