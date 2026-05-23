import pymunk
import pytest

from blueball import collision
from blueball.entities.base import Entity
from blueball.entities.collectible import Collectible
from blueball.entities.falling_hazard import FallingHazard
from blueball.entities.goal import Goal
from blueball.entities.patroller import Patroller
from blueball.entities.spike import Spike
from blueball.world import World


class _BodyStub:
    def __init__(self, x: float) -> None:
        self.position = pymunk.Vec2d(x, 0)


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


def test_patroller_reverses_at_right_bound():
    w = World()
    p = Patroller(w, position=(100, 500), left_bound=80, right_bound=120, speed=60.0)
    w.add_entity(p)
    # Run forward in time; the patroller should hit the right bound and come back
    seen_max_x = 100.0
    for _ in range(120):  # ~2 seconds
        w.step(1 / 60)
        seen_max_x = max(seen_max_x, p.body.position.x)
    assert seen_max_x >= 119  # got close to right bound
    assert p.body.position.x < 119  # reversed back below it


def test_patroller_die_removes_from_space():
    w = World()
    p = Patroller(w, position=(100, 500), left_bound=80, right_bound=120)
    w.add_entity(p)
    p.die()
    assert p.alive is False
    assert p.body not in w.space.bodies


def test_falling_hazard_does_not_fall_before_trigger():
    w = World()
    player_body = _BodyStub(x=0)
    fh = FallingHazard(w, position=(500, 100), trigger_x=400, player_provider=lambda: player_body)
    w.add_entity(fh)
    y0 = fh.body.position.y
    for _ in range(30):
        w.step(1 / 60)
    assert fh.body.position.y == y0


def test_falling_hazard_falls_after_trigger():
    w = World()
    player_body = _BodyStub(x=500)  # already past trigger_x
    fh = FallingHazard(w, position=(500, 100), trigger_x=400, player_provider=lambda: player_body)
    w.add_entity(fh)
    y0 = fh.body.position.y
    for _ in range(30):
        w.step(1 / 60)
    assert fh.body.position.y > y0  # fell under gravity
