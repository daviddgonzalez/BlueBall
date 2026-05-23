import pytest

from blueball.entities.base import Entity
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
