"""Tests for the Cannon emitter and its sin^2 Projectiles."""

import math

from blueball.entities.cannon import Cannon
from blueball.entities.projectile import Projectile
from blueball.world import World


def test_projectile_velocity_never_backward():
    """Speed follows V*sin^2, so vx keeps the firing sign for the whole flight
    (it may dip to 0 at troughs but never reverses)."""
    world = World()
    proj = Projectile(world, position=(100, 300), direction=1.0, speed=200.0,
                      pulse_period_s=0.5, max_travel=1e9)
    world.add_entity(proj)
    saw_positive = False
    for _ in range(2000):
        proj.update(1 / 120)
        assert proj.body.velocity.x >= 0.0
        if proj.body.velocity.x > 0.0:
            saw_positive = True
    assert saw_positive  # it actually moves, not just sits at 0

    # Leftward projectile is the mirror image: vx never positive.
    proj_l = Projectile(world, position=(100, 300), direction=-1.0, speed=200.0,
                        pulse_period_s=0.5, max_travel=1e9)
    world.add_entity(proj_l)
    for _ in range(2000):
        proj_l.update(1 / 120)
        assert proj_l.body.velocity.x <= 0.0


def test_projectile_despawns_after_max_travel():
    # world.step integrates the kinematic body (space.step) then runs entity
    # updates — the same order the game loop uses, so the projectile actually
    # moves and eventually exceeds max_travel.
    world = World()
    proj = Projectile(world, position=(100, 300), direction=1.0, speed=240.0,
                      pulse_period_s=0.4, max_travel=120.0)
    world.add_entity(proj)
    for _ in range(1200):
        world.step(1 / 60)
        if not proj.alive:
            break
    assert not proj.alive
    assert proj.shape not in world.space.shapes
    assert proj.body not in world.space.bodies
    assert abs(proj.body.position.x - 100) >= 120.0


def test_cannon_fires_on_interval():
    world = World()
    cannon = Cannon(world, position=(98, 300), direction="right", interval_s=1.0)
    world.add_entity(cannon)
    before = sum(1 for e in world.entities if isinstance(e, Projectile))
    # Advance just under one interval — no shot yet.
    cannon.update(0.9)
    assert sum(1 for e in world.entities if isinstance(e, Projectile)) == before
    # Cross the interval — exactly one shot.
    cannon.update(0.2)
    assert sum(1 for e in world.entities if isinstance(e, Projectile)) == before + 1


def test_cannon_direction_validation():
    world = World()
    try:
        Cannon(world, position=(0, 0), direction="up")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad direction")


def test_loader_parses_cannons_block():
    from blueball.levels.loader import load_level
    world = World()
    data = {
        "name": "T", "background": "#000000", "ground": "#111111",
        "spawn": [80, 540],
        "cannons": [
            {"x": 98, "y_offset": 200, "dir": "right", "interval_s": 1.5},
            {"x": 286, "y_offset": 400, "dir": "left"},
        ],
        "chunks": [{"type": "flat", "width_tiles": 3}, {"type": "goal"}],
    }
    load_level(data, world)
    cannons = [e for e in world.entities if isinstance(e, Cannon)]
    assert len(cannons) == 2
    assert {c.direction for c in cannons} == {"left", "right"}
