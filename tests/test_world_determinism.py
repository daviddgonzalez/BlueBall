import pickle
import pymunk

from blueball import config
from blueball.world import World


def _add_test_ball(world: World, x: float, y: float) -> pymunk.Body:
    """Add a unit circle body so the physics loop has something to step."""
    body = pymunk.Body(
        mass=config.BALL_MASS,
        moment=pymunk.moment_for_circle(config.BALL_MASS, 0, config.BALL_RADIUS),
    )
    body.position = (x, y)
    shape = pymunk.Circle(body, config.BALL_RADIUS)
    shape.friction = config.BALL_FRICTION
    world.space.add(body, shape)
    return body


def test_two_worlds_with_same_seed_diverge_zero():
    a = World(seed=config.DEFAULT_SEED)
    b = World(seed=config.DEFAULT_SEED)
    ball_a = _add_test_ball(a, 100, 100)
    ball_b = _add_test_ball(b, 100, 100)

    for _ in range(300):  # ~5 seconds at PHYS_HZ
        a.step(1 / 60)
        b.step(1 / 60)

    assert ball_a.position.x == ball_b.position.x
    assert ball_a.position.y == ball_b.position.y
    assert ball_a.angle == ball_b.angle


def test_fixed_substep_count():
    """A 1/60 frame at PHYS_HZ=120 should run exactly 2 substeps."""
    w = World()
    body = _add_test_ball(w, 100, 100)
    substeps = w.step(1 / 60)
    assert substeps == 2


def test_accumulator_carries_remainder():
    """An odd frame time leaves remainder in the accumulator."""
    w = World()
    body = _add_test_ball(w, 100, 100)
    # 1/100s frame at PHYS_HZ=120: 0.01s / 0.0083... = 1.2 -> 1 substep, remainder 0.00167s
    n1 = w.step(1 / 100)
    n2 = w.step(1 / 100)
    # Two consecutive 1/100s frames sum to 0.02s = 2.4 substeps total
    assert n1 + n2 == 2
    # Third frame should now run 1 substep with leftover
    n3 = w.step(1 / 100)
    assert n3 == 1


def test_spiral_of_death_guard():
    """A huge frame_dt drops surplus rather than running unbounded substeps."""
    w = World()
    body = _add_test_ball(w, 100, 100)
    substeps = w.step(10.0)  # would be 1200 substeps without the guard
    assert substeps == config.MAX_ACCUMULATED_STEPS
