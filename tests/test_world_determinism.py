import pickle
from pathlib import Path

import pymunk

from blueball import config
from blueball.world import World
from blueball.entities.player import Player
from blueball.agent import Action, Agent
from blueball.collision import register
from blueball.levels.loader import load_level
from blueball.levels.sampler import ChunkSampler


class _Scripted(Agent):
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, obs):
        a = self.actions[self.i] if self.i < len(self.actions) else Action.IDLE
        self.i += 1
        return a


def _run(level_source, actions, n_ticks=300):
    w = World(seed=1)
    register(w.space, world_ref=w)
    meta = load_level(level_source, w)
    p = Player(agent=_Scripted(actions), spawn_xy=tuple(meta.spawn))
    w.add_entity(p)
    for _ in range(n_ticks):
        w.step(1 / 60)
    return (p.body.position.x, p.body.position.y, p.body.velocity.x, p.body.velocity.y)


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


def test_speed_run_world_determinism():
    actions = [Action.RIGHT] * 600
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "speed_run.json"
    a = _run(path, actions)
    b = _run(path, actions)
    assert a == b


def test_substep_runs_exactly_one_fixed_substep():
    """substep() advances one PHYS_DT substep with no accumulator change."""
    import pymunk
    from blueball import config
    from blueball.world import World

    w = World()
    body = pymunk.Body(mass=1.0, moment=10.0)
    body.position = (0.0, 0.0)
    shape = pymunk.Circle(body, 5.0)
    w.space.add(body, shape)

    accum_before = w._accumulator
    w.substep()
    # One substep of gravity: velocity gains gravity_y * PHYS_DT.
    assert abs(body.velocity.y - config.GRAVITY[1] * config.PHYS_DT) < 1e-9
    # The accumulator must be untouched (substep bypasses it).
    assert w._accumulator == accum_before


def test_substep_calls_entity_update_once():
    """Each substep runs exactly one entity update pass."""
    from blueball.world import World

    class Counter:
        bodies = ()
        shapes = ()
        constraints = ()
        def __init__(self):
            self.n = 0
        def update(self, dt):
            self.n += 1
            self.last_dt = dt

    w = World()
    c = Counter()
    w.add_entity(c)
    w.substep()
    w.substep()
    assert c.n == 2
    from blueball import config
    assert c.last_dt == config.PHYS_DT


def test_sampler_level_world_determinism():
    actions = [Action.RIGHT] * 600
    seq1 = list(ChunkSampler(seed=12345, target_chunks=80))
    seq2 = list(ChunkSampler(seed=12345, target_chunks=80))
    data = {
        "name": "Det", "background": "#000000", "ground": "#111111",
        "spawn": [80, 540], "chunks": seq1,
    }
    a = _run(data, actions)
    data2 = {**data, "chunks": seq2}
    b = _run(data2, actions)
    assert a == b
