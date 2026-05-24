import pymunk

from blueball import config
from blueball.agent import Action, Agent, Observation
from blueball.entities.player import Player
from blueball.world import World


class _ScriptedAgent(Agent):
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, observation):
        a = self.actions[self.i] if self.i < len(self.actions) else Action.IDLE
        self.i += 1
        return a


def _make_world_with_floor():
    w = World()
    # A static floor body
    static = w.space.static_body
    floor = pymunk.Segment(static, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    return w


def test_player_construct():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert len(p.bodies) == 1
    assert len(p.shapes) == 1
    assert p.body.position.x == 100
    assert p.body.position.y == 100
    assert p.body.mass == config.BALL_MASS


def test_right_press_spins_ball_clockwise():
    w = _make_world_with_floor()
    # Spawn just above the floor so the ball lands almost immediately.
    # AIR_CONTROL is 0 so torque only applies once grounded - if we spawned
    # high in the air the ball would float through the whole test without
    # ever spinning.
    p = Player(agent=_ScriptedAgent([Action.RIGHT] * 60), spawn_xy=(100, 580))
    w.add_entity(p)
    for _ in range(30):
        w.step(1 / 60)
    assert p.body.angular_velocity > 0


def test_jump_from_grounded_produces_upward_velocity():
    w = _make_world_with_floor()
    # Settle with an idle agent first; the agent is consulted once per substep
    p = Player(agent=_ScriptedAgent([Action.IDLE] * 200), spawn_xy=(100, 580))
    w.add_entity(p)
    for _ in range(20):
        w.step(1 / 60)
    assert p.grounded
    # Swap to a jump-pressing agent and step once
    p.agent = _ScriptedAgent([Action.JUMP] * 10)
    w.step(1 / 60)
    # In pymunk y-down convention, up is negative y
    assert p.body.velocity.y < -100


def test_die_flips_alive_flag():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.die()
    assert p.dead is True
    assert p.alive is False
