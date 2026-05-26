import pymunk

from blueball import config
from blueball.abilities import Ability
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


def test_player_default_abilities_is_empty():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.abilities == set()
    assert p.jump_ctrl.abilities is p.abilities  # shared by reference


def test_player_constructed_with_abilities_propagates_to_jump_controller():
    p = Player(
        agent=_ScriptedAgent([Action.IDLE]),
        spawn_xy=(100, 100),
        abilities={Ability.DOUBLE_JUMP},
    )
    assert Ability.DOUBLE_JUMP in p.abilities
    assert Ability.DOUBLE_JUMP in p.jump_ctrl.abilities


def test_player_unlock_adds_to_set_and_propagates(monkeypatch, tmp_path):
    """unlock() is in-memory only — no save write. Persistence is the
    PlayScene's job at level-complete time."""
    save_file = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_file))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)

    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.unlock(Ability.DOUBLE_JUMP)
    assert Ability.DOUBLE_JUMP in p.abilities
    assert Ability.DOUBLE_JUMP in p.jump_ctrl.abilities
    # No save file should have been created — unlock is in-memory only.
    assert not save_file.exists()
    assert save_mod.load() == set()


def test_player_unlock_repeat_is_safe(monkeypatch, tmp_path):
    """Re-unlocking an already-unlocked ability is a no-op (set semantics)."""
    save_file = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_file))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)

    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.unlock(Ability.DOUBLE_JUMP)
    p.unlock(Ability.DOUBLE_JUMP)
    assert p.abilities == {Ability.DOUBLE_JUMP}
    assert not save_file.exists()


def test_player_receive_boost_raises_multiplier():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p._boost_multiplier == 1.0
    assert p._aerial_since_pickup is False
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0


def test_player_receive_boost_takes_max():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.receive_boost(1.5)
    assert p._boost_multiplier == 1.5
    p.receive_boost(1.2)
    assert p._boost_multiplier == 1.5  # weaker pad is a no-op
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0  # stronger pad raises


def test_player_boost_clears_on_air_to_ground_transition():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    # Pick up boost while airborne (no contact normals -> not grounded)
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0
    assert p._aerial_since_pickup is True
    # Stay airborne for a tick — boost persists
    p._update_boost(grounded=False)
    assert p._boost_multiplier == 2.0
    assert p._aerial_since_pickup is True
    # First grounded tick after airborne — boost clears
    p._update_boost(grounded=True)
    assert p._boost_multiplier == 1.0
    assert p._aerial_since_pickup is False


def test_player_boost_persists_while_grounded_until_jump_land_cycle():
    w = _make_world_with_floor()
    p = Player(agent=_ScriptedAgent([Action.IDLE] * 200), spawn_xy=(100, 580))
    w.add_entity(p)
    # Settle on the floor so the player is grounded
    for _ in range(20):
        w.step(1 / 60)
    assert p.grounded
    # Pick up boost while grounded — should NOT immediately expire
    p.receive_boost(2.0)
    assert p._boost_multiplier == 2.0
    assert p._aerial_since_pickup is False
    # Several grounded ticks — boost persists (no jump yet)
    for _ in range(5):
        p._update_boost(grounded=True)
    assert p._boost_multiplier == 2.0
    assert p._aerial_since_pickup is False
    # Now go airborne, then land — boost clears
    p._update_boost(grounded=False)
    assert p._aerial_since_pickup is True
    assert p._boost_multiplier == 2.0
    p._update_boost(grounded=True)
    assert p._boost_multiplier == 1.0
    assert p._aerial_since_pickup is False
