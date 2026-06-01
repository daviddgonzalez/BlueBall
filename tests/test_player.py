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


def test_player_receive_boost_kicks_in_pad_direction_right():
    """A right-arrow pad launches you rightward (even from a standstill): the
    instant kick follows the pad's arrow, not the player's current motion."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0.0, 50.0)
    p.body.angular_velocity = 0.0
    p.receive_boost(2.0, direction=1.0)
    cap = config.MAX_LINEAR_SPEED * 2.0
    cap_ang = config.MAX_ANGULAR_VEL * 2.0
    assert p.body.velocity.x == cap * config.BOOST_PAD_KICK_FACTOR
    assert p.body.velocity.x > 0
    assert p.body.velocity.y == 50.0  # vy preserved
    assert p.body.angular_velocity == cap_ang * config.BOOST_PAD_KICK_FACTOR


def test_player_receive_boost_kicks_in_pad_direction_left():
    """A left-arrow pad launches you leftward even while you're moving right."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (300.0, 50.0)
    p.body.angular_velocity = 18.0
    p.receive_boost(2.0, direction=-1.0)
    # Kicked toward the leftward cap: redirected left (vx and ang both drop).
    assert p.body.velocity.x < 300.0
    assert p.body.angular_velocity < 18.0
    assert p.body.velocity.y == 50.0


def test_player_receive_boost_launches_stationary_player_along_arrow():
    """A directional pad launches even a stationary player along its arrow."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0.0, 50.0)
    p.body.angular_velocity = 0.0
    p.receive_boost(2.0, direction=1.0)
    assert p.body.velocity.x > 0.0  # kicked rightward from standstill
    assert p.body.velocity.y == 50.0
    assert p._boost_multiplier == 2.0


def test_player_receive_boost_does_not_slow_already_fast_player():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    overshoot = config.MAX_LINEAR_SPEED * 3.0
    p.body.velocity = (overshoot, 0.0)
    # Right-arrow pad while already faster-than-cap rightward: kick must not slow.
    p.receive_boost(2.0, direction=1.0)
    assert p.body.velocity.x == overshoot


def test_player_unlocking_double_jump_mid_air_grants_immediate_extra_jump():
    """End-to-end: Player takes a ground jump, picks up DOUBLE_JUMP mid-air,
    next jump press immediately fires the air jump. Reproduces the bug where
    the pickup appeared to do nothing until landing."""
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    # Simulate: ground jump → airborne tick (no ability yet, so counter at 0)
    p.jump_ctrl.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    p.jump_ctrl.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Confirm the bug shape: pressing jump in air now would NOT fire
    d = p.jump_ctrl.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False
    p.jump_ctrl.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Pickup happens mid-air
    p.unlock(Ability.DOUBLE_JUMP)
    # Next fresh airborne press should fire the air jump immediately
    d = p.jump_ctrl.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_player_starts_with_no_keys_and_no_respawn():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.keys_held == 0
    assert p.respawn_xy is None
    assert p.has_key(0) is False
    assert p.has_key(5) is False


def test_player_collect_key_sets_bit_and_is_idempotent():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.collect_key(3)
    assert p.has_key(3) is True
    assert p.keys_held == (1 << 3)
    p.collect_key(3)  # idempotent
    assert p.keys_held == (1 << 3)
    p.collect_key(0)
    assert p.has_key(0) is True
    assert p.keys_held == (1 << 3) | (1 << 0)


def test_player_receive_spring_applies_upward_impulse():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0, 0)
    p.receive_spring(impulse=400.0)
    # pymunk y-down: upward velocity is negative
    # Player mass is 1.0; impulse = 400 * 1.0 = 400 => delta-v = -400 y
    assert p.body.velocity.y == -400.0


def test_player_receive_spring_is_consistent_regardless_of_incoming_velocity():
    """The spring sets a floor launch speed rather than adding an impulse, so the
    bounce is the same whether you fall fast onto it or walk on. An already-faster
    upward motion is preserved (the spring never slows a strong rise)."""
    launch = 600.0
    # Falling fast onto the spring.
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0, 500)
    p.receive_spring(impulse=launch)
    assert p.body.velocity.y == -launch
    # Walking on with ~zero vertical speed.
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0, 0)
    p.receive_spring(impulse=launch)
    assert p.body.velocity.y == -launch
    # Already rising faster than the launch -> keep the faster rise.
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0, -900)
    p.receive_spring(impulse=launch)
    assert p.body.velocity.y == -900


def test_add_entity_wires_world_reference():
    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    assert p._world is w


def test_player_ray_filter_excludes_own_shape():
    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    hit = w.space.segment_query_first(
        (100, 100), (200, 100), 0.5, p._ray_filter,
    )
    assert hit is None  # filter excluded our own shape; nothing else in world


def test_observe_ray_hits_wall():
    """A wall placed to the right of the player should register as a GROUND hit
    with alpha < 1.0 on the due-right ray (index 0)."""
    import numpy as np
    from blueball.agent import HitType
    w = World()
    # Vertical wall segment to the right of spawn
    static = w.space.static_body
    wall = pymunk.Segment(static, (300, 50), (300, 550), 5)
    wall.friction = 1.0
    w.space.add(wall)
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 300))
    w.add_entity(p)
    obs = p._observe()
    # ray 0 goes due right; wall is 200 px away, MAX_RAY_LEN = 300 → alpha ≈ 200/300 ≈ 0.667
    assert obs.rays[0] < 1.0, "Ray 0 should hit the wall (alpha < 1.0)"
    assert obs.ray_hit_types[0] == HitType.GROUND


def test_nearest_pickup_finds_collectible():
    """_observe() nearest_pickup returns a delta toward a Collectible entity."""
    import numpy as np

    class _FakeCollectible:
        """Minimal stand-in for a Collectible with a body position."""
        def __init__(self, x, y):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            self.body = body

    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    c = _FakeCollectible(200, 100)
    # Register entity with a matching type name via a subclass trick
    _FakeCollectible.__name__ = "Collectible"
    w.entities.append(c)
    obs = p._observe()
    assert obs.nearest_pickup is not None, "nearest_pickup should find the Collectible"
    dx, dy = obs.nearest_pickup
    assert abs(dx - 100.0) < 1e-3
    assert abs(dy - 0.0) < 1e-3


def test_abilities_bitfield_double_jump_is_bit_0():
    """When DOUBLE_JUMP (first Ability enum member) is in the set, bit 0 is set."""
    p = Player(
        agent=_ScriptedAgent([Action.IDLE]),
        spawn_xy=(100, 100),
        abilities={Ability.DOUBLE_JUMP},
    )
    obs = p._observe()
    assert obs.abilities & 1, "bit 0 should be set when DOUBLE_JUMP is unlocked"


def test_observation_has_enriched_fields():
    import numpy as np
    from blueball.agent import HitType, Observation
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    obs = p._observe()
    assert obs.rays.shape == (8,)
    assert obs.ray_hit_types.shape == (8,)
    assert obs.ray_hit_types.dtype == np.int8
    assert obs.abilities == 0
    assert obs.keys_held == 0
    assert obs.nearest_hazard is None
    assert obs.nearest_pickup is None
    # HitType enum complete
    assert HitType.MISS == 0
    assert HitType.GROUND == 1
    assert HitType.HAZARD == 2
    assert HitType.PICKUP == 3
    assert HitType.GOAL == 4
    assert HitType.ENEMY == 5
    assert HitType.BLOCK == 6
    assert HitType.DOOR == 7
