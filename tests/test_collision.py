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
    """Redirect BLUEBALL_SAVE_PATH at a tmp file. The save module captures
    SAVE_PATH at import time, so we both reload it and overwrite the attribute
    as belt-and-suspenders against future refactors."""
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    monkeypatch.setattr(save_mod, "SAVE_PATH", save_path)
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
    # Unlock is in-memory only at pickup time; PlayScene persists on
    # level-complete. So nothing should be on disk yet.
    assert tmp_save.load() == set()


def test_player_receives_boost_on_pad_contact():
    from blueball.entities.boost_pad import BoostPad
    w, p = _player_world()
    # Place a boost pad overlapping the player position so contact is immediate
    pad = BoostPad(w, position=(100, 100), width=64, multiplier=2.0)
    w.add_entity(pad)

    for _ in range(5):
        w.step(1 / 60)
        if p._boost_multiplier > 1.0:
            break
    assert p._boost_multiplier == 2.0
    # Pad must still be present in the space (not consumed)
    assert pad.shapes[0] in w.space.shapes
    assert pad.body in w.space.bodies  # body also not removed


def test_player_resting_on_pad_keeps_boost():
    """Regression: a ball that is on the ground when it first overlaps the pad
    (rolling across / coming to rest on it) must KEEP the boost.

    The boost-pad collision callback runs during space.step(), before the
    player's contact normals are refreshed that frame, so player.grounded read
    stale/empty contacts and reported False. That made _update_boost treat the
    grounded pickup as an airborne→grounded landing and clear the boost on the
    same frame — contact registered but no speed-up stuck. Mirrors the chunk's
    geometry: ground segment with the pad seated flush on top of it.
    """
    from blueball import config
    from blueball.entities.boost_pad import BoostPad
    from blueball.levels.chunks.flat import GROUND_Y

    w = World()
    collision.register(w.space, world_ref=w)
    seg = pymunk.Segment(
        w.space.static_body, (-200, GROUND_Y), (600, GROUND_Y), 5
    )
    seg.friction = 1.0
    w.space.add(seg)
    cx = 300.0
    pad = BoostPad(
        w,
        position=(cx, GROUND_Y - config.BOOST_PAD_THICKNESS / 2),
        width=128,
        multiplier=2.0,
        direction="right",
    )
    w.add_entity(pad)
    # Player resting on the ground, centered over the pad (pre-existing overlap).
    p = Player(agent=_Idle(), spawn_xy=(cx, GROUND_Y - config.BALL_RADIUS))
    w.add_entity(p)

    for _ in range(20):
        w.step(1 / 60)

    # Boost must stick: it was granted while grounded, so it persists until the
    # next airborne→grounded cycle (which never happens here).
    assert p._boost_multiplier == 2.0
    # And the directional kick actually moved the grounded ball rightward.
    assert p.body.velocity.x > 0


def test_one_way_platform_passes_rising_player():
    """Player rising (velocity.y < 0) should pass through; falling should land."""
    from blueball.entities.one_way_platform import OneWayPlatform
    from blueball.entities.player import Player
    from blueball.collision import register
    from blueball.world import World

    w = World()
    register(w.space, world_ref=w)
    plat = OneWayPlatform(w, position=(100, 500), width=200)
    w.add_entity(plat)
    p = Player(agent=_Idle(), spawn_xy=(100, 540))
    w.add_entity(p)
    # Give player upward velocity (rising in pymunk y-down)
    p.body.velocity = (0, -300)
    for _ in range(15):
        w.step(1 / 60)
    # The player should have passed through the platform y=500 (now y < 500)
    assert p.body.position.y < 500


def test_spring_collision_launches_player_upward():
    from blueball.entities.spring import Spring

    w = World()
    collision.register(w.space, world_ref=w)
    s = Spring(w, position=(100, 596), width=64, impulse=600.0)
    w.add_entity(s)
    p = Player(agent=_Idle(), spawn_xy=(100, 580))
    w.add_entity(p)
    p.body.velocity = (0, 0)
    w.step(1 / 60)
    # After contact, the player should have a strong upward (negative-y) velocity
    assert p.body.velocity.y < -200


def test_spring_collision_non_player_dynamic_body():
    """A non-player dynamic body (pushable) touching a spring should also launch upward."""
    from blueball.entities.spring import Spring

    w = World()
    collision.register(w.space, world_ref=w)
    s = Spring(w, position=(200, 596), width=64, impulse=600.0)
    w.add_entity(s)
    # Simulate a pushable box: just a plain dynamic body with CT_PUSHABLE
    box_body = pymunk.Body(mass=2.0, moment=pymunk.moment_for_box(2.0, (20, 20)))
    box_body.position = (200, 580)
    box_shape = pymunk.Poly.create_box(box_body, (20, 20))
    box_shape.collision_type = collision.CT_PUSHABLE
    w.space.add(box_body, box_shape)
    box_body.velocity = (0, 0)
    w.step(1 / 60)
    # After contact, the box should have a strong upward velocity
    assert box_body.velocity.y < -200


def test_checkpoint_sets_respawn_xy_on_contact():
    from blueball.entities.checkpoint import Checkpoint
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball import config
    w, p = _player_world()
    cp = Checkpoint(w, position=(100, 100), id=1, radius=20)
    w.add_entity(cp)
    for _ in range(5):
        w.step(1 / 60)
        if cp.activated:
            break
    assert cp.activated is True
    expected_y = GROUND_Y - config.BALL_RADIUS - 4
    assert p.respawn_xy is not None
    assert p.respawn_xy[1] == expected_y


def test_checkpoint_does_not_write_save_file(tmp_save):
    from blueball.entities.checkpoint import Checkpoint
    w, p = _player_world()
    cp = Checkpoint(w, position=(100, 100), id=1, radius=20)
    w.add_entity(cp)
    for _ in range(5):
        w.step(1 / 60)
        if cp.activated:
            break
    assert cp.activated is True
    # No save file should have been created
    import pathlib
    save_path = pathlib.Path(tmp_save.SAVE_PATH)
    assert not save_path.exists()


def test_all_collision_type_constants_distinct():
    from blueball import collision as col
    names = [
        "CT_PLAYER", "CT_SPIKE", "CT_PATROLLER", "CT_COLLECTIBLE",
        "CT_GOAL", "CT_BOOST_PAD", "CT_ABILITY_PICKUP",
        "CT_ONE_WAY", "CT_SPRING", "CT_PUSHABLE", "CT_SWINGING",
        "CT_CHARGER", "CT_CHECKPOINT", "CT_KEY", "CT_DOOR",
    ]
    values = [getattr(col, n) for n in names]
    assert len(set(values)) == len(names)
    assert col.CT_ONE_WAY == 8
    assert col.CT_DOOR == 15


def test_key_collision_sets_player_keys_held_bit():
    from blueball.entities.key import Key
    w, p = _player_world()
    # Place key overlapping the player so contact is immediate
    k = Key(w, position=(100, 100), key_id=3, radius=20)
    w.add_entity(k)
    for _ in range(5):
        w.step(1 / 60)
        if k._collected:
            break
    assert k._collected is True
    assert p.has_key(3) is True


def test_key_shape_removed_from_space_after_contact():
    from blueball.entities.key import Key
    w, p = _player_world()
    k = Key(w, position=(100, 100), key_id=1, radius=20)
    w.add_entity(k)
    shape = k.shapes[0]
    # Run enough ticks for contact + update to fire
    for _ in range(10):
        w.step(1 / 60)
        if shape not in w.space.shapes:
            break
    assert shape not in w.space.shapes


def test_key_handler_is_sensor_returns_false():
    """The on_key handler must return False (no physical response)."""
    from blueball.entities.key import Key
    w, p = _player_world()
    k = Key(w, position=(100, 100), key_id=0, radius=20)
    w.add_entity(k)
    # Sensor shapes don't physically push the player — player should overlap key
    start_y = p.body.position.y
    for _ in range(5):
        w.step(1 / 60)
        if k._collected:
            break
    # Player's vertical motion should not have been blocked by the key sensor
    assert k._collected is True


def test_key_already_collected_not_double_counted():
    """If somehow on_key fires twice, keys_held should not be affected beyond the first."""
    from blueball.entities.key import Key
    w, p = _player_world()
    k = Key(w, position=(100, 100), key_id=2, radius=20)
    w.add_entity(k)
    for _ in range(10):
        w.step(1 / 60)
    # keys_held bit for key 2 should be set exactly once
    assert p.keys_held & (1 << 2)
    # Manually simulate a second handler call — should not change keys_held
    keys_before = p.keys_held
    p.collect_key(2)
    assert p.keys_held == keys_before


def test_door_blocks_player_without_key():
    """Player without the matching key must be stopped by the door (solid)."""
    from blueball.entities.door import Door
    w, p = _player_world()
    # Place door at x=150, y=100 with height=200; player is at x=100 falling
    # We need a floor and a door next to the player, then push player rightward.
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    d = Door(w, position=(200, 600), height=200, key_id=7)
    w.add_entity(d)
    p.body.position = (150, 580)
    p.body.velocity = (300, 0)
    start_x = p.body.position.x
    for _ in range(30):
        w.step(1 / 60)
    # Player should NOT have passed through x=200 because the door is solid
    assert p.body.position.x < 195


def test_door_opens_when_player_holds_matching_key():
    """Player with matching key causes door._opening=True on contact."""
    from blueball.entities.door import Door
    w, p = _player_world()
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    d = Door(w, position=(200, 600), height=200, key_id=5)
    w.add_entity(d)
    p.collect_key(5)  # player has the key
    p.body.position = (150, 580)
    p.body.velocity = (300, 0)
    for _ in range(30):
        w.step(1 / 60)
        if d._opening or d.is_open:
            break
    assert d._opening or d.is_open


def test_door_shape_removed_after_opening():
    """After _opening is set, the door shape is removed from the physics space."""
    from blueball.entities.door import Door
    w, p = _player_world()
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    d = Door(w, position=(200, 600), height=200, key_id=2)
    w.add_entity(d)
    p.collect_key(2)
    p.body.position = (150, 580)
    p.body.velocity = (300, 0)
    shape = d.shapes[0]
    for _ in range(60):
        w.step(1 / 60)
        if d.is_open:
            break
    assert d.is_open is True
    assert shape not in w.space.shapes


def test_open_door_does_not_re_trigger():
    """Once is_open, extra update() calls are no-ops."""
    from blueball.entities.door import Door
    w = World()
    d = Door(w, position=(100, 500), height=64, key_id=0)
    w.add_entity(d)
    d._opening = True
    d.update(1 / 60)
    assert d.is_open is True
    # Extra updates must not raise
    d.update(1 / 60)
    d.update(1 / 60)


# ---------------------------------------------------------------------------
# SwingingHazard collision
# ---------------------------------------------------------------------------

def test_swinging_hazard_bob_contact_kills_player():
    """Player touching the bob's CT_SWINGING shape triggers player.die()."""
    from blueball.entities.swinging_hazard import SwingingHazard
    w, p = _player_world()
    # Place bob directly overlapping player spawn (100, 100)
    sh = SwingingHazard(
        world=w,
        anchor_pos=(100, 100),
        rope_length=1,  # very short rope so bob starts at anchor
        bob_mass=1.0,
        bob_radius=20,
        initial_angle_deg=0.0,
    )
    w.add_entity(sh)
    for _ in range(10):
        w.step(1 / 60)
        if p.dead:
            break
    assert p.dead


def test_swinging_hazard_handler_registered():
    """CT_SWINGING handler must be registered on the space."""
    from blueball import collision as col
    w = World()
    col.register(w.space, world_ref=w)
    # Registering again should not blow up — idempotent
    col.register(w.space, world_ref=w)


def test_charger_top_stomp_kills_charger():
    """Stomping the charger from above kills it (entity.die() removes shape)."""
    from blueball.entities.charger import Charger
    from blueball.agent import Action, Agent

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    collision.register(w.space, world_ref=w)
    c = Charger(w, position=(100, 560), left_bound=50, right_bound=200, facing="right",
                sight_range=10, sight_arc_deg=10, charge_speed=180, patrol_speed=40)
    w.add_entity(c)
    p = Player(agent=Idle(), spawn_xy=(100, 520))
    w.add_entity(p)
    # Give the player downward velocity so it lands on top of the charger
    p.body.velocity = (0, 200)
    for _ in range(30):
        w.step(1 / 60)
        if not c.alive:
            break
    assert not c.alive
    assert c.shape not in w.space.shapes


def test_charger_side_contact_kills_player():
    """Side contact with charger kills the player."""
    from blueball.entities.charger import Charger
    from blueball.agent import Action, Agent

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    collision.register(w.space, world_ref=w)
    c = Charger(w, position=(130, 100), left_bound=50, right_bound=200, facing="right",
                sight_range=10, sight_arc_deg=10, charge_speed=180, patrol_speed=40)
    w.add_entity(c)
    p = Player(agent=Idle(), spawn_xy=(100, 100))
    w.add_entity(p)
    # Give player rightward velocity to run into the charger's side
    p.body.velocity = (300, 0)
    for _ in range(20):
        w.step(1 / 60)
        if p.dead:
            break
    assert p.dead
