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
