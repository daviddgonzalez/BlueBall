from blueball import config
from blueball.abilities import Ability
from blueball.agent import Action
from blueball.input_feel import JumpController


def test_grounded_jump_fires_immediately():
    jc = JumpController()
    d = jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True


def test_airborne_jump_does_not_fire():
    jc = JumpController()
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_jump_buffer_fires_on_landing():
    """Press jump in air, then land within buffer window -> jump fires on landing tick."""
    jc = JumpController()
    # Press jump while airborne
    jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    # A few airborne idle ticks within the buffer window
    for _ in range(int(config.JUMP_BUFFER_TIME / config.PHYS_DT) - 1):
        d = jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
        assert d.fire is False
    # Now land - buffered jump should fire
    d = jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True


def test_jump_buffer_expires():
    """Pressing jump too long before landing should NOT trigger a buffered jump."""
    jc = JumpController()
    jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    # Run past the buffer window
    for _ in range(int(config.JUMP_BUFFER_TIME / config.PHYS_DT) + 2):
        jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    assert d.fire is False


def test_coyote_time_allows_jump_after_walkoff():
    jc = JumpController()
    # Several grounded ticks
    for _ in range(5):
        jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Walk off a ledge
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Press jump within coyote window
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_coyote_time_expires():
    jc = JumpController()
    for _ in range(5):
        jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Walk off, then wait past coyote window
    for _ in range(int(config.COYOTE_TIME / config.PHYS_DT) + 2):
        jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_jump_cut_on_release_while_rising():
    jc = JumpController()
    # Hold jump while grounded -> fires
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Keep holding for one tick
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.cut is False
    # Release
    d = jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    assert d.cut is True


def test_double_jump_available_on_first_tick_when_spawned_airborne():
    """A player spawned mid-air with DOUBLE_JUMP unlocked should still
    have their air jump on the very first tick."""
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # First tick: airborne, fresh press
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_double_jump_disabled_when_ability_missing():
    jc = JumpController()
    # Ground jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Release in air
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh airborne press — no ability, should NOT fire
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_double_jump_fires_one_extra_air_jump_when_unlocked():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Ground jump (consumes the primary)
    d = jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True
    # Release in air
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # First airborne fresh press → air jump fires
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Release
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Second airborne fresh press → no more air jumps
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False


def test_double_jump_resets_on_landing():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # First cycle: ground jump, air jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Land
    jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Second cycle: ground jump fires, air jump fires again
    d = jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    assert d.fire is True
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_double_jump_available_after_walk_off_ledge():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Several grounded ticks (no jump used)
    for _ in range(5):
        jc.tick(action=Action.IDLE, grounded=True, dt=config.PHYS_DT)
    # Walk off — grounded becomes False
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Past the coyote window
    for _ in range(int(config.COYOTE_TIME / config.PHYS_DT) + 2):
        jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh press → air jump should fire (we never used the primary)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_double_jump_air_jump_can_be_cut():
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    # Ground jump
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Hold through one airborne tick (so we don't get a 'released' immediately)
    jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    # Release
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Fresh air press → fires
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True
    # Release → cut next tick
    d = jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    assert d.cut is True
