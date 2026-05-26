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


def test_double_jump_unlocked_mid_air_grants_immediate_extra_jump():
    """Unlocking DOUBLE_JUMP while airborne should immediately make the air
    jump available, not wait until the next ground→air cycle.

    Reproduces the bug where collecting the pickup mid-jump appeared to do
    nothing until after the player landed once.
    """
    jc = JumpController(abilities=set())
    # Ground jump (primary fires, refills air-jump counter to 0 since no ability)
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    # Airborne — second press would not fire because counter is 0
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is False
    # Release
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    # Pick up the DOUBLE_JUMP ability mid-air (set mutated by reference + notify)
    jc.abilities.add(Ability.DOUBLE_JUMP)
    jc.on_ability_added(Ability.DOUBLE_JUMP)
    # Next fresh airborne press should now fire the air jump
    d = jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)
    assert d.fire is True


def test_on_ability_added_relies_on_unlock_idempotency():
    """on_ability_added itself tops up the counter unconditionally when the
    multiplier room is there. The safety net against double-refill is the
    Player.unlock idempotency guard, which ensures on_ability_added fires at
    most once per ability per Player lifetime. This test pins that contract:
    if you call on_ability_added twice in the same air phase after the air
    jump has been used, the second call WILL refill — so don't call it twice."""
    jc = JumpController(abilities={Ability.DOUBLE_JUMP})
    jc.tick(action=Action.JUMP, grounded=True, dt=config.PHYS_DT)
    jc.tick(action=Action.IDLE, grounded=False, dt=config.PHYS_DT)
    jc.tick(action=Action.JUMP, grounded=False, dt=config.PHYS_DT)  # air jump used
    assert jc._air_jumps_remaining == 0
    # Documenting current behavior: a second on_ability_added call WOULD refill.
    # Player.unlock is responsible for not double-calling.
    jc.on_ability_added(Ability.DOUBLE_JUMP)
    assert jc._air_jumps_remaining == 1
