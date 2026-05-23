from blueball import config
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
