import pytest

from blueball import config
from blueball.ai.fitness import FitnessInputs, fitness


def _inputs(**over):
    base = dict(
        progress_x=1000.0, collectibles=0, reached_goal=False, died=False,
        steps_taken=1000, keys_collected=0, level_width=0.0,
    )
    base.update(over)
    return FitnessInputs(**base)


def test_gap_bonus_applied_all_modes():
    no_gap = fitness(_inputs(level_width=0.0, purposeful_jumps=0))
    gap = fitness(_inputs(level_width=0.0, purposeful_jumps=3))
    assert gap - no_gap == config.JUMP_GAP_BONUS * 3


def test_efficiency_off_when_goalless():
    # The only step-dependence at level_width=0 is the pre-existing -0.01*steps
    # term; NO efficiency term should be added.
    a = fitness(_inputs(level_width=0.0, steps_taken=500))
    b = fitness(_inputs(level_width=0.0, steps_taken=1500))
    assert (a - b) == pytest.approx(-0.01 * (500 - 1500))


def test_efficiency_on_with_level_width():
    fast = fitness(_inputs(level_width=2000.0, steps_taken=500))
    slow = fitness(_inputs(level_width=2000.0, steps_taken=2000))
    assert fast > slow
    # The entire fast-vs-slow gap is efficiency + the pre-existing step cost.
    expected = (config.SPEED_W * 1000.0 / 500 - 0.01 * 500) - (config.SPEED_W * 1000.0 / 2000 - 0.01 * 2000)
    assert (fast - slow) == pytest.approx(expected)
