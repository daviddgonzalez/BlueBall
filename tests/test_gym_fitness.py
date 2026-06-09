from blueball import config
from blueball.ai.fitness import FitnessInputs, fitness


def _base(**over):
    kw = dict(progress_x=100.0, collectibles=0, reached_goal=False, died=False,
              steps_taken=0, keys_collected=0, level_width=0.0)
    kw.update(over)
    return FitnessInputs(**kw)


def test_segments_cleared_defaults_to_zero_backward_compatible():
    # progress only, no time/death/goal → exactly progress_x
    assert fitness(_base()) == 100.0


def test_each_cleared_segment_adds_the_bonus():
    f0 = fitness(_base())
    f3 = fitness(_base(segments_cleared=3))
    assert f3 - f0 == 3 * config.GYM_SEGMENT_BONUS
