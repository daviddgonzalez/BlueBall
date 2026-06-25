from types import SimpleNamespace

from blueball import config
from blueball.ai.trainer import _episode_fitness


def _stub_player(purposeful):
    return SimpleNamespace(
        collectibles_collected=0, dead=False, keys_held=0,
        purposeful_jumps=purposeful,
    )


def test_episode_fitness_includes_purposeful_jumps():
    base = _episode_fitness(_stub_player(0), spawn_x=0.0, max_x=500.0,
                            steps=500, reached_goal=False, level_width=1000.0)
    more = _episode_fitness(_stub_player(2), spawn_x=0.0, max_x=500.0,
                            steps=500, reached_goal=False, level_width=1000.0)
    assert abs((more - base) - 2 * config.JUMP_GAP_BONUS) < 1e-6
