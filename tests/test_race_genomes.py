import numpy as np

from blueball import config
from blueball.ai.ftnn import GENOME_SIZE


def test_every_static_level_resolves_to_a_loadable_genome():
    for level in ["tutorial_hill", "speed_run", "maze", "lava_rising", "vertical_climb"]:
        path = config.resolve_race_ghost_genome(level)
        assert path is not None, f"no race genome mapped/present for {level}"
        assert path.exists()
        g = np.load(path)
        assert g.shape == (GENOME_SIZE,)


def test_unknown_level_resolves_to_none():
    assert config.resolve_race_ghost_genome("nonexistent") is None
