import numpy as np

from blueball import config
from blueball.ai.ftnn import GENOME_SIZE, migrate_genome


def test_every_static_level_resolves_to_a_loadable_genome():
    for level in ["tutorial_hill", "speed_run", "maze", "lava_rising", "vertical_climb"]:
        path = config.resolve_race_ghost_genome(level)
        assert path is not None, f"no race genome mapped/present for {level}"
        assert path.exists()
        g = np.load(path)
        # Loadable = accepted by the network. Legacy (510) race-ghost assets
        # are zero-pad-migrated to the current size on load; new assets are
        # already current-size. migrate_genome raises on any other shape.
        assert migrate_genome(g).shape == (GENOME_SIZE,)


def test_unknown_level_resolves_to_none():
    assert config.resolve_race_ghost_genome("nonexistent") is None
