"""AI ghost for Race mode: record a genome's deterministic run as a pose track,
then replay it in real time alongside the live human player."""
from __future__ import annotations

import numpy as np

from .. import config


def record_ghost_track(genome, level_path, *, world_seed=config.DEFAULT_SEED,
                       max_steps=config.MAX_STEPS, abilities=()):
    """Run `genome` on a static level through PlaybackSim (the faithful
    deterministic eval loop) to completion. Returns an (N, 3) float32 array of
    [x, y, angle] sampled once per substep."""
    from .playback import PlaybackSim

    sim = PlaybackSim(genome, mode="static", level_path=str(level_path),
                      world_seed=world_seed, max_steps=max_steps, abilities=abilities)
    poses = []
    while not sim.done:
        sim.step_once()
        b = sim.player.body
        poses.append((b.position.x, b.position.y, b.angle))
    return np.asarray(poses, dtype=np.float32).reshape(-1, 3)


class GhostRunner:
    """Replays a recorded pose track in real time. Advances by wall-clock
    frame_dt; the track holds one pose per PHYS_DT, so the ghost moves at the
    AI's true pace. Freezes on the final pose once the run ends."""

    def __init__(self, track: np.ndarray) -> None:
        self._track = np.asarray(track, dtype=np.float32).reshape(-1, 3)
        if len(self._track) == 0:
            raise ValueError("GhostRunner requires a non-empty track")
        self._elapsed = 0.0

    def update(self, frame_dt: float) -> None:
        self._elapsed += max(0.0, frame_dt)

    @property
    def _index(self) -> int:
        return min(int(self._elapsed / config.PHYS_DT), len(self._track) - 1)

    def pose(self) -> tuple[float, float, float]:
        x, y, a = self._track[self._index]
        return float(x), float(y), float(a)

    @property
    def done(self) -> bool:
        return self._index >= len(self._track) - 1
