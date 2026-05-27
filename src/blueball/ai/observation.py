"""Observation → FTNN input adapter.

Packs the existing v1 `Observation` into the 14-float input vector the
FTNN expects.

Layout (indices):
    0–7:  obs.rays                              (8 floats)
    8–9:  obs.vel[0], obs.vel[1]                (2 floats)
     10:  obs.ang_vel                           (1 float)
     11:  1.0 if obs.grounded else 0.0          (1 float)
    12–13: nearest_collectible offset (0, 0 when None)

DEPENDENCY: if the level-design branch's Observation enrichment ever changes
rays.shape from (8,) to a different size, update RAY_COUNT here AND FTNN_INPUTS
in ai/ftnn.py in lockstep. The assertion below catches the mismatch with a
clear message.
"""

from __future__ import annotations

import numpy as np

from ..agent import Observation

RAY_COUNT = 8


def observation_to_inputs(obs: Observation) -> np.ndarray:
    assert obs.rays.shape == (RAY_COUNT,), (
        f"observation_to_inputs expects rays of shape ({RAY_COUNT},), "
        f"got {obs.rays.shape} — update RAY_COUNT and FTNN_INPUTS together."
    )
    x = np.empty(14, dtype=np.float32)
    x[0:8] = obs.rays
    x[8] = obs.vel[0]
    x[9] = obs.vel[1]
    x[10] = obs.ang_vel
    x[11] = 1.0 if obs.grounded else 0.0
    if obs.nearest_collectible is None:
        x[12] = 0.0
        x[13] = 0.0
    else:
        x[12] = obs.nearest_collectible[0]
        x[13] = obs.nearest_collectible[1]
    return x
