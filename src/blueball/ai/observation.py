"""Observation → FTNN input adapter.

Packs the enriched `Observation` (raycasts + hit types + nearest-entity
deltas + ability/key bitfields) into a flat float32 input vector. `INPUT_SIZE`
is the single source of truth for the layout; `ai/ftnn.py` imports it as
`FTNN_INPUTS`, so the two cannot drift.

Layout (INPUT_SIZE = 35 with N_ABILITIES=1, KEY_BITS=8):
    0 – 7    rays                  ray distances in [0, 1] (1.0 = miss)
    8 – 15   ray_semantic          per ray: +1 reward (PICKUP/GOAL),
                                    -1 danger (HAZARD/ENEMY), 0 otherwise
    16       vel_x / MAX_LINEAR_SPEED, clamped to [-1, 1]
    17       vel_y / MAX_LINEAR_SPEED, clamped to [-1, 1]
    18       ang_vel / MAX_ANGULAR_VEL, clamped to [-1, 1]
    19       grounded (0.0 / 1.0)
    20       nearest_pickup dx / NEAREST_DELTA_NORM, clamped (0 if None)
    21       nearest_pickup dy / NEAREST_DELTA_NORM, clamped (0 if None)
    22       nearest_pickup present (1.0 / 0.0)
    23       nearest_hazard dx / NEAREST_DELTA_NORM, clamped (0 if None)
    24       nearest_hazard dy / NEAREST_DELTA_NORM, clamped (0 if None)
    25       nearest_hazard present (1.0 / 0.0)
    26 ..    abilities bits — one float per Ability enum member (bit i = i-th member)
    ..       keys_held bits — KEY_BITS floats (bit i = key id i)

Encoding choices here (semantic ray channel, normalization scales, key-bit
width) are deliberately simple v1 defaults and are expected to be revisited —
see the AI brainstorm follow-ups. DOOR/BLOCK ray hits collapse to 0 in the
semantic channel; the ray distance still signals an obstacle is present.

DEPENDENCY: changing RAY_COUNT, N_ABILITIES (adding an Ability), or KEY_BITS
changes INPUT_SIZE, which changes GENOME_SIZE and invalidates saved genomes.
The runtime guard below catches a rays-shape mismatch with a clear message.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..abilities import Ability
from ..agent import HitType, Observation

RAY_COUNT = 8

# Hit-type buckets for the per-ray semantic channel.
_REWARD_HITS = frozenset({int(HitType.PICKUP), int(HitType.GOAL)})
_DANGER_HITS = frozenset({int(HitType.HAZARD), int(HitType.ENEMY)})

# Normalizer for nearest-entity world-frame deltas. Entities beyond this range
# saturate at ±1 ("far in that direction"), which is fine for control.
NEAREST_DELTA_NORM = 600.0

N_ABILITIES = len(Ability)
KEY_BITS = 8

# Section offsets into the input vector.
_RAYS_OFFSET = 0
_SEMANTIC_OFFSET = RAY_COUNT                 # 8
_VEL_OFFSET = _SEMANTIC_OFFSET + RAY_COUNT   # 16
_ANG_VEL_OFFSET = _VEL_OFFSET + 2            # 18
_GROUNDED_OFFSET = _ANG_VEL_OFFSET + 1       # 19
_PICKUP_OFFSET = _GROUNDED_OFFSET + 1        # 20  (dx, dy, present)
_HAZARD_OFFSET = _PICKUP_OFFSET + 3          # 23  (dx, dy, present)
_ABILITIES_OFFSET = _HAZARD_OFFSET + 3       # 26
_KEYS_OFFSET = _ABILITIES_OFFSET + N_ABILITIES

INPUT_SIZE = _KEYS_OFFSET + KEY_BITS


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _write_nearest(x: np.ndarray, offset: int, delta) -> None:
    """Write (dx_norm, dy_norm, present) for a nearest-entity delta at offset."""
    if delta is None:
        x[offset] = 0.0
        x[offset + 1] = 0.0
        x[offset + 2] = 0.0
    else:
        x[offset] = _clamp_unit(delta[0] / NEAREST_DELTA_NORM)
        x[offset + 1] = _clamp_unit(delta[1] / NEAREST_DELTA_NORM)
        x[offset + 2] = 1.0


def observation_to_inputs(obs: Observation) -> np.ndarray:
    # Explicit raise (not assert) so the check survives `python -O`. Stripping
    # this guard at runtime would let a wrong-shaped rays array propagate to
    # the numpy broadcast and surface as a cryptic
    # "could not broadcast input array from shape (N,) into shape (8,)"
    # buried deep in a worker traceback.
    if obs.rays.shape != (RAY_COUNT,):
        raise ValueError(
            f"observation_to_inputs expects rays of shape ({RAY_COUNT},), "
            f"got {obs.rays.shape} — update RAY_COUNT and INPUT_SIZE together."
        )

    x = np.zeros(INPUT_SIZE, dtype=np.float32)

    x[_RAYS_OFFSET:_RAYS_OFFSET + RAY_COUNT] = obs.rays

    # Per-ray semantic channel.
    for i in range(RAY_COUNT):
        ht = int(obs.ray_hit_types[i])
        if ht in _REWARD_HITS:
            x[_SEMANTIC_OFFSET + i] = 1.0
        elif ht in _DANGER_HITS:
            x[_SEMANTIC_OFFSET + i] = -1.0

    x[_VEL_OFFSET] = _clamp_unit(obs.vel[0] / config.MAX_LINEAR_SPEED)
    x[_VEL_OFFSET + 1] = _clamp_unit(obs.vel[1] / config.MAX_LINEAR_SPEED)
    x[_ANG_VEL_OFFSET] = _clamp_unit(obs.ang_vel / config.MAX_ANGULAR_VEL)
    x[_GROUNDED_OFFSET] = 1.0 if obs.grounded else 0.0

    _write_nearest(x, _PICKUP_OFFSET, obs.nearest_pickup)
    _write_nearest(x, _HAZARD_OFFSET, obs.nearest_hazard)

    for i in range(N_ABILITIES):
        if obs.abilities & (1 << i):
            x[_ABILITIES_OFFSET + i] = 1.0
    for i in range(KEY_BITS):
        if obs.keys_held & (1 << i):
            x[_KEYS_OFFSET + i] = 1.0

    return x
