"""Fixed-Topology Neural Network (FTNN) used by AI agents.

A two-layer fully-connected network: 14 inputs → 12 tanh hidden → 6 outputs.
The 6 outputs correspond one-to-one with the `Action` enum values; the
trainer's `FTNNAgent.act()` picks `argmax` to choose an action.

The whole network is parameterized by a single flat float32 genome array.
Layout:
    [W1 (14*12=168) | b1 (12) | W2 (12*6=72) | b2 (6)]
    GENOME_SIZE = 168 + 12 + 72 + 6 = 258
"""

from __future__ import annotations

import numpy as np

FTNN_INPUTS = 14
FTNN_HIDDEN = 12
FTNN_OUTPUTS = 6   # one per Action

_W1_SIZE = FTNN_INPUTS * FTNN_HIDDEN
_B1_SIZE = FTNN_HIDDEN
_W2_SIZE = FTNN_HIDDEN * FTNN_OUTPUTS
_B2_SIZE = FTNN_OUTPUTS

GENOME_SIZE = _W1_SIZE + _B1_SIZE + _W2_SIZE + _B2_SIZE


class FTNN:
    """A 14 → 12 tanh → 6 fully-connected network. Pure numpy."""

    def __init__(self, genome: np.ndarray) -> None:
        if genome.shape != (GENOME_SIZE,):
            raise ValueError(
                f"FTNN requires a genome of shape ({GENOME_SIZE},), got {genome.shape}"
            )
        if genome.dtype != np.float32:
            genome = genome.astype(np.float32)
        else:
            genome = genome.copy()

        i = 0
        self._W1 = genome[i:i + _W1_SIZE].reshape(FTNN_INPUTS, FTNN_HIDDEN)
        i += _W1_SIZE
        self._b1 = genome[i:i + _B1_SIZE]
        i += _B1_SIZE
        self._W2 = genome[i:i + _W2_SIZE].reshape(FTNN_HIDDEN, FTNN_OUTPUTS)
        i += _W2_SIZE
        self._b2 = genome[i:i + _B2_SIZE]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one observation through the network. Returns shape (FTNN_OUTPUTS,)."""
        h = np.tanh(x @ self._W1 + self._b1)
        return (h @ self._W2 + self._b2).astype(np.float32, copy=False)
