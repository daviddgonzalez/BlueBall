"""Input feel — jump buffering, coyote time, jump cut.

Pure state machine. No PyGame, no pymunk. Takes per-tick (action, grounded)
inputs and emits a JumpDecision telling the Player what to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .agent import Action


_JUMP_ACTIONS = {Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP}


@dataclass(frozen=True)
class JumpDecision:
    fire: bool   # apply jump impulse this tick
    cut: bool    # apply jump-cut (multiply upward velocity by JUMP_CUT_FACTOR) this tick


class JumpController:
    """Tracks jump-buffer, coyote-time, and jump-cut state across ticks."""

    def __init__(self) -> None:
        self._buffer_remaining = 0.0      # seconds until buffered jump expires
        self._coyote_remaining = 0.0      # seconds we still allow a jump after walking off
        self._was_grounded = False
        self._was_jump_held = False

    def tick(self, action: Action, grounded: bool, dt: float) -> JumpDecision:
        jump_held = action in _JUMP_ACTIONS

        # Coyote timer: starts when we lose grounding while previously grounded
        if grounded:
            self._coyote_remaining = config.COYOTE_TIME
        else:
            self._coyote_remaining = max(0.0, self._coyote_remaining - dt)

        # Jump buffer: a fresh press while airborne starts (or refreshes) the buffer
        fresh_press = jump_held and not self._was_jump_held
        if fresh_press and not grounded:
            self._buffer_remaining = config.JUMP_BUFFER_TIME
        else:
            self._buffer_remaining = max(0.0, self._buffer_remaining - dt)

        # Decide fire:
        # 1. Fresh press while grounded -> fire
        # 2. Fresh press during coyote window -> fire
        # 3. Landing with a live buffer -> fire
        fire = False
        if fresh_press and grounded:
            fire = True
        elif fresh_press and self._coyote_remaining > 0.0:
            fire = True
        elif grounded and not self._was_grounded and self._buffer_remaining > 0.0:
            fire = True

        if fire:
            self._buffer_remaining = 0.0
            self._coyote_remaining = 0.0

        # Decide cut: released jump this tick (held last tick, not held now)
        released = (not jump_held) and self._was_jump_held
        cut = released

        self._was_grounded = grounded
        self._was_jump_held = jump_held
        return JumpDecision(fire=fire, cut=cut)
