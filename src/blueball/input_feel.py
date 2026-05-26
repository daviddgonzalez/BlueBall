"""Input feel — jump buffering, coyote time, jump cut, optional double jump.

Pure state machine. No PyGame, no pymunk. Takes per-tick (action, grounded)
inputs and emits a JumpDecision telling the Player what to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .abilities import Ability
from .agent import Action


_JUMP_ACTIONS = {Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP}


@dataclass(frozen=True)
class JumpDecision:
    fire: bool   # apply jump impulse this tick
    cut: bool    # apply jump-cut (multiply upward velocity by JUMP_CUT_FACTOR) this tick


class JumpController:
    """Tracks jump-buffer, coyote-time, jump-cut, and air-jump state across ticks.

    `abilities` is shared by reference with the Player so unlocks land without
    a push. We hold the reference; reads happen each tick.
    """

    def __init__(self, abilities: set[Ability] | None = None) -> None:
        self.abilities: set[Ability] = abilities if abilities is not None else set()
        self._buffer_remaining = 0.0      # seconds until buffered jump expires
        self._coyote_remaining = 0.0      # seconds we still allow a jump after walking off
        self._was_grounded = False
        self._was_jump_held = False
        # Initialize with a full stock so a player spawned mid-air with
        # DOUBLE_JUMP unlocked still has their air jump on tick 1.
        self._air_jumps_remaining = self._max_air_jumps()

    def _max_air_jumps(self) -> int:
        return 1 if Ability.DOUBLE_JUMP in self.abilities else 0

    def tick(self, action: Action, grounded: bool, dt: float) -> JumpDecision:
        jump_held = action in _JUMP_ACTIONS

        # Coyote timer: starts when we lose grounding while previously grounded
        if grounded:
            self._coyote_remaining = config.COYOTE_TIME
        else:
            self._coyote_remaining = max(0.0, self._coyote_remaining - dt)

        # Air-jump counter: reset on the grounded→airborne transition. Walking
        # off a ledge restocks the air jump; landing does too (handled below
        # by the next grounded→airborne transition).
        if self._was_grounded and not grounded:
            self._air_jumps_remaining = self._max_air_jumps()

        # Jump buffer: a fresh press while airborne starts (or refreshes) the buffer
        fresh_press = jump_held and not self._was_jump_held
        if fresh_press and not grounded:
            self._buffer_remaining = config.JUMP_BUFFER_TIME
        else:
            self._buffer_remaining = max(0.0, self._buffer_remaining - dt)

        # Decide fire:
        # 1. Fresh press while grounded → fire (primary)
        # 2. Fresh press during coyote window → fire (primary)
        # 3. Landing with a live buffer → fire (primary)
        # 4. Fresh airborne press, no coyote left, air-jump available → fire (air jump)
        fire = False
        primary_consumed = False
        if fresh_press and grounded:
            fire = True
            primary_consumed = True
        elif fresh_press and self._coyote_remaining > 0.0:
            fire = True
            primary_consumed = True
        elif grounded and not self._was_grounded and self._buffer_remaining > 0.0:
            fire = True
            primary_consumed = True
        elif fresh_press and not grounded and self._air_jumps_remaining > 0:
            fire = True
            self._air_jumps_remaining -= 1

        if primary_consumed:
            # The primary jump just fired — refill the air-jump counter so the
            # player can use it during this airborne phase.
            self._air_jumps_remaining = self._max_air_jumps()

        if fire:
            self._buffer_remaining = 0.0
            self._coyote_remaining = 0.0

        # Decide cut: released jump this tick (held last tick, not held now)
        released = (not jump_held) and self._was_jump_held
        cut = released

        self._was_grounded = grounded
        self._was_jump_held = jump_held
        return JumpDecision(fire=fire, cut=cut)
