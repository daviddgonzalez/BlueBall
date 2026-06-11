"""tune_boost_gap.py — sweep the lava-gap width for a BOOST-OR-DIE corridor.

BoostGapSegment puts a boost pad on a LONG runway right before a wide lava gap.
The boost is meant to be the ONLY way across: a boosted apex-fired double jump
clears the gap, but a bare (no-boost) double jump falls short into the lava and
dies. This probe brackets the corridor of gap widths W for which:

  - WITH the boost pad present   -> DoubleJumpVaultAgent reaches GOAL
  - WITHOUT the boost pad (stripped via remove_entity) -> the SAME agent DIES
    (or otherwise fails to reach GOAL).

The same DoubleJumpVaultAgent is the boost-gap SOLVER: when a pad sits on its
run-up, the boost comes from the WORLD, not the agent, so the boosted arc is
exactly what the segment ships with.

Layout (mirrors BoostGapSegment, LONG runway so spawn is not on the pad):
    Flat(8) | BoostPadChunk(3, mult=2.0) | LavaGapChunk(pit_tiles=W) | Goal(2)
The pit's left edge sits at (8 + 3 + approach_tiles=2) * 32 = 416px, so the
solver launches just before it at launch_x = 416 - 8 = 408.

The boost is now +30% stronger (BOOST_STRENGTH_SCALE) with a 2s lock-in timer,
so reach is LARGER than older tuning — the sweep runs wide (W ~22..44) to be
sure it brackets the boost-or-die corridor.

Running this script prints a `W | with_boost | no_boost` table and a summary of
the safe corridor (W that are GOAL-with / DEAD-without), then asserts the
corridor is non-empty and that the chosen W is in it. Pick the LARGEST W with a
tile of margin on each side.

CONTINGENCY: if no boost-or-die corridor exists across the whole sweep, this
prints a BLOCKED report with the closest misses and exits non-zero.
"""

import sys
import os

# Allow `import tests.segment_maneuvers` without installing the tests package.
# Insert repo root so `tests` is importable as a package, and src/ for blueball.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from blueball.abilities import Ability
from blueball.entities.player import Player
from blueball.levels.chunks.flat import Flat, GROUND_Y
from blueball.levels.chunks.goal import GoalChunk
from blueball.levels.chunks.boost_pad import BoostPadChunk
from blueball.levels.chunks.lava_gap import LavaGapChunk
from tests.segment_maneuvers import (
    DoubleJumpVaultAgent,
    fresh_world,
    find_entity,
    run_segment,
    remove_entity,
)

# Geometry constants — keep these in lock-step with BoostGapSegment.
_RUNWAY_TILES = 8
_PAD_TILES = 3
_PAD_MULT = 2.0
_APPROACH_TILES = LavaGapChunk().approach_tiles  # 2
# Pit left edge in px, and the launch point just before it.
_PIT_LEFT = (_RUNWAY_TILES + _PAD_TILES + _APPROACH_TILES) * 32  # 416
_LAUNCH_X = _PIT_LEFT - 8  # 408


def _build_world(pit_tiles):
    """Build Flat(8) | BoostPadChunk(3) | LavaGapChunk(W) | GoalChunk(2)."""
    w = fresh_world()
    x = 0.0
    x += Flat(width_tiles=_RUNWAY_TILES).build(w, x_offset=x)
    x += BoostPadChunk(width_tiles=_PAD_TILES, multiplier=_PAD_MULT).build(w, x_offset=x)
    x += LavaGapChunk(pit_tiles=pit_tiles).build(w, x_offset=x)
    x += GoalChunk(width_tiles=2).build(w, x_offset=x)
    return w


def _spawn_player(w, agent):
    p = Player(
        agent=agent,
        spawn_xy=(40.0, GROUND_Y - 30.0),
        abilities={Ability.DOUBLE_JUMP},
    )
    agent.player = p
    w.add_entity(p)
    return p


def with_boost_result(pit_tiles):
    """Run the boosted solver (pad present). Returns the run result string."""
    w = _build_world(pit_tiles)
    agent = DoubleJumpVaultAgent(launch_x=_LAUNCH_X)
    p = _spawn_player(w, agent)
    return run_segment(w, p, steps=2000)


def no_boost_result(pit_tiles):
    """Run the SAME solver with the boost pad STRIPPED. Returns the result."""
    w = _build_world(pit_tiles)
    pad = find_entity(w, "BoostPad")
    remove_entity(w, pad)
    agent = DoubleJumpVaultAgent(launch_x=_LAUNCH_X)
    p = _spawn_player(w, agent)
    return run_segment(w, p, steps=2000)


# Chosen width — the largest W with a tile of margin on each side of the
# boost-or-die corridor (discovered corridor is [23, 28]; W=27 has both 26 and
# 28 in-corridor, while W=28 lacks a 29). Asserted to be in the corridor below.
_CHOSEN_W = 27


def main():
    # Sweep wide: the +30%-stronger, lock-in boost reaches farther than the old
    # tuning, so start below where a bare double jump dies and run well past it.
    widths = list(range(22, 45))

    header = f"{'W':>4}  {'with_boost':>10}  {'no_boost':>9}  STATUS"
    print(header)
    print("-" * len(header))

    corridor = []   # W where boosted=GOAL AND no-boost!=GOAL (boost-or-die)
    near = []       # (W, with, without) rows that miss the corridor
    for W in widths:
        wb = with_boost_result(W)
        nb = no_boost_result(W)
        boost_or_die = (wb == "GOAL" and nb != "GOAL")
        if boost_or_die:
            status = "CORRIDOR"
            corridor.append(W)
        elif wb == "GOAL" and nb == "GOAL":
            status = "too-easy(no-boost also clears)"
            near.append((W, wb, nb))
        elif wb != "GOAL":
            status = "too-hard(boost cant clear)"
            near.append((W, wb, nb))
        else:
            status = "?"
            near.append((W, wb, nb))
        print(f"{W:>4}  {wb:>10}  {nb:>9}  {status}")

    print()
    if not corridor:
        print("=" * 64)
        print("BLOCKED: no boost-or-die corridor — no W is GOAL-with-boost AND "
              "DEAD-without-boost.")
        print("Closest rows (W | with_boost | no_boost):")
        for W, wb, nb in near:
            print(f"  W={W:>3}  with={wb:>8}  without={nb:>8}")
        print("=" * 64)
        sys.exit(1)

    lo, hi = min(corridor), max(corridor)
    print(f"SAFE CORRIDOR ({len(corridor)} widths): {corridor}  (W in [{lo}, {hi}])")
    # Pick the LARGEST W that still has a tile of margin on each side, i.e. both
    # W-1 and W+1 are also in the corridor. Fall back to the largest corridor W.
    bracketed = [W for W in corridor if (W - 1) in corridor and (W + 1) in corridor]
    pick = max(bracketed) if bracketed else max(corridor)
    print(f"Largest W with a tile of margin on each side: {pick}")
    print(f"CHOSEN _BOOST_GAP_TILES = {_CHOSEN_W}")

    assert corridor, "boost-or-die corridor is empty"
    assert _CHOSEN_W in corridor, (
        f"chosen W={_CHOSEN_W} not in corridor {corridor}"
    )
    print(f"Chosen W={_CHOSEN_W} confirmed in the boost-or-die corridor.")


if __name__ == "__main__":
    main()
