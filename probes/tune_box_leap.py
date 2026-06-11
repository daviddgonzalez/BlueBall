"""tune_box_leap.py — find a ROBUST BoxLeapSegment (curriculum stage 2) cell.

Stage 2 teaches the "box as stepping stone" via a DOUBLE jump: run up, fire a
max apex-fired double jump ONTO a bigger pre-placed box, then a second double
jump OFF it to the goal. The discriminator from stage 1 is that a SINGLE jump
must NOT reach the box — stage 2 needs the double jump's horizontal reach.

For each geometry cell (pit_tiles, depth, box_size, box_frac) this sweeps the
solver's (launch_x, on_box_run) and reports:
  - SOLVABLE-ROBUSTLY : >= _ROBUST_MIN distinct DoubleStepAgent combos reach GOAL
                        (a single solving combo is a knife-edge, NOT robust).
  - VAULT_PROOF       : NO box-removed DoubleJumpVaultAgent (swept launch_x) wins.
  - NOT_SINGLE_JUMPABLE: NO SingleStepAgent combo reaches GOAL (so a single jump
                        cannot get onto/across the box — this is what makes it
                        stage 2 and not stage 1).
  - SAFE = robustly-solvable AND vault_proof AND not_single_jumpable.

Running this script prints the full sweep and a SAFE-set summary, then asserts a
non-empty SAFE set exists and that the chosen cell is in it. Pick the SAFE cell
with the widest solving margin.

CONTINGENCY: if no cell is robustly-solvable AND vault-proof AND
not-single-jumpable after a thorough sweep, this prints a BLOCKED report with the
closest misses and exits non-zero — a geometry rethink is the controller's call.
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
from blueball.levels.chunks.box_lava_gap import BoxLavaGap
from tests.segment_maneuvers import (
    DoubleJumpVaultAgent,
    DoubleStepAgent,
    SingleStepAgent,
    fresh_world,
    find_entity,
    run_segment,
    remove_entity,
)

# A cell needs at least this many distinct solving (launch_x, on_box_run) combos
# to count as ROBUST — one solving combo is a knife-edge and is rejected.
_ROBUST_MIN = 3


def _build_world(pit_tiles, depth, box_size, box_frac):
    """Build Flat(2) | Flat(3) | BoxLavaGap(..., box_frac=...) | GoalChunk(2).
    Flat(2)+Flat(3)+approach_tiles(3) = 8 tiles = 256px keeps pit_left at the
    x=256 calibration the scripted agents are tuned to."""
    w = fresh_world()
    x = 0.0
    x += Flat(width_tiles=2).build(w, x_offset=x)
    x += Flat(width_tiles=3).build(w, x_offset=x)
    x += BoxLavaGap(
        pit_tiles=pit_tiles, depth=depth, box_size=box_size, box_frac=box_frac
    ).build(w, x_offset=x)
    x += GoalChunk(width_tiles=2).build(w, x_offset=x)
    return w


def _spawn_player(w, agent=None):
    p = Player(
        agent=agent,
        spawn_xy=(40.0, GROUND_Y - 30.0),
        abilities={Ability.DOUBLE_JUMP},
    )
    w.add_entity(p)
    return p


# launch_x values bracket the x=256 pit_left edge: a double jump fired from
# ~226..256 (the near ledge) carries the apex-fired arc onto the mid-pit box.
_SOLVE_LAUNCH_X = (226, 232, 238, 244, 250, 256)
_SOLVE_ON_BOX_RUN = (4, 6, 8, 10)


def solving_combos(pit_tiles, depth, box_size, box_frac):
    """Return the list of (launch_x, on_box_run) DoubleStepAgent combos that
    reach GOAL on this cell."""
    solved = []
    for launch_x in _SOLVE_LAUNCH_X:
        for on_box_run in _SOLVE_ON_BOX_RUN:
            w = _build_world(pit_tiles, depth, box_size, box_frac)
            p = _spawn_player(w)
            agent = DoubleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
            agent.player = p
            agent.box = find_entity(w, "PushableBox")
            p.agent = agent
            if run_segment(w, p) == "GOAL":
                solved.append((launch_x, on_box_run))
    return solved


def vault_proof(pit_tiles, depth, box_size, box_frac):
    """True if NO box-removed DoubleJumpVaultAgent (swept launch_x) reaches GOAL."""
    for lx in (220, 232, 240, 248, 254, 260):
        w = _build_world(pit_tiles, depth, box_size, box_frac)
        remove_entity(w, find_entity(w, "PushableBox"))
        agent = DoubleJumpVaultAgent(launch_x=lx)
        p = _spawn_player(w, agent=agent)
        agent.player = p
        if run_segment(w, p, steps=1500) == "GOAL":
            return False
    return True


def not_single_jumpable(pit_tiles, depth, box_size, box_frac):
    """True if NO SingleStepAgent combo reaches GOAL — a single jump can't get
    onto/across the box. This is the stage-2 discriminator."""
    for launch_x in _SOLVE_LAUNCH_X:
        for on_box_run in _SOLVE_ON_BOX_RUN:
            w = _build_world(pit_tiles, depth, box_size, box_frac)
            p = _spawn_player(w)
            agent = SingleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
            agent.player = p
            agent.box = find_entity(w, "PushableBox")
            p.agent = agent
            if run_segment(w, p) == "GOAL":
                return False
    return True


def main():
    # Sweep around the task's starting points. depth >= box_size always (so the
    # box-top never pokes above the ledge). The grid is focused around the
    # known-good region to keep the run inside the time budget while still
    # exploring pit width, box size, depth, and frac for the widest margin.
    cells = []
    for box_size in (96, 104):
        for depth in (96, 112):
            if depth < box_size:
                continue
            for pit_tiles in (34, 36, 38, 40, 42, 44):
                for box_frac in (0.45, 0.50, 0.52, 0.55, 0.60):
                    cells.append((pit_tiles, depth, box_size, box_frac))

    header = (f"{'pit':>4} {'depth':>5} {'box':>4} {'frac':>5}  "
              f"{'#solve':>6} {'vault_proof':>11} {'not_single':>10}  STATUS")
    print(header)
    print("-" * len(header))

    safe_set = []          # (cell, n_solve)
    near_misses = []       # cells that solve robustly but fail one guard
    for cell in cells:
        pit_tiles, depth, box_size, box_frac = cell
        combos = solving_combos(*cell)
        n = len(combos)
        robust = n >= _ROBUST_MIN
        # Only run the (expensive) guards when the cell is at least robust-solvable.
        vp = nsj = None
        if robust:
            vp = vault_proof(*cell)
            nsj = not_single_jumpable(*cell)
        safe = bool(robust and vp and nsj)
        if safe:
            status = "SAFE"
            safe_set.append((cell, n))
        elif robust:
            status = "near(" + ("vp " if vp else "!vp ") + \
                     ("nsj" if nsj else "!nsj") + ")"
            near_misses.append((cell, n, vp, nsj))
        else:
            status = "thin" if n else "unsolved"
        print(f"{pit_tiles:>4} {depth:>5} {box_size:>4} {box_frac:>5.2f}  "
              f"{n:>6} {str(vp):>11} {str(nsj):>10}  {status}")

    print()
    if not safe_set:
        print("=" * 60)
        print("BLOCKED: no SAFE (robust-solvable AND vault-proof AND "
              "not-single-jumpable) cell found.")
        print("Closest misses (robust-solvable but failing a guard):")
        for cell, n, vp, nsj in sorted(near_misses, key=lambda r: -r[1])[:10]:
            print(f"  {cell}  #solve={n}  vault_proof={vp}  not_single={nsj}")
        print("=" * 60)
        sys.exit(1)

    # Choose the SAFE cell with the widest solving margin (then lowest pit for a
    # gentler stage, then mid box_frac for centering).
    safe_set.sort(key=lambda r: (-r[1], r[0][0], abs(r[0][3] - 0.52)))
    chosen, chosen_n = safe_set[0]
    print(f"SAFE set ({len(safe_set)} cells): "
          f"{[(c, n) for c, n in safe_set]}")
    print(f"CHOSEN (widest margin): {chosen}  with {chosen_n} solving combos")

    assert safe_set, "SAFE set is empty — no cell is robust+vault-proof+not-single"
    assert (chosen, chosen_n) in safe_set, "chosen cell not in SAFE set"
    print("Chosen cell confirmed in SAFE set.")


if __name__ == "__main__":
    main()
