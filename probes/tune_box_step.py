"""tune_box_step.py — sweep pit_tiles × box_frac for BoxStepSegment geometry.

For each (pit_tiles, box_frac) pair, prints:
  - SOLVABLE if at least one SingleStepAgent combo reaches GOAL
  - VAULT_PROOF if no DoubleJumpVaultAgent (box removed) reaches GOAL
  - SAFE if both conditions hold

Running this script prints the full sweep and a SAFE set summary at the end.
The SAFE set must be non-empty and must include pit_tiles=24.
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
    SingleStepAgent,
    fresh_world,
    find_entity,
    run_segment,
)


def _build_world(pit_tiles, depth, box_size, box_frac):
    """Build Flat(2) | Flat(3) | BoxLavaGap(..., box_frac=...) | GoalChunk(2)."""
    w = fresh_world()
    x = 0.0
    x += Flat(width_tiles=2).build(w, x_offset=x)
    x += Flat(width_tiles=3).build(w, x_offset=x)
    x += BoxLavaGap(
        pit_tiles=pit_tiles, depth=depth, box_size=box_size, box_frac=box_frac
    ).build(w, x_offset=x)
    x += GoalChunk(width_tiles=2).build(w, x_offset=x)
    return w


def _remove_entity(world, entity):
    if entity in world.entities:
        world.entities.remove(entity)
    for shp in list(getattr(entity, "shapes", [])):
        if shp in world.space.shapes:
            world.space.remove(shp)
        world._shape_to_entity.pop(shp, None)
    for bod in list(getattr(entity, "bodies", [])):
        if bod in world.space.bodies:
            world.space.remove(bod)


def check_solvable(pit_tiles, depth, box_size, box_frac):
    """Return True if at least one SingleStepAgent combo reaches GOAL."""
    sweep = [
        (lx, obr)
        for lx in (232, 238, 245, 252)
        for obr in (3, 6, 10)
    ]
    for launch_x, on_box_run in sweep:
        w = _build_world(pit_tiles, depth, box_size, box_frac)
        p = Player(
            agent=None,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        w.add_entity(p)
        agent = SingleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
        agent.player = p
        agent.box = find_entity(w, "PushableBox")
        p.agent = agent
        result = run_segment(w, p)
        if result == "GOAL":
            return True
    return False


def check_vault_proof(pit_tiles, depth, box_size):
    """Return True if NO DoubleJumpVaultAgent (box removed) reaches GOAL."""
    for lx in (220, 240, 248, 254, 260):
        w = _build_world(pit_tiles, depth, box_size, box_frac=0.5)
        box = find_entity(w, "PushableBox")
        _remove_entity(w, box)
        agent = DoubleJumpVaultAgent(launch_x=lx)
        p = Player(
            agent=agent,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        agent.player = p
        w.add_entity(p)
        result = run_segment(w, p, steps=1500)
        if result == "GOAL":
            return False  # not vault-proof
    return True


def main():
    pit_tiles_range = [20, 22, 24, 26, 28]
    box_frac_range = [0.40, 0.45, 0.50, 0.55, 0.60]
    depth = 72
    box_size = 64

    safe_set = []

    print(f"{'pit_tiles':>10}  {'box_frac':>8}  {'SOLVABLE':>8}  {'VAULT_PROOF':>11}  {'STATUS':>6}")
    print("-" * 54)

    for pit_tiles in pit_tiles_range:
        vp = check_vault_proof(pit_tiles, depth, box_size)
        for box_frac in box_frac_range:
            sol = check_solvable(pit_tiles, depth, box_size, box_frac)
            status = "SAFE" if sol and vp else ("solvable" if sol else ("vault_proof" if vp else "neither"))
            print(f"{pit_tiles:>10}  {box_frac:>8.2f}  {str(sol):>8}  {str(vp):>11}  {status:>6}")
            if sol and vp:
                safe_set.append((pit_tiles, box_frac))

    print()
    print(f"SAFE set ({len(safe_set)} combos): {safe_set}")
    assert safe_set, "SAFE set is empty — no geometry is both solvable and vault-proof!"
    assert any(pt == 24 for pt, _ in safe_set), (
        f"pit_tiles=24 not in SAFE set: {safe_set}"
    )
    print("pit_tiles=24 confirmed in SAFE set.")


if __name__ == "__main__":
    main()
