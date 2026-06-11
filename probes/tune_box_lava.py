"""Throwaway tuning probe: try to find a BoxHopAgent config that solves the
retuned stage-3 BoxLavaSegment (pit=24, depth=72, box=64, box on the approach
ledge — the player must PUSH it into the pit, then box-step to the goal).

The 2026-06-10 PROGRESS handoff predicts this scripted push-solve will FAIL
(stage 3 is the human-solvable-but-not-reliably-scriptable EXPERT tier). This
probe is the genuine effort behind that conclusion: it sweeps push_steps,
jump1_x, and box_run and reports the closest the agent ever gets (max player x
reached). Run under .venv/bin/python; exits 0 regardless of outcome.

    PYTHONPATH=<worktree>/src .venv/bin/python probes/tune_box_lava.py
"""
from __future__ import annotations

import os
import sys

# Make the worktree importable when run directly (editable install points at the
# main repo, so we prepend the worktree's src + root for the tests package).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_ROOT, "src"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from blueball.entities.player import Player
from blueball.levels.chunks.flat import GROUND_Y
from blueball.abilities import Ability
from blueball.levels.segments import BoxLavaSegment
from tests.segment_maneuvers import (
    BoxHopAgent, fresh_world, find_entity, run_segment,
)

# Far edge of the pit (the goal sits just past it). pit_left is calibrated to 256.
PIT_LEFT = 256
FAR_EDGE = PIT_LEFT + BoxLavaSegment().pit_tiles * 32  # x=1024 at pit=24


def try_config(push_steps: int, jump1_x: float, box_run: int):
    w = fresh_world()
    BoxLavaSegment().build(w, x_offset=0.0)
    p = Player(agent=None, spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    agent = BoxHopAgent(push_steps=push_steps, jump1_x=jump1_x, box_run=box_run)
    agent.player = p
    agent.box = find_entity(w, "PushableBox")
    p.agent = agent
    # Run and track the furthest the player ever reached (proxy for "how close").
    maxx = 0.0
    result = "TIMEOUT"
    for _ in range(2500):
        w.substep()
        maxx = max(maxx, p.body.position[0])
        if p.reached_goal:
            result = "GOAL"
            break
        if p.dead:
            result = "DEAD"
            break
    return result, maxx, agent.box.body.position


def main() -> int:
    solved = []
    best = (-1.0, None)  # (maxx, config)
    n = 0
    for push_steps in range(8, 41, 4):          # how long to drive the box right
        for jump1_x in (200, 215, 230, 245, 256):  # where to launch the first jump
            for box_run in (0, 2, 4, 6):        # steps to walk on the box top
                n += 1
                res, maxx, boxpos = try_config(push_steps, jump1_x, box_run)
                cfg = (push_steps, jump1_x, box_run)
                if res == "GOAL":
                    solved.append(cfg)
                if maxx > best[0]:
                    best = (maxx, (cfg, res, tuple(round(c) for c in boxpos)))
    print(f"BoxLavaSegment stage-3 push-solve sweep: {n} configs tried")
    print(f"  pit far edge (goal lip) at x={FAR_EDGE}")
    if solved:
        print(f"  SOLVED by {len(solved)} config(s): {solved[:10]}")
    else:
        print("  UNSOLVED — no BoxHopAgent config reached GOAL")
        maxx, detail = best
        print(f"  closest: player reached max x={maxx:.0f} "
              f"(config/result/box-pos: {detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
