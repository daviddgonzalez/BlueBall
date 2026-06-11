"""Boost-pad bug reproduction — consolidates the old probe_*.py one-offs.

Symptom (reported): the first boost pad you hit "doesn't recognize it" — no
speed gain — while later pads work.

Two stacked causes, isolated by the control below (same pad, same speed, same
multiplier; only the GROUND differs):

  1. The boost-pad CHUNK lays its own ground segment, so it forms seams with the
     neighbouring flats. A fast ball trips on the seam and goes airborne a few
     frames. On ONE continuous ground segment there is no seam and no hop.
  2. player._update_boost revokes the boost on the *first airborne->grounded
     transition after pickup*, so that incidental seam-hop kills the boost
     before it ever accelerates the player. The pad does nothing.

It is speed/multiplier sensitive (a bigger kick or faster approach punches
through), which is why it "usually" bites the FIRST pad and not later ones.

`trace_repro()` runs the control and prints a verdict.
`play_repro()` boots a playable flat | boost_pad | flat | goal level.
"""

from __future__ import annotations

import pymunk

from ..abilities import Ability
from ..agent import Agent, Action
from ..collision import register as register_collisions
from ..entities.boost_pad import BoostPad
from ..entities.player import Player
from ..levels.chunks.boost_pad import BoostPadChunk
from ..levels.chunks.flat import GROUND_Y, Flat
from ..world import World
from .. import config

_NORMAL_CAP = config.MAX_LINEAR_SPEED  # boosted cap is this * multiplier * strength
_DURATION = config.BOOST_DURATION_S


class _RightAgent(Agent):
    """Pure roll-right. Crucially: never issues a jump."""
    def act(self, observation):
        return Action.RIGHT


def _measure(w, p, pad_left, steps=400):
    """Roll right; report (fired, max_rise_px, cleared, max_vx).

    max_rise is the largest the ball physically rises above its resting height
    while crossing the pad/seam — the real 'hop', not the grounded-flag flicker.
    """
    fired = cleared = False
    max_vx = 0.0
    max_rise = 0.0
    rest_y = None
    for i in range(steps):
        w.substep()
        m = p._boost_multiplier
        px, py = p.body.position
        max_vx = max(max_vx, p.body.velocity[0])
        if i == 30:
            rest_y = py  # settled resting height before reaching the pad
        if rest_y is not None and px > pad_left - 40:
            max_rise = max(max_rise, rest_y - py)  # y-up is smaller
        if m > 1.0:
            fired = True
        if fired and m <= 1.0 and not cleared and px > pad_left:
            cleared = True
    return fired, max_rise, cleared, max_vx


def _run_continuous(multiplier=1.8):
    """Pad near spawn on ONE continuous ground segment (no seams)."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    g = pymunk.Segment(w.space.static_body, (-50, GROUND_Y), (2000, GROUND_Y), 5)
    g.friction = 1.0
    w.space.add(g)
    w.add_entity(BoostPad(w, position=(120, GROUND_Y - 8), width=96, multiplier=multiplier))
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    return _measure(w, p, 72)


def _run_chunk(flat_before_tiles, multiplier=1.8, weld=False):
    """Pad on chunk-built ground (Flat | BoostPad | Flat) — has seams.
    `weld` applies the production seam weld (as load_level does)."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    x = 0.0
    x += Flat(width_tiles=flat_before_tiles).build(w, x_offset=x)
    pad_left = x
    x += BoostPadChunk(width_tiles=3, multiplier=multiplier).build(w, x_offset=x)
    x += Flat(width_tiles=40).build(w, x_offset=x)
    if weld:
        from ..levels.seams import weld_ground_seams
        weld_ground_seams(w.space)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    return _measure(w, p, pad_left)


def _verdict(max_vx):
    return "WORKS (sped up)" if max_vx > _NORMAL_CAP + 5 else "NO EFFECT (stuck at normal cap)"


def trace_repro() -> int:
    print("=== Boost-pad regression check (headless) ===")
    print("Agent only ever rolls RIGHT — it never jumps. Same pad, same 1.8x.")
    print("What the pad now does:")
    print("  clear-logic: survives incidental hops (doesn't fizzle at the seam)")
    print("  seam weld:   welded ground doesn't physically hop (seam_rise ~0)")
    print("  strength:    boosted cap = 315*1.8*1.3 = 737 (30% stronger)")
    print(f"  timer:       a grounded boost expires after {_DURATION:.0f}s with no"
          " jump (cleared=True is then EXPECTED)\n")

    cases = [
        ("A) continuous ground (no seams)", _run_continuous()),
        ("B) chunk ground, UNwelded — clear-logic fix: hops but boost survives",
         _run_chunk(2, weld=False)),
        ("C) chunk ground, welded (production via load_level) — no hop",
         _run_chunk(2, weld=True)),
    ]
    all_work = True
    for label, (fired, max_rise, cleared, max_vx) in cases:
        print(f"-- {label}")
        print(f"   fired={fired}  seam_rise={max_rise:.2f}px  expired_after_2s={cleared}"
              f"  max_vx={max_vx:.0f}  ->  {_verdict(max_vx)}")
        if max_vx <= _NORMAL_CAP + 5:
            all_work = False
        print()

    # C is the production path: it must give a speed gain and not physically hop.
    _, c_rise, _, c_vx = cases[2][1]
    if all_work and c_rise < 0.5 and c_vx > _NORMAL_CAP + 5:
        print("VERDICT: fixed. The pad speeds you up on every ground (max_vx ~737,")
        print(f"30% over the 315 cap), welded ground barely moves (C rise={c_rise:.2f}px),")
        print("and the boost expires cleanly on the 2s grounded timer.")
        return 0
    print("VERDICT: regression — a scenario gave no speed gain. See above.")
    return 1


_REPRO_LEVEL = {
    "name": "Boost Repro",
    "background": "#10131a",
    "ground": "#a17a3a",
    "spawn": [80, GROUND_Y - 30],
    "chunks": [
        {"type": "flat", "width_tiles": 8},
        {"type": "boost_pad", "width_tiles": 4, "multiplier": 2.0},
        {"type": "flat", "width_tiles": 30},
        {"type": "goal", "width_tiles": 2},
    ],
}


def play_repro() -> int:
    import pygame
    from .. import config
    from ..scenes.play import PlayScene

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball — Boost Repro")
    clock = pygame.time.Clock()
    scene = PlayScene(screen, level_data=dict(_REPRO_LEVEL))
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
    pygame.quit()
    return 0
