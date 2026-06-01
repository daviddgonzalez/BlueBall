# Box-Puzzle Chunks — Design (2026-06-01)

## Overview

Three new pushable-box puzzle chunks for Blue Ball Phase 3. All three reuse
existing physics — **no new collision handlers, no changes to `src/blueball/ai/`
or `agent.py`'s `Action` enum.**

1. **`box_lava_gap`** (horizontal) — push a box into a long, shallow lava pit so
   it becomes a mid-pit stepping stone to cross.
2. **`box_spring_trampoline`** (vertical) — push a box onto a spring so it bounces;
   bounce off the rising box to reach a high exit.
3. **`box_spring_relay`** (vertical) — relay a box across two springs to reach an
   exit too high for a single spring.

## Physics the design leans on (verified in the current code)

- **Lava** (`CT_LAVA`) is a *sensor*; `on_lava` kills only the player. A
  `PushableBox` is unaffected and falls straight through lava to rest on solid
  geometry beneath it.
- **Spring** (`CT_SPRING`) — `on_spring` sets any dynamic body's
  `vy = min(vy, -impulse)` on each *begin* contact, so a box that lands back on a
  spring is re-launched at a consistent height → perpetual, even bounce.
- Static, localized lava is achieved by spawning the existing `Lava` entity with
  `rise_speed = 0` (velocity 0 → it does not rise), sized to the pit.

## Chunk 1: `box_lava_gap` (horizontal traversal)

```
   approach ledge                          exit ledge
 ────────────────┐                       ┌────────────────
   [box]→        │ ▓▓▓▓▓ lava (long) ▓▓▓▓ │
                 │                       │
                 └───────────────────────┘  solid pit floor (shallow)
```

**Geometry** (`base_y` = ground line; pymunk y-down):
- Approach ledge: static segment, `approach_tiles` wide at `base_y`.
- Pit: `pit_tiles` wide. Solid floor segment at `base_y + depth`; two vertical
  wall segments (near & far) from `base_y` down to `base_y + depth`.
- Lava: `Lava(rise_speed=0)`, surface a few px below `base_y`, width = pit span,
  height = `depth`. Lethal to the player only.
- Box: a `PushableBox` on the approach ledge near the pit edge.
- Exit ledge: static segment, `exit_tiles` wide at `base_y`.

**Solve:** the pit is **too long to clear in one jump**. Shove the box off the
near edge → it drops to the pit floor and (low-friction floor + the push's
momentum) settles near the **middle** → cross in two jumps: approach ledge →
box top → exit ledge.

**Tuning (validate via `/run` playtest):** `pit_tiles` long enough to defeat a
single jump; `depth` shallow enough that the box-top is a reachable step; pit
floor friction low so a firm push carries the box toward center.

**Sampler:** `sampler_include = True`, `difficulty = 3`. `random_params` picks
`pit_tiles` (≈4–6), `depth`, and box size. Must stream/cull correctly in
Infinite Run — **implementation must confirm the `Lava` entity culls with the
chunk** (it is normally loader-spawned, not chunk-spawned).

**Placement:** also inserted into `maze.json`.

**Risk (recorded, non-blocking):** in the *endless* Infinite Run a non-solving
agent can idle safely on the approach ledge (no forced death) → a possible
no-progress stall for a simple GA. Revisit with an idle/no-progress timeout or a
push-puzzle curriculum when GA training begins. Tracked alongside the deferred
key/door curriculum work.

## Chunk 2: `box_spring_trampoline` (vertical)

```
                    ┌─── high exit ledge
       you ↗        │
      [B] ↕  (bouncing)
     ╲╱╲╱ spring
 ───────────────── ground
```

**Geometry:** ground segment at `base_y` with a `Spring` on top; a `PushableBox`
on the ground near the spring; a high exit ledge at `exit_height` above `base_y`,
`exit_dx` to the side — unreachable by spring-alone or a single jump.

**Solve:** push the box onto the spring → perpetual bounce; jump onto the rising
box near its apex → carried upward + jump-off → reach the exit ledge.

**Feasibility:** depends on box-carry + jump stacking giving enough height. Tune
spring `impulse`, box `mass`/`size`, and `exit_height` by playtest. If it is not
reliably solvable, lower the exit or raise the impulse rather than ship a
coin-flip.

**Sampler:** `sampler_include = False`, `difficulty = 4`.
**Placement:** `vertical_climb.json`.

## Chunk 3: `box_spring_relay` (vertical)

```
                        ┌── high exit
              [B]↑↑      │
        ╲╱  →   ╲╱       spring 2 (on a raised platform)
      spring 1
 ───────────────────── ground
```

**Geometry:** Spring 1 on the ground at `base_y`; a raised platform at height
`h2` carrying Spring 2, offset `relay_dx` horizontally; **guide walls** to
constrain the box's arc between the springs; an exit ledge above Spring 2.

**Solve:** push the box onto Spring 1 *with horizontal momentum* → it arcs up and
over onto Spring 2 → re-launched from the higher platform → reaches the exit; the
player follows up the relay.

**Feasibility:** the most fragile of the three — the horizontal arc depends on the
box's `vx` at launch. Guide walls + tuning. **If it proves unsolvable-by-design
after tuning, flag it and simplify (e.g. fall back to a trampoline variant)
rather than ship something janky.**

**Sampler:** `sampler_include = False`, `difficulty = 5`.
**Placement:** `vertical_climb.json`.

## Code touched

- **No new entities.** Reuse `Lava` (with `rise_speed=0`), `Spring`,
  `PushableBox`. Confirm a chunk may `world.add_entity(Lava(...))` directly and
  that it culls under streaming.
- **New chunk files:** `src/blueball/levels/chunks/box_lava_gap.py`,
  `box_spring_trampoline.py`, `box_spring_relay.py`.
- **Register** all three in `src/blueball/levels/chunks/__init__.py`.
- **Edit levels:** `maze.json` (insert `box_lava_gap`), `vertical_climb.json`
  (insert both spring puzzles).

## Testing

**Structural unit tests** (`tests/test_chunks.py`), per chunk:
- `box_lava_gap`: builds approach + pit-floor + two walls + exit segments, one
  `Lava` entity (sensor, `CT_LAVA`), one `PushableBox`; returns correct total
  width; `sampler_include` True, `difficulty` set, `random_params` constructible.
- `box_spring_trampoline`: ground + `Spring` + `PushableBox` + exit-ledge segment
  present; `sampler_include` False.
- `box_spring_relay`: two `Spring`s + raised platform + guide walls + box + exit
  present; `sampler_include` False.

**Physics smoke tests:**
- `box_lava_gap`: step the world; the box comes to rest on the pit floor
  (`y ≈ floor − size/2`) and is *not* destroyed by lava; a player overlapping the
  lava dies.
- spring chunks: after the box contacts a spring its `vy ≤ −impulse`, and it
  bounces repeatedly across successive steps.

**Streaming:** extend the existing Infinite Run streaming smoke test to confirm
`box_lava_gap` appears in the stream and its `Lava` + box entities spawn and cull
without leaking entity count.

**Full human-solve** of chunks #2 and #3 is validated by **playtest (`/run`)**,
not asserted in unit tests (timing/arc-dependent).

## Out of scope / deferred

- No new collision handlers; no floating-box buoyancy.
- No GA curriculum for these chunks yet (deferred with key/door curriculum).
- Sampler inclusion of the two spring puzzles (vertical + multi-step; deferred).

## Hard rules respected

- Do not touch `src/blueball/ai/` or `agent.py`'s `Action` enum.
- Hand-authored level edits stay blocky / un-curved.
- pymunk-ce: `arbiter.process_collision = False`, `apply_impulse_at_world_point`.

---

## Aside: rebalance existing levels for element coverage (follow-up scope)

Separate from the new chunks above, the hand-authored levels under-use the
element vocabulary — a few elements carry every level while most appear once or
twice. This is a **follow-up rebalancing pass**, not part of the box-chunk
implementation plan; captured here so it isn't lost.

**Current usage audit** (chunk `"type"` counts across all 5 hand-authored JSON
levels: `tutorial_hill`, `speed_run`, `maze`, `vertical_climb`, `lava_rising`):

| Heavily used | Count | Rarely used | Count |
|---|---|---|---|
| `flat` | 27 | `swinging_hazard` (pendulum) | 2 |
| `platform` | 15 | `patrol_platform` (enemy) | 2 |
| `vertical_column` | 7 | `moving_platform` | 2 |
| `boost_pad` | 5 | `stairs_up` / `stairs_down` | 1 each |
| `gap` | 4 | `ice_floor` | 1 |
| `spring` | 3 | `pushable_box` | 1 *(addressed by this spec)* |
| | | `charger_platform` (enemy) | 1 |
| | | `cannon_lane` (chunk) | 0 in hand levels |

**Observations:**
- **Enemies are scarce** — only 3 enemy instances total (`charger_platform` ×1 +
  `patrol_platform` ×2) across five levels.
- **Pendulum** (`swinging_hazard`) appears twice; **box** is rare (this spec adds
  more); **ice, stairs, moving platforms** are nearly unused.

**Proposed rebalancing principles (for the follow-up pass):**
- Every level should draw from a wider slice of the vocabulary; aim for each
  non-structural element to appear in **at least two levels**, and no single
  hazard to be a one-off.
- Increase enemy density (charger + patroller) so combat/avoidance is a
  consistent thread, not a rarity — without overloading any single level.
- Use the new box chunks (this spec) plus more `pushable_box`, `ice_floor`,
  `stairs`, `moving_platform`, and `swinging_hazard` placements to even out
  coverage.
- Keep hand-authored levels blocky/un-curved and difficulty-curved within each
  level.

**Method:** the audit table above becomes a coverage checklist; the pass edits
the existing level JSONs only (no new chunk code needed beyond this spec) and is
verified by re-running the frequency audit + the existing level-load/streaming
tests. **This pass gets its own spec + plan** — it is intentionally out of scope
for the box-chunk implementation.
