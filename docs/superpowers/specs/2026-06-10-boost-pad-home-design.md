# Boost-pad's home in the Completion Gym + box-lava re-tune

**Date:** 2026-06-10
**Status:** design approved (sections 1–3); ready for implementation planning.

## Problem

The boost pad was folded into the box-lava segment, where probing showed it
does **not** earn its place: it doesn't make box-lava solvable, only shoves the
box ~80px further, and removing it actually *helps* a competent agent vault the
pit. Worse, box-lava as currently tuned is **not solvable** by the intended
box-step mechanic (the player overshoots the box at the speed cap) and is only
marginally vault-proof (~48px margin under a competent double-jump).

Two decisions came out of brainstorming:

1. **Give the boost a real home** — a dedicated *boost-gap* segment where the
   boost is *the* mechanic (a lava pit clearable only with a boosted jump).
2. **Re-tune box-lava** so it is genuinely solvable (box-step) *and*
   vault-proof, removing the boost from it.

Both depend on the boost pad actually working. That dependency is now satisfied
— the boost-pad bug was fixed separately (commit `720d300`):

- A boost survives **incidental** airborne moments (seam/bump hops) and is
  consumed only on the landing after a **deliberate jump**.
- A grounded boost expires after `BOOST_DURATION_S = 2.0s` if you don't jump.
- Boosts are **30% stronger** (`BOOST_STRENGTH_SCALE = 1.3`).
- Ground seams are welded (`levels/seams.py`, wired into `load_level`) so a fast
  ball doesn't catch on chunk-joint endcaps.

## Approach (selected: A)

Probe-driven fixed geometry + a shared correct-double-jump harness. A tuning
probe sweeps the freed knobs, picks the widest-margin geometry, and we hard-code
it. Each segment ships two committed invariant tests: *solvable-by-intended-
maneuver* and *not-cheesable-by-bare-double-jump*. This also fixes a
false-confidence bug: the existing vault-proof test uses an agent that only
single-jumps (it "spams RIGHT_JUMP", which `input_feel.py` fires once), so it
never tested a real vault.

---

## Section 1 — Components & shared harness

Five work items:

1. **Shared maneuver harness** (`tests/segment_maneuvers.py`) — one *correct*
   double-jump agent + per-segment *intended-solver* agents, imported by **both**
   the tuning probes and the committed tests, so "we tuned it" and "we guard it"
   run byte-identical agents.
   - `DoubleJumpVaultAgent(launch_x)` — the strongest cheese attempt: run to
     `launch_x`, then a correct max-distance double jump (press → hold to apex →
     release one tick at apex → re-press air-jump → hold to 2nd apex → drift).
     **Replaces** the broken `_DelayedJumpAgent`.
   - `BoxHopAgent(push, launch_x)` — box-lava intended solver: shove box, brake
     on the near ledge, double-jump near-ledge → box-top → far-ledge → goal.
   - `BoostLeapAgent(launch_x)` — boost-gap intended solver: roll across the pad
     staying grounded (keeps the boost), then a jump within the 2s window so the
     boost locks in and carries the player across the lava → goal.
2. New **`BoostGapSegment`** (+ a small `LavaGapChunk`).
3. **Box-lava re-tune** (pit width + depth) and removal of its boost pad.
4. Two **tuning probes** (dev tools under `probes/`, not committed guards).
5. **Committed invariant tests** + wiring/cleanup.

## Section 2 — The two segments

### 2a. `BoostGapSegment` (new, tier 2, `min_abilities={DOUBLE_JUMP}`)

Layout (left→right): `Flat(2)` → `BoostPadChunk(right)` → `Flat(short)` →
`LavaGapChunk(W)` → `Flat(landing)` → `GoalChunk(2)`.

- The `Flat(short)` between pad and pit keeps the run-up flat; with the boost
  fix, an incidental hop there no longer kills the boost, but a short flat keeps
  the jump timing clean.
- **New `LavaGapChunk`** — a boxless lava pit (near/far walls, floor, full-width
  `Lava`), fall = death. It is box-lava's pit minus the box; shared pit-building
  is extracted into a helper both chunks call. `box_lava_gap`'s public behavior
  is unchanged.

**Invariants:**
- *Solvable:* `BoostLeapAgent` reaches the goal. The player crosses the pad
  (boost granted, 30% stronger), jumps within the 2s window (boost locks in),
  and the boosted horizontal speed carries the double-jump across `W`; the boost
  clears on the far landing.
- *Anti-cheese (boost-required):* `DoubleJumpVaultAgent` with the pad swapped
  for a plain flat (no boost) **dies** — `W` exceeds the bare double-jump reach
  (~720px measured) but is within the boosted-leap reach. The probe measures the
  boosted reach (now larger, since boosts are 30% stronger) and sets `W`
  mid-corridor with margin.

### 2b. Box-lava re-tune (`BoxLavaSegment` + `KeyDoorBoxLavaSegment`)

- **Remove the boost pad** from both (revert the uncommitted `segments.py` edit;
  replace with a plain `Flat` so pit edges are unchanged).
- **Tune `pit_tiles` (narrower) + `depth` (shallower)**, box fixed at 64px,
  box-push mechanic preserved. Shallower depth raises the box-top toward ledge
  level → the short-run-up jump *off* the box can reach the far ledge; a narrower
  pit shrinks both gaps. The probe sweeps a `(pit_tiles, depth)` grid.
- **Invariants:** *Solvable:* `BoxHopAgent` reaches the goal. *Vault-proof:*
  `DoubleJumpVaultAgent` with the box removed dies.
- **Contingency:** if no `(pit_tiles, depth)` in the allowed ranges is both
  solvable and vault-proof, the probe reports the closest miss and we **stop and
  escalate** (revisit box size or the mechanic) rather than ship an unsolvable
  box-lava or change box size unilaterally.

Scope: one boost-gap segment now (no key-door-boost variant — YAGNI).

## Section 3 — Probes, tests, wiring, cleanup

**Tuning probes** (`probes/`, dev tools):
- `probes/tune_boost_gap.py` — sweep `W` (and sanity-check pad multiplier/width):
  does `BoostLeapAgent` solve? does no-boost `DoubleJumpVaultAgent` die? Print
  the safe corridor; hard-code `W` at max margin.
- `probes/tune_box_lava.py` — sweep `(pit_tiles, depth)`: `BoxHopAgent` solves?
  bare-vault dies? Print safe cells; hard-code the best (or escalate per 2b).

**Committed invariant tests** (`tests/test_segments.py`):
- Replace `_DelayedJumpAgent` with the shared `DoubleJumpVaultAgent`; repoint
  `test_boxlava_pit_requires_the_box_not_vaultable` at it.
- Add `test_boxlava_is_solvable_by_box_hop`,
  `test_boostgap_is_solvable_with_boost`,
  `test_boostgap_requires_boost_not_double_jumpable`, and a `BoostGapSegment`
  composition/requirements test.
- Update `test_all_four_tiers_registered` → five templates.

**Wiring:**
- Add `BoostGapSegment` to `SEGMENT_TEMPLATES` (tier 2,
  `min_abilities={DOUBLE_JUMP}`); `SegmentSampler` picks it up via the
  ability-filter + tier-weight. Test it appears in a double-jump pool and is
  excluded from a single-jump pool.
- **Seam welding in the gym:** `SegmentStream` materializes segments directly
  (not via `load_level`), so it does not weld. Headless training tolerates the
  cosmetic hop and the boost now survives hops, so welding is **not required**
  for gym solvability — but call `weld_ground_seams` after a segment materializes
  so on-screen gym playback (watch/visualizer) is smooth too. Low-risk, optional;
  include it.

**Cleanup:**
- Revert the uncommitted `segments.py` box-lava boost edit (superseded by 2b).
- The one-off probes were already consolidated; the two new tuning probes live
  under `probes/`.

**Out of scope / deferred:** no `KeyDoorBoostGap` variant; no GA training run
(resumes once the gym is healthy); no change to box size or the box-push
mechanic.

**Whole-effort verification:** full `pytest -q` green; the four new invariant
tests pass; both tuning probes print a non-empty safe corridor (or the box-lava
probe escalates per the 2b contingency).
