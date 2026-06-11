# Boost-pad's Home — PROGRESS & HANDOFF (paused 2026-06-10)

Resume of the subagent-driven execution of `2026-06-10-boost-pad-home.md`. The plan
was **reshaped mid-execution** (see "Big change" below). This doc is the source of
truth for resuming; `*.tasks.json` mirrors the status.

## Big change: box-lava → a 3-stage BOX CURRICULUM

Task 2 ("re-tune one box-lava so it's box-solvable AND double-jump-proof") hit its
contingency: **no pit-width/depth is both** — a bare double jump's vault reach
(~700px ≈ 22 tiles) **exceeds** the box-hop's max solvable width (~20 tiles), so the
box is never simultaneously necessary and sufficient. David playtested (`play-gym`,
see below) and confirmed pit-22/depth-72 is vault-proof + human-solvable, but the
push-the-box-then-box-step maneuver is too precise to script reliably or train.

**Decision (David):** replace the single re-tune with a difficulty curriculum that
teaches the box mechanic in stages:
- **Stage 1 — single jump onto a (pre-placed) box.** Easiest.
- **Stage 2 — double jump onto/off a BIGGER (pre-placed) box.** Medium.
- **Stage 3 — the current level: push the box yourself, then box-step.** Expert.

Why pre-placing the box (stages 1–2) works: put the box exactly where a **natural**
jump lands, so no frame-perfect timing is needed → scriptable AND trainable. The
shove (stage 3) is the hard part and stays the expert tier.

## Controller-validated geometry (all confirmed solvable + vault-proof)
- **Stage 1 (BoxStepSegment):** `Flat(2)|Flat(3)|BoxLavaGap(pit=24, depth=72, box=64, box_frac=0.5)|Goal(2)`; pit_left=256, box centered ~640. `SingleStepAgent(launch_x≈232, on_box_run=3)` solves; box-removed `DoubleJumpVaultAgent` can't vault 768px. **Robust** (SAFE set: pit 24–28 × frac 0.45–0.6).
- **Stage 2 (BoxLeapSegment):** `Flat(2)|Flat(3)|BoxLavaGap(pit=38, depth=96, box=96, box_frac=0.52)|Goal(2)`. `DoubleStepAgent` solves; vault-proof. **MARGINAL** — in 2a only solved at `launch_x=230, frac=0.52`. **Task 2c's probe must find a ROBUST stage-2 cell** (try depth/frac/box variations); if it stays knife-edge, reconsider stage-2 geometry.
- **Stage 3 (BoxLavaSegment):** pit=22, depth=72, box=64, **pushed** from the approach ledge. Vault-proof + human-solvable; scripted solver UNSOLVED. Task 2d: remove the boost pad (replace with `Flat(3)` spacer to keep pit_left=256 — `BoxHopAgent` is calibrated to 256), and **if no reliable scripted solve, flag stage 3 expert** (commit only vault-proof + composition invariants, no solvable-by-agent test). Do NOT change box size/mechanic unilaterally.

## Key facts / gotchas
- Interpreter: `.venv/bin/python` (`python` not on PATH).
- `play-gym` CLI (committed `34be626`): `.venv/bin/python main.py play-gym box-lava [--pit N --depth D]` / `play-gym boost-gap [--gap N]`. Lets David hand-test segments. Boost-gap has a long run-up (David: "don't spawn on the pad") — carry the same generous runway into the real `BoostGapSegment` (Task 3).
- `pit_left=256` calibration: box-lava segments MUST keep the pit edge at x=256 (Flat(2)+Flat(3)+approach_tiles=3) or the scripted agents (calibrated to 256) break.
- Pre-placed box: `BoxLavaGap(..., box_frac=F)` places the box on the floor at fraction F across the pit; `box_frac=None` (default) = on the approach ledge (stage 3 push). Assumes `depth >= box_size`.
- Solver agents (in `tests/segment_maneuvers.py`): `SingleStepAgent`, `DoubleStepAgent`, plus `DoubleJumpVaultAgent` (the strongest-cheese guard, apex-fired max double jump) and `BoxHopAgent` (stage-3 push solver, currently can't solve a wide pushed pit).
- Known committed test failure: `tests/test_segments.py::test_boxlava_random_varies_pit_width` — fails because the WIP fixed pit to 24. **Task 2d deletes/replaces it.** Until 2d, expect "1 failed" in full runs; that one is EXPECTED.

## Status (commits on branch feature/completion-gym)
- ✅ Task 0 — maneuver harness (`45bc32a`) — reviewed
- ✅ Task 1 — LavaGapChunk + build_lava_pit (`8081d2e`) — reviewed
- ➕ play-gym dev tool (`34be626`)
- ✅ Task 2a — pre-placed-box option + SingleStep/DoubleStep agents (`23b036d`) — reviewed
- 🟡 Task 2b — BoxStepSegment / stage 1 (`1bbfd01`) — **implemented+committed; spec+quality review PENDING**
- ⬜ Task 2c — BoxLeapSegment / stage 2 (probe must find a robust cell)
- ⬜ Task 2d — retune BoxLavaSegment / stage 3 (drop boost pad; flag expert if unscriptable)
- ⬜ Task 3 — BoostGapSegment (boost-or-die; use a long runway per David)
- ⬜ Task 4 — wire all new segments into SEGMENT_TEMPLATES + sampler; fix the false-confidence vault test; update tier-count test
- ⬜ Task 5 — weld seams on gym segment materialize (cosmetic)

## Next action on resume
1. Spec + quality review Task 2b (`git diff 34be626 1bbfd01`), then mark it done.
2. Build Task 2c (stage 2) — **probe for a robust stage-2 cell first**.
3. Then 2d, 3, 4, 5. Task 4 wires everything; update the tier-count test to the final template count (will be >5 now: Goal, KeyDoorGoal, BoxStep, BoxLeap, BoxLava, BoostGap, KeyDoorBoxLava).
