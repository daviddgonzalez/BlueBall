# Race mode (AI ghost) — design

**Date:** 2026-06-12
**Status:** Approved (design)

## Goal

Add a **Race** mode: a landing screen offers "Single Player" or "Race"; Race plays
a static level with the AI's best run rendered as a **ghost** — a blue ball,
darkened and semi-transparent — running its recorded trajectory alongside the
live human player.

Scope (locked during brainstorming): best-genome-per-level (bundled assets),
visual ghost only (no win/lose banner in v1), static levels only (Infinite Run
stays single-player).

## Architecture

The AI is deterministic (fixed-substep eval); the human plays real-time
variable-dt. Coupling them in one physics world would make the AI diverge from
its trained behaviour. So the ghost is a **pre-recorded trajectory** (the
standard time-trial pattern), recorded at race launch and replayed in real time:

1. **Record at launch** — run the level's genome through the existing
   `PlaybackSim` (mode="static", the faithful deterministic eval loop) to
   completion, capturing `(x, y, angle)` per substep. Measured cost: 30–157 ms
   once, during the menu→play transition (imperceptible).
2. **Replay in real time** — a `GhostRunner` holds the track + a time
   accumulator. Each frame it advances by `frame_dt`; the displayed pose is
   `track[floor(elapsed / PHYS_DT)]`, clamped to the last pose (the ghost
   **freezes** where its run ended — finished or died partway).
3. **Render** — a darkened, translucent ball sprite drawn at the ghost's pose.

## Components / files

### 1. `scenes/ghost.py` (new)
- `record_ghost_track(genome, level_path, *, world_seed, max_steps, abilities) -> np.ndarray`
  Builds a `PlaybackSim(mode="static", ...)`, steps to `done`, returns an
  `(N, 3) float32` array of `[x, y, angle]` per substep. Reuses PlaybackSim — no
  duplicated sim logic.
- `class GhostRunner`:
  - `__init__(self, track: np.ndarray)`
  - `update(self, frame_dt: float)` → `self._elapsed += frame_dt`
  - `pose(self) -> tuple[float, float, float]` → `track[min(floor(elapsed/PHYS_DT), N-1)]`
  - `done` property → `floor(elapsed/PHYS_DT) >= N-1` (for a future banner; unused in v1)

### 2. `render/renderer.py`
- `draw_ghost_ball(self, world_pos, deg, *, opacity=0.5, darken=0.55)` — blits the
  `ball` sprite tinted darker (RGB×`darken`) and semi-transparent
  (`alpha=int(255*opacity)`) at `world_pos` through the camera, mirroring
  `draw_ball`'s transform. The tinted surface is cached (built once) so it costs
  no more than a normal blit per frame.

### 3. `scenes/mode_select.py` (new)
- `ModeSelectScene` — landing screen, two entries: "Single Player", "Race".
  Enter → `MenuScene(screen, mode=<single|race>)`. Esc → quit (returns `None`).
  Same key handling / draw style as the current menu.

### 4. `scenes/menu.py`
- `MenuScene.__init__(self, screen, mode="single")`.
  - `single`: entries unchanged (5 static levels + Infinite Run); selecting →
    `PlayScene(..., mode="single")` (same level launch as today). Esc now returns
    to the landing screen rather than quitting (see below) — the one behavioural
    change, since a landing screen now sits above the level select.
  - `race`: entries = the 5 static levels only (Infinite Run hidden — it has no
    goal/finish). Selecting a level → resolve its ghost genome, record the track,
    and launch `PlayScene(..., ghost=GhostRunner(track), mode="race")`. If no
    genome is bundled for that level (or it fails to load) → launch
    `PlayScene(..., mode="race")` with **no ghost** (graceful single-player
    fallback).
  - Esc → `ModeSelectScene(screen)` (back), in both modes.

### 5. `scenes/play.py`
- `PlayScene.__init__` gains `ghost: GhostRunner | None = None` and `mode: str = "single"`.
- `update`: if `ghost is not None`, call `ghost.update(frame_dt)` (advances in real
  time alongside the human's `world.step`).
- `draw`: if `ghost is not None`, `self.renderer.draw_ghost_ball(pos, deg)` after the
  world entities, before particles/present.
- Esc / level-complete navigation returns `MenuScene(screen, mode=self._mode)` so
  race stays in race mode.
- When `ghost is None`, the update/draw paths are unchanged from today.
- **Fairness:** in race mode the human is granted the same abilities the ghost
  used — `extra_abilities={Ability.DOUBLE_JUMP}` (all bundled race genomes are
  double-jump-capable) — so both sides play with the same moveset.

### 6. `config.py`
- `RACE_GHOST_ABILITIES = ("double_jump",)` — abilities granted to both the ghost
  recording and the human in race mode.
- `RACE_GHOST_GENOMES: dict[str, str]` — maps each static level name to a bundled
  genome filename under the race-ghost asset dir. A resolver returns the absolute
  path, or `None` if the level is unmapped or the file is absent.

### 7. `assets/race_ghosts/` (new, committed)
- One `<level>.npy` genome per static level — the best available run:
  generalist for `tutorial_hill` / `speed_run`; the strongest specialist (or
  generalist) for `maze` / `lava_rising` / `vertical_climb`. These are committed
  game assets (unlike the gitignored training `genomes/`). Chosen at implementation
  time from the current best runs.

### 8. `cli.py`
- `cmd_play` opens `ModeSelectScene(screen)` instead of `MenuScene(screen)`.

## Data flow

```
ModeSelectScene --mode--> MenuScene(mode)
  race + level select:
    resolve genome (config.RACE_GHOST_GENOMES) -> load .npy
    record_ghost_track(genome, level_path, abilities=RACE_GHOST_ABILITIES) -> (N,3)
    GhostRunner(track)
  -> PlayScene(level_path, ghost=runner, mode="race")
       update(): world.step(frame_dt)  +  ghost.update(frame_dt)
       draw():   entities ... + draw_ghost_ball(ghost.pose())
```

## Testing (TDD, all headless)

- `GhostRunner`: `update`/`pose` index math — real-time pacing (`elapsed/PHYS_DT`),
  clamp/freeze at the final pose past track end.
- `record_ghost_track`: returns a non-empty `(N,3)` array; the poses match a direct
  `PlaybackSim` run of the same genome/level (faithfulness).
- `draw_ghost_ball`: headless render smoke (dummy SDL surface) — blits without
  error; produces a surface distinct from the opaque ball.
- `ModeSelectScene`: Enter on "Race" → `MenuScene` with `mode=="race"`; "Single
  Player" → `mode=="single"`; Esc → `None`.
- `MenuScene(mode="race")`: entries exclude Infinite Run; `mode="single"` includes it.
- `MenuScene` race fallback: a level with no bundled genome yields a `PlayScene`
  with `ghost is None` (no crash).
- `PlayScene(ghost=...)`: constructs, updates, draws over a few frames without
  error; `ghost=None` leaves the single-player update/draw path unchanged.
- `config.RACE_GHOST_GENOMES`: every mapped asset file exists and loads to a
  `GENOME_SIZE`-length array.

## Out of scope (v1)

- Win/lose banner, race timer, leaderboard.
- Racing Infinite Run (distance race).
- Multiple ghosts / ghost selection UI.
- Pre-baked tracks (record-at-launch chosen; the hitch is 30–157 ms).
```
