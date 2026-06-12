# Terrain-aware curriculum spawns (declarative waypoints)

**Date:** 2026-06-12
**Status:** Approved (design)

## Problem

`build_spawn_curriculum` (ai/curriculum.py) places every reverse-curriculum
stage's spawn at the level's **start-y** (`meta.spawn[1]`). This works for
horizontal levels (maze: ~constant ground height) but is broken for vertical
levels.

Evidence (lava_rising, start-y = 540): every intermediate stage spawns into a
~560px void above the rising lava —

```
near_goal    spawn=(2368,540)  ground_below=560px
before_key1  spawn=(1728,540)  ground_below=560px
before_key0  spawn=(704,540)   ground_below=560px
start        spawn=(80,540)    ground_below=44px   (only the true start lands)
```

So a `train maze --level lava_rising` specialist would drop the agent into the
lava every stage. This is *why* lava_rising has no working specialist, and the
generalist fails it for a related reason (it skips both keys on a high climb
line → the two-key `goal_vault` is unopenable → it falls into the lava).

Automatic terrain probing was investigated and rejected: keys/goal float
~400–800px above their platforms (collected mid-jump), and ray-casting gives
inconsistent results on irregular vertical geometry (sometimes the right ledge,
sometimes the deep base ground 800px down). There is no robust automatic rule
for "the platform this stage should start on." Probing also cannot help
`vertical_climb`, which has no keys to anchor on.

## Approach: hybrid (declarative override, entity-derived default)

Keep today's entity-derived stages as the default so **maze and all horizontal
levels stay byte-identical**. Let a level *optionally* declare explicit spawn
waypoints that override the derived stages — used by vertical levels only.

This also fixes `vertical_climb`: with no keys it currently yields a single
`start` stage (no scaffolding), which is why its specialist plateaus at ~50% of
the climb. Declarative waypoints give it real intermediate stages.

## Design

### 1. New optional level field: `curriculum_spawns`

Listed easiest → hardest (near-goal first):

```json
"curriculum_spawns": [
  {"x": 2368, "y": 120, "keys": [0, 1], "label": "near_goal"},
  {"x": 1728, "y": 300, "keys": [0],    "label": "before_key1"},
  {"x": 704,  "y": 480, "keys": [],     "label": "before_key0"}
]
```

- `x`, `y` — spawn position, author-placed on a real platform.
- `keys` — key_ids pre-granted at this spawn (OR'd into a bitmask), so gates /
  the goal_vault behind it are openable. Default `[]`.
- `label` — for history/logging. Default `stage<N>`.

### 2. Loader (levels/loader.py)

Parse `curriculum_spawns` into a new `LevelMeta.curriculum_spawns` field — a
tuple of plain dicts `{"x", "y", "keys", "label"}` (picklable, simple),
defaulting to empty via `data.get("curriculum_spawns", [])`. Absent field →
empty tuple → every existing level unchanged.

### 3. `build_spawn_curriculum` (ai/curriculum.py)

One branch added at the top:

- If `meta.curriculum_spawns` is non-empty → build `CurriculumStage`s directly
  from the waypoints (`keys:[0,1]` → `granted_keys = (1<<0)|(1<<1)`), then
  **append the true `start` stage** (`meta.spawn`, `granted_keys=0`) as the
  final/hardest stage — exactly as the derived path does today, guaranteeing the
  saved genome is always ultimately graded from the real spawn.
- Else → today's entity-derived logic, byte-identical.

`CurriculumStage`, `evaluate_curriculum`, `train_curriculum`, and the CLI are
untouched. The CLI verdict already uses `stages[-1]` (the appended true start),
so it needs no change.

### 4. Author waypoints

- `lava_rising` — ascending spawns before each key, granting prior keys.
- `vertical_climb` — spawns up the column, `keys:[]`.

Positions are derived during implementation by probing each level's platforms
(authoring-time only, not runtime).

## Testing (TDD)

- **Loader:** `curriculum_spawns` parsed into meta; absent → empty.
- **Generator:** waypoint level → stages match waypoints + appended start;
  **maze (no field) → byte-identical** to current stages (locks the default
  path).
- **Anti-regression on the original bug:** every declared lava_rising /
  vertical_climb spawn has static ground within ~N px below it (ray-cast
  assertion) — no void/lava spawns.
- **Integration:** `evaluate_curriculum` at the easiest lava_rising waypoint
  does not instantly die from falling into the lava.

## Out of scope

- Automatic geometry probing.
- Tuning waypoint positions until a level is actually *solved* — that's training
  iteration, separate from the spawn mechanism this spec delivers.
```
