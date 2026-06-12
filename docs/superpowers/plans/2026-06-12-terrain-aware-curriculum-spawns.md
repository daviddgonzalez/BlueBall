# Terrain-Aware Curriculum Spawns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let levels declare explicit `curriculum_spawns` waypoints that override the start-y entity-derived stages, so the reverse spawn-curriculum works on vertical levels (lava_rising, vertical_climb) instead of spawning into a void.

**Architecture:** Hybrid. The loader parses an optional `curriculum_spawns` list into `LevelMeta`. `build_spawn_curriculum` gains one branch: if waypoints are declared, build stages from them (then append the true start); otherwise the existing entity-derived logic runs unchanged. `CurriculumStage`, `evaluate_curriculum`, `train_curriculum`, and the CLI are untouched.

**Tech Stack:** Python, pymunk physics, numpy GA, pytest.

Spec: `docs/superpowers/specs/2026-06-12-terrain-aware-curriculum-spawn-design.md`

---

### Task 1: Loader parses `curriculum_spawns` into `LevelMeta`

**Goal:** The level loader exposes an optional `curriculum_spawns` field; absent → empty tuple (every existing level unchanged).

**Files:**
- Modify: `src/blueball/levels/loader.py` (LevelMeta dataclass ~line 17-24; load_level return ~line 96-103)
- Test: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] `LevelMeta` has a `curriculum_spawns: tuple[dict, ...] = ()` field.
- [ ] A level JSON with `curriculum_spawns` round-trips into `meta.curriculum_spawns` as a tuple of dicts.
- [ ] A level JSON without the field yields `meta.curriculum_spawns == ()`.

**Verify:** `.venv/bin/python -m pytest tests/test_level_loader.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** in `tests/test_level_loader.py` (append at end):

```python
def test_curriculum_spawns_parsed(tmp_path):
    import json
    from blueball.levels.loader import load_level
    from blueball.world import World
    level = {
        "name": "T", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0],
        "curriculum_spawns": [{"x": 100, "y": 50, "keys": [0], "label": "a"}],
        "chunks": [{"type": "flat", "width_tiles": 2}],
    }
    path = tmp_path / "l.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.curriculum_spawns == ({"x": 100, "y": 50, "keys": [0], "label": "a"},)


def test_curriculum_spawns_absent_is_empty(tmp_path):
    import json
    from blueball.levels.loader import load_level
    from blueball.world import World
    level = {
        "name": "T", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "chunks": [{"type": "flat", "width_tiles": 2}],
    }
    path = tmp_path / "l.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.curriculum_spawns == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_level_loader.py::test_curriculum_spawns_parsed -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument` / `AttributeError: 'LevelMeta' object has no attribute 'curriculum_spawns'`

- [ ] **Step 3: Add the field to `LevelMeta`** (`src/blueball/levels/loader.py`):

```python
@dataclass(frozen=True)
class LevelMeta:
    name: str
    spawn: tuple[float, float]
    background: tuple[int, int, int]
    ground: tuple[int, int, int]
    total_width: float
    starting_abilities: frozenset[Ability] = frozenset()
    curriculum_spawns: tuple[dict, ...] = ()
```

- [ ] **Step 4: Parse it in `load_level`** — change the return block (~line 95-103) to:

```python
    spawn = tuple(data["spawn"])
    # Optional reverse-curriculum spawn waypoints (terrain-aware override of the
    # entity-derived stages; used by vertical levels). Absent -> empty.
    curriculum_spawns = tuple(data.get("curriculum_spawns", []))
    return LevelMeta(
        name=data["name"],
        spawn=spawn,
        background=_hex_to_rgb(data["background"]),
        ground=_hex_to_rgb(data["ground"]),
        total_width=x,
        starting_abilities=starting_abilities,
        curriculum_spawns=curriculum_spawns,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_level_loader.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/blueball/levels/loader.py tests/test_level_loader.py
git commit -m "feat(loader): parse optional curriculum_spawns into LevelMeta"
```

---

### Task 2: `build_spawn_curriculum` declarative-waypoint branch

**Goal:** When a level declares `curriculum_spawns`, build stages from them (+ appended true start); otherwise the entity-derived path runs byte-identical.

**Files:**
- Modify: `src/blueball/ai/curriculum.py` (add `_stages_from_waypoints` helper; branch at top of `build_spawn_curriculum` after `meta = load_level(...)`)
- Test: `tests/test_ai_curriculum.py`

**Acceptance Criteria:**
- [ ] A level dict with `curriculum_spawns` yields stages in declared order, each with the right `spawn_xy` and `granted_keys` bitmask (`keys:[0,1]` → `0b11`), with a final `start` stage at `meta.spawn`, `granted_keys=0`.
- [ ] The maze (no `curriculum_spawns`) still yields the 4 entity-derived stages `["near_goal","before_key1","before_key0","start"]` — default path untouched.

**Verify:** `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** in `tests/test_ai_curriculum.py` (append at end):

```python
def test_build_spawn_curriculum_uses_declared_waypoints():
    from blueball.ai.curriculum import build_spawn_curriculum
    level = {
        "name": "V", "background": "#000000", "ground": "#000000",
        "spawn": [80, 540],
        "curriculum_spawns": [
            {"x": 2740, "y": 50, "keys": [0, 1], "label": "near_goal"},
            {"x": 720, "y": 220, "keys": [], "label": "before_key0"},
        ],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    stages = build_spawn_curriculum(level)
    assert [s.label for s in stages] == ["near_goal", "before_key0", "start"]
    assert stages[0].spawn_xy == (2740.0, 50.0)
    assert stages[0].granted_keys == 0b11
    assert stages[1].spawn_xy == (720.0, 220.0)
    assert stages[1].granted_keys == 0
    assert stages[-1].label == "start"
    assert stages[-1].spawn_xy == (80.0, 540.0)
    assert stages[-1].granted_keys == 0


def test_build_spawn_curriculum_maze_unchanged_without_waypoints():
    from blueball.ai.curriculum import build_spawn_curriculum
    from blueball.ai.episodes import resolve_level_paths
    stages = build_spawn_curriculum(resolve_level_paths(["maze"])[0])
    assert [s.label for s in stages] == [
        "near_goal", "before_key1", "before_key0", "start"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_build_spawn_curriculum_uses_declared_waypoints -q`
Expected: FAIL — declared waypoints ignored (stages built from the (goal-only) entities instead), so labels/spawn_xy mismatch.

- [ ] **Step 3: Add the helper** to `src/blueball/ai/curriculum.py` (above `build_spawn_curriculum`):

```python
def _stages_from_waypoints(waypoints, start_xy) -> list[CurriculumStage]:
    """Build curriculum stages from declared waypoints (easiest -> hardest),
    then append the true start (no granted keys) as the final/hardest stage so
    the saved genome is always ultimately graded from the real spawn."""
    stages: list[CurriculumStage] = []
    for i, wp in enumerate(waypoints):
        mask = 0
        for k in wp.get("keys", []):
            mask |= (1 << int(k))
        stages.append(CurriculumStage(
            spawn_xy=(float(wp["x"]), float(wp["y"])),
            granted_keys=mask,
            label=str(wp.get("label", f"stage{i}")),
        ))
    stages.append(CurriculumStage(
        spawn_xy=(float(start_xy[0]), float(start_xy[1])),
        granted_keys=0, label="start"))
    return stages
```

- [ ] **Step 4: Add the branch** in `build_spawn_curriculum`, immediately after `meta = load_level(level, world)`:

```python
    # Terrain-aware override: a level may declare explicit spawn waypoints
    # (vertical levels, where the start-y entity-derived spawns land in a void).
    if meta.curriculum_spawns:
        return _stages_from_waypoints(meta.curriculum_spawns, meta.spawn)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (all — including the pre-existing maze tests)

- [ ] **Step 6: Commit**

```bash
git add src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(curriculum): declarative curriculum_spawns override entity-derived stages"
```

---

### Task 3: Author lava_rising + vertical_climb waypoints (+ validity/integration tests)

**Goal:** Both vertical levels declare `curriculum_spawns` that land on real platforms (not the void), and the easiest lava_rising stage no longer plunges into the lava.

**Files:**
- Modify: `src/blueball/levels/lava_rising.json` (add `curriculum_spawns`)
- Modify: `src/blueball/levels/vertical_climb.json` (add `curriculum_spawns`)
- Test: `tests/test_curriculum_waypoints.py` (new)

**Acceptance Criteria:**
- [ ] `lava_rising.json` declares 3 ascending waypoints (before_key0 → before_key1 → near_goal, easiest first) granting prior keys; each lands within 120px of static ground.
- [ ] `vertical_climb.json` declares 4 ascending waypoints (`keys:[]`); each lands within 120px of static ground.
- [ ] At the easiest lava_rising stage, a fixed genome is alive after 90 substeps and within 250px of spawn-y (supported by ground, not plunging into lava) — fails before the JSON change (start-y void spawn), passes after.

**Verify:** `.venv/bin/python -m pytest tests/test_curriculum_waypoints.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — create `tests/test_curriculum_waypoints.py`:

```python
"""Authored curriculum waypoints land on real platforms (not the void), and the
easiest lava_rising stage no longer plunges into the rising lava."""
import numpy as np
import pymunk

from blueball.ai.curriculum import build_spawn_curriculum, evaluate_curriculum
from blueball.ai.episodes import resolve_level_paths
from blueball.ai.genome import random_genome
from blueball.collision import register as register_collisions
from blueball.levels.loader import load_level
from blueball.world import World


def _ground_below(w, x, y, maxd=120):
    seg = w.space.segment_query_first((x, y), (x, y + maxd), 5, pymunk.ShapeFilter())
    return (seg.point.y - y) if seg else None


def _assert_waypoints_on_ground(level):
    path = resolve_level_paths([level])[0]
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    assert meta.curriculum_spawns, f"{level} declares no curriculum_spawns"
    for wp in meta.curriculum_spawns:
        d = _ground_below(w, float(wp["x"]), float(wp["y"]))
        assert d is not None and d >= 0, f"{level} spawn {wp} is over a void"


def test_lava_rising_waypoints_land_on_ground():
    _assert_waypoints_on_ground("lava_rising")


def test_vertical_climb_waypoints_land_on_ground():
    _assert_waypoints_on_ground("vertical_climb")


def test_lava_rising_easiest_stage_does_not_plunge():
    path = resolve_level_paths(["lava_rising"])[0]
    stages = build_spawn_curriculum(path)
    s = stages[0]  # easiest (near_goal) — was a start-y void spawn before the fix
    w = World(seed=1)
    register_collisions(w.space, world_ref=w)
    meta = load_level(path, w)
    from blueball.ai.curriculum import make_curriculum_player
    pl = make_curriculum_player(w, random_genome(np.random.default_rng(0)),
                                s.spawn_xy, s.granted_keys, meta.starting_abilities)
    for _ in range(90):
        w.substep()
        if pl.dead:
            break
    assert not pl.dead, "easiest stage died within 90 steps (plunged into lava)"
    assert abs(pl.body.position.y - s.spawn_xy[1]) < 250, "ball fell far from spawn-y"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_curriculum_waypoints.py -q`
Expected: FAIL — `lava_rising`/`vertical_climb` declare no `curriculum_spawns` yet (`assert meta.curriculum_spawns`), and the easiest stage (start-y spawn) plunges.

- [ ] **Step 3: Add waypoints to `src/blueball/levels/lava_rising.json`** — insert after the `"spawn"` line. Order is **easiest first** (near_goal), because the trainer starts at stage index 0 and recedes toward the true start. Coordinates are verified against the level's platform map (ground sits ~20-30px below each):

```json
  "spawn": [80, 540],
  "curriculum_spawns": [
    {"x": 2740, "y": 50,  "keys": [0, 1], "label": "near_goal"},
    {"x": 1744, "y": 220, "keys": [0],    "label": "before_key1"},
    {"x": 720,  "y": 220, "keys": [],     "label": "before_key0"}
  ],
```

- [ ] **Step 4: Add waypoints to `src/blueball/levels/vertical_climb.json`** — insert after the `"starting_abilities"` line (column steps at x≈1800, ascending; near_goal first):

```json
  "starting_abilities": ["double_jump"],
  "curriculum_spawns": [
    {"x": 1800, "y": -1970, "keys": [], "label": "near_goal"},
    {"x": 1800, "y": -1200, "keys": [], "label": "stage_3"},
    {"x": 1800, "y": -510,  "keys": [], "label": "stage_2"},
    {"x": 1800, "y": 180,   "keys": [], "label": "stage_1"}
  ],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_curriculum_waypoints.py -q`
Expected: PASS. If any `*_land_on_ground` assertion fails, the spawn y is over a gap at that x — adjust that waypoint's `x`/`y` onto a platform from the level's map (use the probe in Step 6) and re-run.

- [ ] **Step 6: (If adjustment needed) probe platforms** — run this to list horizontal platforms and pick on-platform coordinates:

```bash
.venv/bin/python -c "
import pymunk
from blueball.ai.episodes import resolve_level_paths
from blueball.world import World
from blueball.collision import register
from blueball.levels.loader import load_level
p=resolve_level_paths(['lava_rising'])[0]
w=World(seed=1); register(w.space,world_ref=w); load_level(p,w)
for s in sorted([s for s in w.space.shapes if isinstance(s,pymunk.Segment) and abs(s.a.y-s.b.y)<1], key=lambda s:(min(s.a.x,s.b.x),s.a.y)):
    lo,hi=sorted((s.a.x,s.b.x))
    if s.a.y<580: print(f'x[{lo:.0f},{hi:.0f}] y={s.a.y:.0f}')
"
```

- [ ] **Step 7: Run the full suite** to confirm no regressions:

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add src/blueball/levels/lava_rising.json src/blueball/levels/vertical_climb.json tests/test_curriculum_waypoints.py
git commit -m "feat(levels): terrain-aware curriculum waypoints for lava_rising + vertical_climb"
```

---

## Post-implementation (not part of this plan)

After merging, launch specialists to validate the fix produces a learning gradient (the spawns are *correct*; whether the levels *solve* is separate training iteration):

```
python main.py train maze --level lava_rising
python main.py train maze --level vertical_climb
```
