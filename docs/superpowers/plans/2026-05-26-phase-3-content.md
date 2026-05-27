# Phase 3 — Level Design & Content Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Phase 3 content slice — 14 new chunk types, 10 new entity types, 3 hand-built levels, a level-select menu, the ChunkSampler with Infinite Run, and the enriched agent Observation the AI session is waiting on.

**Architecture:** Three layers added in sequence. **Foundations** (chunk difficulty attrs, Player API additions, World→entity wiring, Spike orientation, new collision-type constants) ship first because every chunk depends on them. **Entities and their chunks** (one task per entity+chunk+handler triple) ship next, smallest-first to build confidence. **Plumbing** (Observation rewrite, loader dict-mode, ChunkSampler, MenuScene, PlayScene level_data path, renderer) glues it together. Three hand-built level JSONs + determinism guards finish the slice.

**Tech Stack:** Python 3.11+, PyGame-ce, Pymunk, pytest. No new third-party dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-26-phase-3-content-design.md`.

---

## File structure

Final layout after this plan lands. New files marked `+`, modified files marked `~`.

```
src/blueball/
├── config.py                              (~ Phase 3 tunables + Observation constants)
├── world.py                               (~ add_entity sets entity._world = self)
├── agent.py                               (~ HitType enum, enriched Observation)
├── collision.py                           (~ 8 new CT_* + handlers for new entities)
├── abilities.py                           (unchanged)
├── save.py                                (unchanged)
├── camera.py                              (unchanged)
├── input_feel.py                          (unchanged)
├── entities/
│   ├── player.py                          (~ keys_held, respawn_xy, collect_key, has_key, receive_spring, ShapeFilter group, _observe rewrite, _world consumer)
│   ├── spike.py                           (~ orientation parameter)
│   ├── moving_platform.py                 (+ kinematic body oscillator)
│   ├── spring.py                          (+ sensor strip)
│   ├── checkpoint.py                      (+ sensor flag)
│   ├── crumbling_platform.py              (+ static seg + delayed self-removal)
│   ├── key.py                             (+ sensor pickup, key_id bitfield)
│   ├── door.py                            (+ solid segment that opens on key)
│   ├── pushable_box.py                    (+ dynamic body)
│   ├── swinging_hazard.py                 (+ static anchor + PinJoint + dynamic bob)
│   ├── one_way_platform.py                (+ static seg + pre_solve filter)
│   └── charger.py                         (+ kinematic enemy with FOV + LOS + charge state)
├── levels/
│   ├── loader.py                          (~ dict-mode source)
│   ├── sampler.py                         (+ ChunkSampler)
│   ├── tutorial_hill.json                 (unchanged)
│   ├── vertical_climb.json                (+ new level)
│   ├── speed_run.json                     (+ new level)
│   ├── maze.json                          (+ new level)
│   └── chunks/
│       ├── __init__.py                    (~ imports for 14 new chunks)
│       ├── base.py                        (~ Chunk.difficulty + sampler_include + random_params hook)
│       ├── flat.py                        (~ difficulty=0, random_params)
│       ├── gap.py                         (~ difficulty=1, random_params)
│       ├── spike_pit.py                   (~ difficulty=2, random_params)
│       ├── patrol_platform.py             (~ difficulty=2, random_params)
│       ├── stairs.py                      (~ difficulty=0 both)
│       ├── bump.py                        (~ difficulty=0, random_params)
│       ├── falling_hazard.py              (~ difficulty=3, random_params)
│       ├── goal.py                        (~ sampler_include=False)
│       ├── ability_pickup.py              (~ sampler_include=False)
│       ├── boost_pad.py                   (~ difficulty=1, random_params)
│       ├── platform.py                    (+ floating segment)
│       ├── vertical_column.py             (+ stacked platforms macro)
│       ├── moving_platform.py             (+ kinematic chunk)
│       ├── spring.py                      (+ vertical bouncer)
│       ├── checkpoint.py                  (+ respawn marker)
│       ├── one_way_platform.py            (+ jump-through)
│       ├── crumbling_platform.py          (+ timed self-removal)
│       ├── key.py                         (+ key pickup)
│       ├── door.py                        (+ locked barrier)
│       ├── pushable_box.py                (+ dynamic crate)
│       ├── spike_wall.py                  (+ oriented spike chunk)
│       ├── swinging_hazard.py             (+ pendulum)
│       ├── ice_floor.py                   (+ low-friction floor)
│       └── charger_platform.py            (+ Charger spawner)
├── render/
│   └── renderer.py                        (~ draw methods + colors for all new entities)
└── scenes/
    ├── menu.py                            (+ level-select scene with 5 entries)
    └── play.py                            (~ Esc to menu, _exit_to_menu, level_data path, checkpoint respawn)

main.py                                    (~ start in MenuScene instead of PlayScene)

tests/
├── test_player.py                         (~ keys, respawn_xy, _observe rewrite tests)
├── test_entities.py                       (~ tests for each new entity)
├── test_chunks.py                         (~ registry includes all 24, per-chunk smoke)
├── test_collision.py                      (~ handlers for each new CT)
├── test_level_loader.py                   (~ smoke-load new levels + dict mode)
├── test_play_scene.py                     (~ checkpoint, esc-to-menu, level-complete-to-menu)
├── test_menu_scene.py                     (+ cursor + Enter + Esc + Infinite Run)
├── test_sampler.py                        (+ determinism, ramp, includes/excludes, goal terminator)
└── test_world_determinism.py              (~ speed_run + sampler-built level cases)
```

---

## Task 1: Chunk base — difficulty, sampler_include, random_params hook

**Goal:** Add `difficulty: int = 0`, `sampler_include: bool = True`, and `random_params(cls, rng) -> dict` (default returns `{}`) to the `Chunk` base class. The sampler later reads these without inspecting chunk internals.

**Files:**
- Modify: `src/blueball/levels/chunks/base.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `Chunk` exposes class attributes `difficulty: int = 0` and `sampler_include: bool = True` with the documented defaults.
- [ ] `Chunk.random_params(rng)` is a classmethod returning `{}` by default.
- [ ] Every existing concrete chunk subclass inherits these attributes without subclass changes.
- [ ] One test asserts the default values on `Flat` (which never overrides them in this task).

**Verify:** `pytest -q tests/test_chunks.py -v` → all pass (existing + 1 new).

**Steps:**

- [ ] **Step 1: Write failing test**

Append to `tests/test_chunks.py`:

```python
def test_chunk_base_defaults_difficulty_and_sampler_include():
    assert Flat.difficulty == 0
    assert Flat.sampler_include is True
    # random_params is a classmethod returning {}
    import random as _rng
    assert Flat.random_params(_rng.Random(0)) == {}
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest -q tests/test_chunks.py::test_chunk_base_defaults_difficulty_and_sampler_include -v`
Expected: FAIL — `AttributeError: type object 'Flat' has no attribute 'difficulty'`.

- [ ] **Step 3: Add the attributes and method to `Chunk`**

Replace the `Chunk` class body in `src/blueball/levels/chunks/base.py`:

```python
class Chunk(abc.ABC):
    # Sampler integration. Default values keep existing chunks sampler-eligible
    # at trivial difficulty until concrete subclasses override.
    difficulty: int = 0
    sampler_include: bool = True

    @classmethod
    def random_params(cls, rng) -> dict:
        """Return a kwargs dict the sampler should pass to __init__.
        Default: use the chunk's __init__ defaults."""
        return {}

    @abc.abstractmethod
    def build(self, world, x_offset: float) -> float:
        """Materialize the chunk's bodies and entities into `world`, anchored at
        `x_offset` (left edge of the chunk). Returns the chunk's width in world
        units so the level loader can place the next chunk.
        """
```

- [ ] **Step 4: Run test, confirm pass**

Run: `pytest -q tests/test_chunks.py -v`
Expected: PASS for the new test; no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/chunks/base.py tests/test_chunks.py
git commit -m "feat: Chunk gains difficulty + sampler_include + random_params hook"
```

---

## Task 2: Assign difficulty values to existing chunks

**Goal:** Tag every existing chunk type with its `difficulty` and `sampler_include` per the spec tables. No behavior change; only class-attribute assignments and `random_params` overrides where the sampler needs randomization.

**Files:**
- Modify: `src/blueball/levels/chunks/flat.py`
- Modify: `src/blueball/levels/chunks/gap.py`
- Modify: `src/blueball/levels/chunks/spike_pit.py`
- Modify: `src/blueball/levels/chunks/patrol_platform.py`
- Modify: `src/blueball/levels/chunks/stairs.py`
- Modify: `src/blueball/levels/chunks/bump.py`
- Modify: `src/blueball/levels/chunks/falling_hazard.py`
- Modify: `src/blueball/levels/chunks/goal.py`
- Modify: `src/blueball/levels/chunks/ability_pickup.py`
- Modify: `src/blueball/levels/chunks/boost_pad.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] Difficulty and `sampler_include` match the spec table for every existing chunk.
- [ ] `random_params` is overridden on chunks the sampler must randomize: `flat`, `gap`, `spike_pit`, `bump`, `boost_pad`, `falling_hazard`, `patrol_platform`, `stairs_up`, `stairs_down`. (Other chunks keep the base default.)
- [ ] Tests assert the difficulty value for one representative of each tier (0/1/2/3) and assert `sampler_include` for `goal` and `ability_pickup` is False.

**Verify:** `pytest -q tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunks.py`:

```python
def test_existing_chunks_difficulty_assigned():
    assert Flat.difficulty == 0
    assert Gap.difficulty == 1
    assert SpikePit.difficulty == 2
    from blueball.levels.chunks.falling_hazard import FallingHazardChunk
    assert FallingHazardChunk.difficulty == 3


def test_goal_and_ability_pickup_excluded_from_sampler():
    from blueball.levels.chunks.ability_pickup import AbilityPickupChunk
    assert GoalChunk.sampler_include is False
    assert AbilityPickupChunk.sampler_include is False


def test_flat_random_params_returns_width_in_range():
    import random as _rng
    params = Flat.random_params(_rng.Random(0))
    assert 2 <= params["width_tiles"] <= 5
```

- [ ] **Step 2: Run, confirm failures**

Run: `pytest -q tests/test_chunks.py -v`
Expected: 3 failures.

- [ ] **Step 3: Assign attributes (modifications to each chunk file)**

In `src/blueball/levels/chunks/flat.py`, inside `class Flat`, after `def __init__`, add:

```python
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}
```

In `src/blueball/levels/chunks/gap.py`:

```python
    difficulty: int = 1

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}
```

In `src/blueball/levels/chunks/spike_pit.py`:

```python
    difficulty: int = 2

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 4), "spikes": rng.randint(2, 4)}
```

In `src/blueball/levels/chunks/patrol_platform.py`:

```python
    difficulty: int = 2

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"length_tiles": rng.randint(4, 8), "patroller_speed": rng.choice([40.0, 60.0, 80.0])}
```

In `src/blueball/levels/chunks/stairs.py`, on both `StairsUp` and `StairsDown`:

```python
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"steps": rng.randint(2, 4), "step_height": rng.choice([24, 32, 40])}
```

In `src/blueball/levels/chunks/bump.py`:

```python
    difficulty: int = 0

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"height": rng.randint(24, 48), "width_tiles": rng.randint(2, 3)}
```

In `src/blueball/levels/chunks/falling_hazard.py`:

```python
    difficulty: int = 3

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "hazard_height": rng.randint(160, 240)}
```

In `src/blueball/levels/chunks/goal.py`, inside `class GoalChunk`, add:

```python
    sampler_include: bool = False
```

In `src/blueball/levels/chunks/ability_pickup.py`, inside `class AbilityPickupChunk`, add:

```python
    sampler_include: bool = False
```

In `src/blueball/levels/chunks/boost_pad.py`:

```python
    difficulty: int = 1

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 6), "multiplier": round(rng.uniform(1.5, 2.2), 2)}
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_chunks.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/chunks/ tests/test_chunks.py
git commit -m "feat: tag existing chunks with difficulty + sampler params"
```

---

## Task 3: Player API additions — keys_held, respawn_xy, collect_key, has_key, receive_spring

**Goal:** Extend `Player` with the bitfield, respawn point, key methods, and spring-impulse method. No motion/physics changes; purely additive new fields and methods.

**Files:**
- Modify: `src/blueball/entities/player.py`
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] `Player` initializes `keys_held: int = 0` and `respawn_xy: tuple[float, float] | None = None`.
- [ ] `collect_key(key_id)` sets the matching bit; idempotent (calling twice with same id leaves the bit set).
- [ ] `has_key(key_id)` returns `True` iff the bit is set.
- [ ] `receive_spring(impulse)` applies `(0, -impulse * self.body.mass)` at the body's local center.

**Verify:** `pytest -q tests/test_player.py -v`

**Steps:**

- [ ] **Step 1: Write failing tests**

Append to `tests/test_player.py`:

```python
def test_player_starts_with_no_keys_and_no_respawn():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    assert p.keys_held == 0
    assert p.respawn_xy is None
    assert p.has_key(0) is False
    assert p.has_key(5) is False


def test_player_collect_key_sets_bit_and_is_idempotent():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.collect_key(3)
    assert p.has_key(3) is True
    assert p.keys_held == (1 << 3)
    p.collect_key(3)  # idempotent
    assert p.keys_held == (1 << 3)
    p.collect_key(0)
    assert p.has_key(0) is True
    assert p.keys_held == (1 << 3) | (1 << 0)


def test_player_receive_spring_applies_upward_impulse():
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    p.body.velocity = (0, 0)
    p.receive_spring(impulse=400.0)
    # pymunk y-down: upward velocity is negative
    # Player mass is 1.0; impulse = 400 * 1.0 = 400 => delta-v = -400 y
    assert p.body.velocity.y == -400.0
```

- [ ] **Step 2: Run, confirm failures**

Run: `pytest -q tests/test_player.py -v`
Expected: 3 AttributeError failures.

- [ ] **Step 3: Add the new state and methods to `Player`**

In `src/blueball/entities/player.py`, inside `Player.__init__`, after the existing fields and before the closing of `__init__`, add:

```python
        self.keys_held: int = 0
        self.respawn_xy: tuple[float, float] | None = None
```

After the existing `unlock` method, add:

```python
    def collect_key(self, key_id: int) -> None:
        """Set the bit for `key_id` in keys_held. Idempotent."""
        self.keys_held |= (1 << key_id)

    def has_key(self, key_id: int) -> bool:
        return bool(self.keys_held & (1 << key_id))

    def receive_spring(self, impulse: float) -> None:
        """Vertical upward impulse, mass-scaled so the resulting delta-v
        is the same regardless of body mass. Pymunk y-down → up is -y."""
        self.body.apply_impulse_at_local_point(
            (0, -impulse * self.body.mass), (0, 0)
        )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/test_player.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: Player gains keys_held, respawn_xy, key + spring methods"
```

---

## Task 4: World.add_entity sets entity._world

**Goal:** When `World.add_entity` is called, it sets `entity._world = self` so entities that need to query the space (Player raycasts, Charger LOS) can do so without a manual wire-up.

**Files:**
- Modify: `src/blueball/world.py`
- Modify: `tests/test_player.py` or new tests file

**Acceptance Criteria:**
- [ ] After `world.add_entity(p)`, `p._world is world`.
- [ ] Existing entities (Patroller, Goal, etc.) are unaffected — the attribute just exists on them, unused.

**Verify:** `pytest -q tests/test_player.py -v` and `pytest -q tests/ -v` (no regressions).

**Steps:**

- [ ] **Step 1: Write failing test**

Append to `tests/test_player.py`:

```python
def test_world_add_entity_wires_world_reference():
    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    assert p._world is w
```

- [ ] **Step 2: Run, confirm failure**

Expected: AttributeError on `p._world`.

- [ ] **Step 3: Modify `World.add_entity`**

In `src/blueball/world.py`, change `add_entity`:

```python
    def add_entity(self, entity) -> None:
        """Register an entity with the world. Adds the entity's bodies and shapes
        to the pymunk space and tracks the entity for per-tick updates. Also
        wires `entity._world = self` so entities that need to query the space
        (raycasts, LOS) can do so without a manual hookup.
        """
        entity._world = self
        for body in getattr(entity, "bodies", ()):
            self.space.add(body)
        for shape in getattr(entity, "shapes", ()):
            self.space.add(shape)
        self.entities.append(entity)
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/blueball/world.py tests/test_player.py
git commit -m "feat: World.add_entity wires entity._world for space queries"
```

---

## Task 5: Player ShapeFilter group for raycast self-exclusion

**Goal:** Assign the Player's circle shape a unique `pymunk.ShapeFilter` group so subsequent raycasts (Task 21) can exclude the player's own shape without per-hit Python filtering.

**Files:**
- Modify: `src/blueball/entities/player.py`
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] Module constant `_PLAYER_RAY_GROUP = 1` exported.
- [ ] `Player.shape.filter.group == _PLAYER_RAY_GROUP` after construction.
- [ ] `Player._ray_filter` is a `pymunk.ShapeFilter(group=_PLAYER_RAY_GROUP)` instance ready for use in `segment_query_first`.
- [ ] Sanity check: a segment query from the ball's center through the ball with `Player._ray_filter` returns None (the ball is excluded).

**Verify:** `pytest -q tests/test_player.py -v`

**Steps:**

- [ ] **Step 1: Write failing test**

Append to `tests/test_player.py`:

```python
def test_player_ray_filter_excludes_own_shape():
    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    # Ray from inside the ball going out; without the filter, the ball's own
    # circle shape would be the nearest hit.
    hit = w.space.segment_query_first(
        (100, 100), (200, 100), 0.5, p._ray_filter,
    )
    assert hit is None  # filter excluded our own shape; nothing else in world
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Wire up filter in Player**

At top of `src/blueball/entities/player.py` (module level, after imports):

```python
_PLAYER_RAY_GROUP = 1
```

Inside `Player.__init__`, right after `self.shape = pymunk.Circle(...)`, add:

```python
        self.shape.filter = pymunk.ShapeFilter(group=_PLAYER_RAY_GROUP)
        self._ray_filter = pymunk.ShapeFilter(group=_PLAYER_RAY_GROUP)
```

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: Player shape filter group enables raycast self-exclusion"
```

---

## Task 6: Spike orientation parameter

**Goal:** Add `orientation: str = "up"` to `Spike.__init__`; the vertex set rotates to match. Existing `spike_pit` chunk usage stays default-up (no API regression).

**Files:**
- Modify: `src/blueball/entities/spike.py`
- Modify: `tests/test_entities.py`

**Acceptance Criteria:**
- [ ] `Spike(world, position, width, height, orientation)` accepts "up" | "down" | "left" | "right".
- [ ] Vertex set is rotated so spike point faces away from the wall:
  - "up": base at bottom, tip at top (current behavior).
  - "down": base at top, tip at bottom.
  - "left": base on right, tip on left.
  - "right": base on left, tip on right.
- [ ] Default orientation is "up" (preserves all existing tests).
- [ ] Invalid orientation raises `ValueError`.

**Verify:** `pytest -q tests/test_entities.py -v`

**Steps:**

- [ ] **Step 1: Read existing Spike to see its current vertex generation**

Read `src/blueball/entities/spike.py` and note the current triangle vertices.

- [ ] **Step 2: Write failing tests**

Append to `tests/test_entities.py`:

```python
def test_spike_default_orientation_is_up():
    from blueball.entities.spike import Spike
    w = World()
    s = Spike(w, position=(100, 600), width=24, height=24)
    # Default-up tip is above the position (y less than position.y)
    verts = [s.shape.get_vertices()[i] for i in range(3)]
    tip = min(verts, key=lambda v: v.y)
    assert tip.y < 0  # tip is "up" in local coords (relative to body at position.y)


def test_spike_orientation_down_inverts_tip():
    from blueball.entities.spike import Spike
    w = World()
    s = Spike(w, position=(100, 100), width=24, height=24, orientation="down")
    verts = [s.shape.get_vertices()[i] for i in range(3)]
    tip = max(verts, key=lambda v: v.y)
    assert tip.y > 0  # tip is "down"


def test_spike_orientation_left_and_right():
    from blueball.entities.spike import Spike
    w = World()
    s_left = Spike(w, position=(100, 100), width=24, height=24, orientation="left")
    s_right = Spike(w, position=(100, 100), width=24, height=24, orientation="right")
    verts_l = [s_left.shape.get_vertices()[i] for i in range(3)]
    verts_r = [s_right.shape.get_vertices()[i] for i in range(3)]
    assert min(v.x for v in verts_l) < 0
    assert max(v.x for v in verts_r) > 0


def test_spike_invalid_orientation_raises():
    import pytest
    from blueball.entities.spike import Spike
    w = World()
    with pytest.raises(ValueError):
        Spike(w, position=(100, 100), width=24, height=24, orientation="diagonal")
```

- [ ] **Step 3: Run, confirm failures**

- [ ] **Step 4: Implement orientation**

In `src/blueball/entities/spike.py`, modify `__init__` to accept `orientation: str = "up"` and rotate the vertex list:

```python
def _spike_verts(width: int, height: int, orientation: str) -> list[tuple[float, float]]:
    """Triangle vertices in body-local coordinates. Orientation names face the
    *tip direction* (where the point is). Pymunk y-down so 'up' is negative y.
    """
    hw = width / 2
    if orientation == "up":
        return [(-hw, 0), (hw, 0), (0, -height)]
    if orientation == "down":
        return [(-hw, 0), (hw, 0), (0, height)]
    if orientation == "left":
        return [(0, -hw), (0, hw), (-height, 0)]
    if orientation == "right":
        return [(0, -hw), (0, hw), (height, 0)]
    raise ValueError(
        f"Spike orientation must be 'up', 'down', 'left', or 'right'; got {orientation!r}"
    )
```

Replace the existing vertex computation with a call to `_spike_verts(width, height, orientation)` and store `self.orientation = orientation` for the renderer.

- [ ] **Step 5: Run tests, confirm pass**

Run all tests: `pytest -q tests/ -v` — confirm `spike_pit` tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/spike.py tests/test_entities.py
git commit -m "feat: Spike accepts orientation parameter"
```

---

## Task 7: New collision-type constants in collision.py

**Goal:** Add the 8 new `CT_*` constants per spec. No handlers yet — they arrive with their owning entity tasks.

**Files:**
- Modify: `src/blueball/collision.py`
- Modify: `tests/test_collision.py`

**Acceptance Criteria:**
- [ ] `CT_ONE_WAY = 8`, `CT_SPRING = 9`, `CT_PUSHABLE = 10`, `CT_SWINGING = 11`, `CT_CHARGER = 12`, `CT_CHECKPOINT = 13`, `CT_KEY = 14`, `CT_DOOR = 15` are defined in `collision.py`.
- [ ] All 8 are distinct from each other and from existing CTs.

**Verify:** `pytest -q tests/test_collision.py -v`

**Steps:**

- [ ] **Step 1: Write test**

Append to `tests/test_collision.py`:

```python
def test_all_collision_type_constants_distinct():
    from blueball import collision as col
    names = [
        "CT_PLAYER", "CT_SPIKE", "CT_PATROLLER", "CT_COLLECTIBLE",
        "CT_GOAL", "CT_BOOST_PAD", "CT_ABILITY_PICKUP",
        "CT_ONE_WAY", "CT_SPRING", "CT_PUSHABLE", "CT_SWINGING",
        "CT_CHARGER", "CT_CHECKPOINT", "CT_KEY", "CT_DOOR",
    ]
    values = [getattr(col, n) for n in names]
    assert len(set(values)) == len(names)
    assert col.CT_ONE_WAY == 8
    assert col.CT_DOOR == 15
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Add constants**

Append to `src/blueball/collision.py` (right after the existing CT constants):

```python
CT_ONE_WAY = 8
CT_SPRING = 9
CT_PUSHABLE = 10
CT_SWINGING = 11
CT_CHARGER = 12
CT_CHECKPOINT = 13
CT_KEY = 14
CT_DOOR = 15
```

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/collision.py tests/test_collision.py
git commit -m "feat: add Phase 3 collision-type constants"
```

---

## Task 8: `platform` chunk — floating segment, no ground

**Goal:** A `platform` chunk places one static `pymunk.Segment` at `GROUND_Y - y_offset` spanning `width_tiles`, with NO ground segment beneath. Foundational for vertical sections.

**Files:**
- Create: `src/blueball/levels/chunks/platform.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `platform` registered in `CHUNK_REGISTRY`.
- [ ] `Platform(width_tiles=4, y_offset=96).build(world, x_offset=0)` adds exactly ONE static segment at `y == GROUND_Y - 96`, returns width `4 * TILE`, and adds no other shapes.
- [ ] `Platform.difficulty == 0`, `Platform.sampler_include is True`.
- [ ] `Platform.random_params(rng)` returns `{"width_tiles": rng.randint(3, 5), "y_offset": rng.choice([64, 96, 128])}`.

**Verify:** `pytest -q tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Write failing test**

Append to `tests/test_chunks.py`:

```python
def test_platform_chunk_adds_one_floating_segment():
    from blueball.levels.chunks.platform import Platform
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    width = Platform(width_tiles=4, y_offset=96).build(w, x_offset=100)
    assert width == 4 * TILE
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 1
    a, b = segs[0].a, segs[0].b
    assert a.y == GROUND_Y - 96
    assert b.y == GROUND_Y - 96


def test_platform_chunk_registry_and_attributes():
    from blueball.levels.chunks.platform import Platform
    assert "platform" in CHUNK_REGISTRY
    assert Platform.difficulty == 0
    assert Platform.sampler_include is True
    import random as _rng
    params = Platform.random_params(_rng.Random(0))
    assert 3 <= params["width_tiles"] <= 5
    assert params["y_offset"] in (64, 96, 128)
```

- [ ] **Step 2: Run, confirm failures**

- [ ] **Step 3: Create the chunk**

Create `src/blueball/levels/chunks/platform.py`:

```python
"""Platform — a single floating horizontal segment. No ground beneath; the
chunk consumes its horizontal slot but the level's ground is invisible here.
"""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("platform")
class Platform(Chunk):
    difficulty: int = 0

    def __init__(self, width_tiles: int = 4, y_offset: int = 96) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "y_offset": rng.choice([64, 96, 128])}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        y = GROUND_Y - self.y_offset
        seg = pymunk.Segment(world.space.static_body, (x_offset, y), (x_offset + w, y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        return w
```

Add to `src/blueball/levels/chunks/__init__.py` import list: `platform,`

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/chunks/platform.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat: platform chunk — floating segment at y_offset"
```

---

## Task 9: `ice_floor` chunk — low-friction ground

**Goal:** `ice_floor` places a `flat`-style ground segment with `friction = config.ICE_FLOOR_FRICTION`.

**Files:**
- Create: `src/blueball/levels/chunks/ice_floor.py`
- Modify: `src/blueball/config.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `config.ICE_FLOOR_FRICTION == 0.05`.
- [ ] `ice_floor` registered; adds one segment at `GROUND_Y` with `friction == ICE_FLOOR_FRICTION`.
- [ ] `IceFloor.difficulty == 1`, sampler-includable, `random_params` returns `{"width_tiles": rng.randint(2, 5)}`.

**Verify:** `pytest -q tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Write failing test**

```python
def test_ice_floor_chunk_uses_low_friction():
    from blueball.levels.chunks.ice_floor import IceFloor
    from blueball.levels.chunks.flat import GROUND_Y
    from blueball import config
    w = World()
    IceFloor(width_tiles=3).build(w, x_offset=0)
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 1
    assert segs[0].friction == config.ICE_FLOOR_FRICTION
    assert segs[0].a.y == GROUND_Y


def test_ice_floor_attributes():
    from blueball.levels.chunks.ice_floor import IceFloor
    assert "ice_floor" in CHUNK_REGISTRY
    assert IceFloor.difficulty == 1
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Add config tunable**

Append to `src/blueball/config.py` under a new section:

```python
# Phase 3 chunks
ICE_FLOOR_FRICTION = 0.05
SPRING_DEFAULT_IMPULSE = 600.0
CRUMBLE_DEFAULT_DELAY_S = 0.5
MOVING_PLATFORM_DEFAULT_SPEED = 80.0
CHARGER_DEFAULT_SIGHT_RANGE = 200.0
CHARGER_DEFAULT_SIGHT_ARC_DEG = 60.0
CHARGER_DEFAULT_CHARGE_SPEED = 180.0
CHARGER_DEFAULT_PATROL_SPEED = 40.0

# Observation
MAX_RAY_LEN = 300.0
NUM_RAYS = 8
```

- [ ] **Step 4: Create the chunk**

Create `src/blueball/levels/chunks/ice_floor.py`:

```python
"""IceFloor — flat ground with very low friction. Momentum-management chunk."""

from __future__ import annotations

import pymunk

from ... import config
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("ice_floor")
class IceFloor(Chunk):
    difficulty: int = 1

    def __init__(self, width_tiles: int = 4) -> None:
        self.width_tiles = width_tiles

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 5)}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = config.ICE_FLOOR_FRICTION
        world.space.add(seg)
        return w
```

Add `ice_floor,` to the `__init__.py` import list.

- [ ] **Step 5: Run tests, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/config.py src/blueball/levels/chunks/ice_floor.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat: ice_floor chunk + Phase 3 config tunables"
```

---

## Task 10: `vertical_column` chunk — stacked platforms macro

**Goal:** Macro chunk that places `steps` floating platforms inside its horizontal slot, alternating left and right hugs, each `step_height` apart vertically. Built on top of the `platform` primitive.

**Files:**
- Create: `src/blueball/levels/chunks/vertical_column.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `vertical_column` registered; difficulty 2.
- [ ] `VerticalColumn(width_tiles=6, steps=4, step_height=80, bottom_offset=96, platform_tiles=2).build(world, x_offset=0)` returns `6 * TILE` and adds exactly 4 segments, at y values `GROUND_Y - (96 + i*80)` for i in 0..3.
- [ ] Segments alternate left/right: even-i platforms span [0, platform_tiles*TILE]; odd-i platforms span [(width_tiles - platform_tiles)*TILE, width_tiles*TILE].

**Verify:** `pytest -q tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_vertical_column_builds_alternating_platforms():
    from blueball.levels.chunks.vertical_column import VerticalColumn
    from blueball.levels.chunks.flat import GROUND_Y
    w = World()
    width = VerticalColumn(width_tiles=6, steps=4, step_height=80, bottom_offset=96, platform_tiles=2).build(w, x_offset=0)
    assert width == 6 * TILE
    segs = [s for s in w.space.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 4
    # Sorted by y descending (bottom to top in pymunk y-down): smallest y is highest.
    segs_sorted = sorted(segs, key=lambda s: -s.a.y)  # bottom first
    ys = [s.a.y for s in segs_sorted]
    assert ys == [GROUND_Y - 96, GROUND_Y - 176, GROUND_Y - 256, GROUND_Y - 336]
    # Even-index hug left (start at x=0); odd-index hug right (end at width)
    assert segs_sorted[0].a.x == 0
    assert segs_sorted[1].b.x == 6 * TILE
    assert segs_sorted[2].a.x == 0
    assert segs_sorted[3].b.x == 6 * TILE
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Create the chunk**

```python
"""VerticalColumn — N stacked floating platforms inside the chunk's horizontal
slot, alternating left/right hugs so the player must zig-zag jump.
"""

from __future__ import annotations

import pymunk

from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("vertical_column")
class VerticalColumn(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 6,
        steps: int = 5,
        step_height: int = 80,
        bottom_offset: int = 96,
        platform_tiles: int = 2,
    ) -> None:
        self.width_tiles = width_tiles
        self.steps = steps
        self.step_height = step_height
        self.bottom_offset = bottom_offset
        self.platform_tiles = platform_tiles

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": 6,
            "steps": rng.randint(3, 6),
            "step_height": rng.choice([64, 80, 96]),
            "bottom_offset": 96,
            "platform_tiles": 2,
        }

    def build(self, world, x_offset: float) -> float:
        slot_w = self.width_tiles * TILE
        plat_w = self.platform_tiles * TILE
        for i in range(self.steps):
            y = GROUND_Y - (self.bottom_offset + i * self.step_height)
            if i % 2 == 0:
                a = (x_offset, y)
                b = (x_offset + plat_w, y)
            else:
                a = (x_offset + slot_w - plat_w, y)
                b = (x_offset + slot_w, y)
            seg = pymunk.Segment(world.space.static_body, a, b, 5)
            seg.friction = 1.0
            world.space.add(seg)
        return slot_w
```

Add `vertical_column,` to `__init__.py`.

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/levels/chunks/vertical_column.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat: vertical_column chunk for vertical sections"
```

---

## Task 11: OneWayPlatform entity + chunk + pre_solve filter

**Goal:** `OneWayPlatform` is a static segment that allows passage from below (player rising) and blocks from above (player descending). Implemented via pymunk's `pre_solve` callback returning False for the rising case.

**Files:**
- Create: `src/blueball/entities/one_way_platform.py`
- Create: `src/blueball/levels/chunks/one_way_platform.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`
- Modify: `tests/test_collision.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `OneWayPlatform` entity exposes one static segment shape with `collision_type == CT_ONE_WAY`.
- [ ] Collision handler `on_one_way` is a pre_solve callback (NOT begin): returns False when the dynamic body's `velocity.y < 0` (rising in pymunk y-down), True otherwise.
- [ ] Player rising into the platform passes through; player falling onto it lands solidly.
- [ ] `one_way_platform` chunk registered; difficulty 1; `random_params` returns randomized `width_tiles` and `y_offset`.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

Append to `tests/test_entities.py`:

```python
def test_one_way_platform_entity_has_correct_collision_type():
    from blueball.entities.one_way_platform import OneWayPlatform
    from blueball import collision as col
    w = World()
    p = OneWayPlatform(w, position=(100, 500), width=128)
    assert p.shape.collision_type == col.CT_ONE_WAY
    assert p.shape.body.body_type == pymunk.Body.STATIC
```

Append to `tests/test_collision.py`:

```python
def test_one_way_platform_passes_rising_player():
    """Player rising (velocity.y < 0) should pass through; falling should land."""
    from blueball.entities.one_way_platform import OneWayPlatform
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register
    from blueball.world import World

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    register(w.space, world_ref=w)
    plat = OneWayPlatform(w, position=(100, 500), width=200)
    w.add_entity(plat)
    p = Player(agent=Idle(), spawn_xy=(100, 540))
    w.add_entity(p)
    # Give player upward velocity (rising in pymunk y-down)
    p.body.velocity = (0, -300)
    starting_y = p.body.position.y
    for _ in range(15):
        w.step(1 / 60)
    # The player should have passed through the platform y=500 (now y < 500)
    assert p.body.position.y < 500
```

Append to `tests/test_chunks.py`:

```python
def test_one_way_platform_chunk_registered():
    from blueball.levels.chunks.one_way_platform import OneWayPlatformChunk
    assert "one_way_platform" in CHUNK_REGISTRY
    assert OneWayPlatformChunk.difficulty == 1
```

- [ ] **Step 2: Run, confirm failures**

- [ ] **Step 3: Create entity**

Create `src/blueball/entities/one_way_platform.py`:

```python
"""OneWayPlatform — a static segment that allows passage from below and blocks
from above. Implemented as a single Segment with CT_ONE_WAY; the collision
dispatcher's pre_solve handler filters out 'rising' contacts.
"""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class OneWayPlatform(Entity):
    def __init__(self, world, position: tuple[float, float], width: float) -> None:
        super().__init__()
        cx, cy = position
        hw = width / 2
        # Static-body segment; the body itself is space.static_body so we don't
        # create a new pymunk.Body — just the shape.
        self.shape = pymunk.Segment(
            world.space.static_body,
            (cx - hw, cy), (cx + hw, cy), 5,
        )
        self.shape.collision_type = _col.CT_ONE_WAY
        self.shape.friction = 1.0
        self.shapes.append(self.shape)
        self.position = position
        self.width = width
```

- [ ] **Step 4: Add pre_solve handler in collision.py**

Inside `register(space, world_ref)`, append:

```python
    def on_one_way_presolve(arbiter, space_, data):
        # In pymunk, arbiter.shapes is (shape_a, shape_b). The player (or any
        # dynamic body) is the non-static shape; we identify it by body_type.
        for shape in arbiter.shapes:
            if shape.body.body_type == pymunk.Body.DYNAMIC:
                # Rising = velocity.y < 0 in pymunk y-down. Disable contact.
                if shape.body.velocity.y < 0:
                    return False
        return True

    space.on_collision(
        collision_type_a=CT_PLAYER, collision_type_b=CT_ONE_WAY,
        pre_solve=on_one_way_presolve,
    )
    space.on_collision(
        collision_type_a=CT_PUSHABLE, collision_type_b=CT_ONE_WAY,
        pre_solve=on_one_way_presolve,
    )
```

Add `import pymunk` at top if not already there (it is).

- [ ] **Step 5: Create chunk**

Create `src/blueball/levels/chunks/one_way_platform.py`:

```python
"""one_way_platform chunk — places one OneWayPlatform sensor segment."""

from __future__ import annotations

from ...entities.one_way_platform import OneWayPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("one_way_platform")
class OneWayPlatformChunk(Chunk):
    difficulty: int = 1

    def __init__(self, width_tiles: int = 4, y_offset: int = 96) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(3, 5), "y_offset": rng.choice([64, 96, 128])}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        y = GROUND_Y - self.y_offset
        world.add_entity(OneWayPlatform(world, position=(x_offset + w / 2, y), width=w))
        return w
```

Add `one_way_platform,` to `__init__.py`.

- [ ] **Step 6: Run tests, confirm pass**

- [ ] **Step 7: Commit**

```bash
git add src/blueball/entities/one_way_platform.py src/blueball/levels/chunks/one_way_platform.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/test_entities.py tests/test_collision.py tests/test_chunks.py
git commit -m "feat: one_way_platform — jump through from below, land from above"
```

---

## Task 12: Spring entity + chunk + handler

**Goal:** `Spring` is a sensor strip; contact with any dynamic body applies a mass-scaled upward impulse so delta-v is identical across body masses. Re-triggerable per contact.

**Files:**
- Create: `src/blueball/entities/spring.py`
- Create: `src/blueball/levels/chunks/spring.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`
- Modify: `tests/test_collision.py`

**Acceptance Criteria:**
- [ ] `Spring(world, position, width, impulse)` exposes a static sensor Poly with `CT_SPRING`.
- [ ] Collision handler `on_spring`: Player → `player.receive_spring(spring.impulse)`; other dynamic bodies → `body.apply_impulse_at_local_point((0, -spring.impulse * body.mass), (0, 0))`. Sensor (returns False).
- [ ] Re-triggerable: re-entering the spring after leaving applies the impulse again.
- [ ] Chunk: `spring` registered, difficulty 1, sampler-includable.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_spring_entity_is_sensor_with_ct_spring():
    from blueball.entities.spring import Spring
    from blueball import collision as col
    w = World()
    s = Spring(w, position=(100, 600), width=64, impulse=400.0)
    assert s.shape.sensor is True
    assert s.shape.collision_type == col.CT_SPRING


def test_spring_collision_launches_player_upward():
    from blueball.entities.spring import Spring
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register
    from blueball.world import World

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    register(w.space, world_ref=w)
    s = Spring(w, position=(100, 596), width=64, impulse=600.0)
    w.add_entity(s)
    p = Player(agent=Idle(), spawn_xy=(100, 580))
    w.add_entity(p)
    p.body.velocity = (0, 0)
    w.step(1 / 60)
    # After contact, the player should have a strong upward (negative-y) velocity
    assert p.body.velocity.y < -200
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/spring.py`:

```python
"""Spring — sensor strip that launches any dynamic body upward on contact."""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class Spring(Entity):
    def __init__(self, world, position: tuple[float, float], width: float, impulse: float) -> None:
        super().__init__()
        self.impulse = impulse
        self.position = position
        self.width = width
        cx, cy = position
        hw = width / 2
        half_thick = 8
        verts = [
            (-hw, -half_thick),
            (hw, -half_thick),
            (hw, half_thick),
            (-hw, half_thick),
        ]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        self.shape = pymunk.Poly(body, verts)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_SPRING
        self.bodies.append(body)
        self.shapes.append(self.shape)
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/spring.py`:

```python
"""spring chunk — flat ground with a sensor Spring on top."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.spring import Spring
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("spring")
class SpringChunk(Chunk):
    difficulty: int = 1

    def __init__(self, width_tiles: int = 2, impulse: float = config.SPRING_DEFAULT_IMPULSE) -> None:
        self.width_tiles = width_tiles
        self.impulse = impulse

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 3), "impulse": rng.choice([500.0, 600.0, 720.0])}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(Spring(
            world,
            position=(x_offset + w / 2, GROUND_Y - 8),
            width=w,
            impulse=self.impulse,
        ))
        return w
```

Add to `__init__.py`: `spring,`

- [ ] **Step 4: Wire collision handler**

In `src/blueball/collision.py`, inside `register(...)`:

```python
    def on_spring(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        # Find the Spring entity for the impulse value
        spring_entity = None
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is not None and hasattr(entity, "impulse") and shape.collision_type == CT_SPRING:
                spring_entity = entity
                break
        if spring_entity is None:
            return False
        if player is not None:
            player.receive_spring(spring_entity.impulse)
        # Also handle non-player dynamic bodies (pushable boxes)
        for shape in arbiter.shapes:
            if shape.body.body_type != pymunk.Body.DYNAMIC:
                continue
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            shape.body.apply_impulse_at_local_point(
                (0, -spring_entity.impulse * shape.body.mass), (0, 0)
            )
        return False  # sensor

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_SPRING, begin=on_spring)
    space.on_collision(collision_type_a=CT_PUSHABLE, collision_type_b=CT_SPRING, begin=on_spring)
```

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/spring.py src/blueball/levels/chunks/spring.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: spring — sensor strip that launches any dynamic body upward"
```

---

## Task 13: Checkpoint entity + chunk + handler

**Goal:** `Checkpoint` is a sensor; contact sets `player.respawn_xy` (in-memory). Never persisted to save.

**Files:**
- Create: `src/blueball/entities/checkpoint.py`
- Create: `src/blueball/levels/chunks/checkpoint.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_collision.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `Checkpoint(world, position, id)` is a static sensor Circle with `CT_CHECKPOINT`.
- [ ] Contact sets `player.respawn_xy = (checkpoint.body.position.x, GROUND_Y - BALL_RADIUS - 4)`.
- [ ] No save-file write occurs on checkpoint contact.
- [ ] Chunk: `checkpoint`, `sampler_include = False` (the sampler sprinkles checkpoints via its own rule), difficulty 0.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_checkpoint_entity_is_sensor_with_ct_checkpoint():
    from blueball.entities.checkpoint import Checkpoint
    from blueball import collision as col
    w = World()
    cp = Checkpoint(w, position=(100, 540), id=0)
    assert cp.shape.sensor is True
    assert cp.shape.collision_type == col.CT_CHECKPOINT


def test_checkpoint_contact_sets_player_respawn_xy(monkeypatch, tmp_path):
    from blueball.entities.checkpoint import Checkpoint
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register
    from blueball.world import World
    from blueball import config

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    save_file = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_file))

    w = World()
    register(w.space, world_ref=w)
    cp = Checkpoint(w, position=(300, 540), id=1)
    w.add_entity(cp)
    p = Player(agent=Idle(), spawn_xy=(300, 540))
    w.add_entity(p)
    w.step(1 / 60)
    assert p.respawn_xy is not None
    assert p.respawn_xy[0] == 300
    assert not save_file.exists()  # never persisted
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/checkpoint.py`:

```python
"""Checkpoint — sensor that updates player.respawn_xy on contact."""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class Checkpoint(Entity):
    def __init__(self, world, position: tuple[float, float], id: int = 0, radius: int = 18) -> None:
        super().__init__()
        self.id = id
        self.position = position
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        self.shape = pymunk.Circle(body, radius)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_CHECKPOINT
        self.bodies.append(body)
        self.shapes.append(self.shape)
        self.activated = False
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/checkpoint.py`:

```python
"""checkpoint chunk — ground segment + Checkpoint sensor."""

from __future__ import annotations

import pymunk

from ...entities.checkpoint import Checkpoint
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("checkpoint")
class CheckpointChunk(Chunk):
    difficulty: int = 0
    sampler_include: bool = False  # sampler sprinkles via a separate rule

    def __init__(self, width_tiles: int = 2, y_offset: int = 64, id: int = 0) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset
        self.id = id

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(Checkpoint(world, position=(x_offset + w / 2, GROUND_Y - self.y_offset), id=self.id))
        return w
```

Add `checkpoint,` to `__init__.py`.

- [ ] **Step 4: Wire collision handler**

In `collision.py` inside `register(...)`:

```python
    def on_checkpoint(arbiter, space_, data):
        # GROUND_Y lives in a level-chunk module; import locally to avoid
        # collision -> chunks -> entities cycles.
        from .levels.chunks.flat import GROUND_Y
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if hasattr(entity, "id") and shape.collision_type == CT_CHECKPOINT:
                if player is not None:
                    cx = shape.body.position.x
                    player.respawn_xy = (cx, GROUND_Y - config.BALL_RADIUS - 4)
                    entity.activated = True
        return False  # sensor

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_CHECKPOINT, begin=on_checkpoint)
```

(Note the local imports avoid a cycle between `collision` and `config`/`levels.chunks.flat`.)

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/checkpoint.py src/blueball/levels/chunks/checkpoint.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: checkpoint sensor sets player.respawn_xy (no save)"
```

---

## Task 14: CrumblingPlatform entity + chunk

**Goal:** Static platform segment that starts a timer on first dynamic-body contact and removes itself when the timer expires. Removal happens inside `update()` (not the collision callback) to avoid pymunk's mid-step shape-removal hazard.

**Files:**
- Create: `src/blueball/entities/crumbling_platform.py`
- Create: `src/blueball/levels/chunks/crumbling_platform.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `CrumblingPlatform(world, position, width, crumble_delay_s)` exposes a static segment shape (no new CT — collides as ground via default rules).
- [ ] On first per-tick `update()` call where the segment is in contact with a dynamic body, the entity sets `self._contacted_at = elapsed_time`.
- [ ] When `elapsed - _contacted_at >= crumble_delay_s`, the entity removes its shape from the space and sets `self._removed = True`. Subsequent updates are no-ops.

**Verify:** `pytest -q tests/test_entities.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_crumbling_platform_removes_after_delay():
    from blueball.entities.crumbling_platform import CrumblingPlatform
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.world import World

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    cp = CrumblingPlatform(w, position=(100, 600), width=100, crumble_delay_s=0.1)
    w.add_entity(cp)
    p = Player(agent=Idle(), spawn_xy=(100, 580))
    w.add_entity(p)
    # Let the player land and sit on the platform
    for _ in range(20):
        w.step(1 / 60)
    assert cp._removed is False  # still around
    # Continue stepping past the crumble delay
    for _ in range(30):
        w.step(1 / 60)
    assert cp._removed is True
    assert cp.shape not in w.space.shapes
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/crumbling_platform.py`:

```python
"""CrumblingPlatform — static segment that despawns after timed contact."""

from __future__ import annotations

import pymunk

from .. import config
from .base import Entity


class CrumblingPlatform(Entity):
    def __init__(self, world, position: tuple[float, float], width: float, crumble_delay_s: float = config.CRUMBLE_DEFAULT_DELAY_S) -> None:
        super().__init__()
        self.position = position
        self.width = width
        self.crumble_delay_s = crumble_delay_s
        cx, cy = position
        hw = width / 2
        self.shape = pymunk.Segment(
            world.space.static_body,
            (cx - hw, cy), (cx + hw, cy), 5,
        )
        self.shape.friction = 1.0
        self.shapes.append(self.shape)
        self._contacted = False
        self._contact_timer: float = 0.0
        self._removed = False
        self._world_ref = world

    def update(self, dt: float) -> None:
        if self._removed:
            return
        if not self._contacted:
            # Static-body shapes can't use each_arbiter directly across all
            # contacts. Pragmatic probe: AABB-overlap against any dynamic body
            # in the space. Cheap, and a false positive here just starts the
            # crumble timer slightly early — acceptable trade.
            in_contact = False
            for shape in self._world_ref.space.shapes:
                if shape is self.shape:
                    continue
                if shape.body.body_type != pymunk.Body.DYNAMIC:
                    continue
                # Bounding-box overlap is enough — the contact happened if the
                # AABB overlaps the segment's AABB.
                if self.shape.bb.intersects(shape.bb):
                    in_contact = True
                    break
            if in_contact:
                self._contacted = True
        else:
            self._contact_timer += dt
            if self._contact_timer >= self.crumble_delay_s:
                # Safe removal — we're outside any collision callback.
                if self.shape in self._world_ref.space.shapes:
                    self._world_ref.space.remove(self.shape)
                self._removed = True
```

(The "AABB intersects" probe is a pragmatic contact check that avoids the mid-step removal hazard of using collision callbacks for this entity.)

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/crumbling_platform.py`:

```python
"""crumbling_platform chunk — places a CrumblingPlatform entity."""

from __future__ import annotations

from ... import config
from ...entities.crumbling_platform import CrumblingPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("crumbling_platform")
class CrumblingPlatformChunk(Chunk):
    difficulty: int = 2

    def __init__(self, width_tiles: int = 2, y_offset: int = 96, crumble_delay_s: float = config.CRUMBLE_DEFAULT_DELAY_S) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset
        self.crumble_delay_s = crumble_delay_s

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 3), "y_offset": rng.choice([0, 64, 96]), "crumble_delay_s": round(rng.uniform(0.3, 0.7), 2)}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        y = GROUND_Y - self.y_offset
        world.add_entity(CrumblingPlatform(world, position=(x_offset + w / 2, y), width=w, crumble_delay_s=self.crumble_delay_s))
        return w
```

Add `crumbling_platform,` to `__init__.py`.

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/crumbling_platform.py src/blueball/levels/chunks/crumbling_platform.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: crumbling_platform — timed self-removal on contact"
```

---

## Task 15: MovingPlatform entity + chunk

**Goal:** Kinematic body that oscillates between two waypoints along a configurable axis at a configurable speed. Player rides via pymunk's default surface-friction.

**Files:**
- Create: `src/blueball/entities/moving_platform.py`
- Create: `src/blueball/levels/chunks/moving_platform.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `MovingPlatform(world, position, length, axis, range_px, speed)` exposes a kinematic body with a single Segment shape.
- [ ] `update(dt)` advances the body along `axis` between `-range_px/2` and `+range_px/2` of spawn, reversing at bounds; constant speed via `body.velocity`.
- [ ] Position stays within `[spawn - range_px/2, spawn + range_px/2]` after N steps.
- [ ] Chunk: `moving_platform` registered, difficulty 2.

**Verify:** `pytest -q tests/test_entities.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_moving_platform_oscillates_along_x():
    from blueball.entities.moving_platform import MovingPlatform
    w = World()
    mp = MovingPlatform(w, position=(500, 500), length=64, axis="x", range_px=200, speed=120)
    w.add_entity(mp)
    spawn_x = mp.body.position.x
    for _ in range(200):
        w.step(1 / 60)
    # After several seconds we expect to have moved and bounded
    assert abs(mp.body.position.x - spawn_x) <= 100 + 1  # within range bound
    assert mp.body.position.x != spawn_x  # actually moved


def test_moving_platform_chunk_registered():
    from blueball.levels.chunks.moving_platform import MovingPlatformChunk
    assert "moving_platform" in CHUNK_REGISTRY
    assert MovingPlatformChunk.difficulty == 2
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/moving_platform.py`:

```python
"""MovingPlatform — kinematic body oscillating between two waypoints."""

from __future__ import annotations

import pymunk

from .base import Entity


class MovingPlatform(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        length: float,
        axis: str = "x",
        range_px: float = 160,
        speed: float = 80,
    ) -> None:
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        self.axis = axis
        self.range_px = range_px
        self.speed = speed
        self.length = length
        self._spawn = position
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        hl = length / 2
        # Lay the segment horizontally regardless of axis — it's the platform's
        # visible width; vertical movers also use a horizontal top edge.
        self.shape = pymunk.Segment(body, (-hl, 0), (hl, 0), 5)
        self.shape.friction = 1.0
        self.bodies.append(body)
        self.shapes.append(self.shape)
        # Initial velocity along axis
        if axis == "x":
            body.velocity = (speed, 0)
        else:
            body.velocity = (0, speed)
        self.body = body

    def update(self, dt: float) -> None:
        if self.axis == "x":
            delta = self.body.position.x - self._spawn[0]
            if delta > self.range_px / 2 and self.body.velocity.x > 0:
                self.body.velocity = (-self.speed, 0)
            elif delta < -self.range_px / 2 and self.body.velocity.x < 0:
                self.body.velocity = (self.speed, 0)
        else:
            delta = self.body.position.y - self._spawn[1]
            if delta > self.range_px / 2 and self.body.velocity.y > 0:
                self.body.velocity = (0, -self.speed)
            elif delta < -self.range_px / 2 and self.body.velocity.y < 0:
                self.body.velocity = (0, self.speed)
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/moving_platform.py`:

```python
"""moving_platform chunk — places a MovingPlatform entity."""

from __future__ import annotations

from ... import config
from ...entities.moving_platform import MovingPlatform
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("moving_platform")
class MovingPlatformChunk(Chunk):
    difficulty: int = 2

    def __init__(
        self,
        width_tiles: int = 4,
        length_tiles: int = 2,
        axis: str = "x",
        range_px: float = 160,
        speed: float = config.MOVING_PLATFORM_DEFAULT_SPEED,
        y_offset: int = 96,
    ) -> None:
        self.width_tiles = width_tiles
        self.length_tiles = length_tiles
        self.axis = axis
        self.range_px = range_px
        self.speed = speed
        self.y_offset = y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(4, 6),
            "length_tiles": 2,
            "axis": rng.choice(["x", "y"]),
            "range_px": rng.choice([120, 160, 200]),
            "speed": rng.choice([60.0, 80.0, 100.0]),
            "y_offset": rng.choice([64, 96, 128]),
        }

    def build(self, world, x_offset: float) -> float:
        slot_w = self.width_tiles * TILE
        plat_len = self.length_tiles * TILE
        cx = x_offset + slot_w / 2
        cy = GROUND_Y - self.y_offset
        world.add_entity(MovingPlatform(world, position=(cx, cy), length=plat_len, axis=self.axis, range_px=self.range_px, speed=self.speed))
        return slot_w
```

Add `moving_platform,` to `__init__.py`.

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/moving_platform.py src/blueball/levels/chunks/moving_platform.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: moving_platform — kinematic oscillator on x or y axis"
```

---

## Task 16: PushableBox entity + chunk

**Goal:** `PushableBox` is a dynamic body that the player can push laterally. No custom collision handler needed — pymunk's default solid contact provides the push.

**Files:**
- Create: `src/blueball/entities/pushable_box.py`
- Create: `src/blueball/levels/chunks/pushable_box.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `PushableBox(world, position, size, mass)` exposes a dynamic body with a box `Poly` shape, `collision_type = CT_PUSHABLE`, `friction = 0.6`.
- [ ] Player rolling into the box at constant velocity displaces the box horizontally (box.position.x increases over time).
- [ ] Chunk: `pushable_box`, difficulty 2.

**Verify:** `pytest -q tests/test_entities.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_pushable_box_is_dynamic_with_ct_pushable():
    from blueball.entities.pushable_box import PushableBox
    from blueball import collision as col
    w = World()
    b = PushableBox(w, position=(100, 580), size=32, mass=0.5)
    assert b.body.body_type == pymunk.Body.DYNAMIC
    assert b.shape.collision_type == col.CT_PUSHABLE
    assert b.body.mass == 0.5


def test_player_pushes_box():
    from blueball.entities.pushable_box import PushableBox
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register

    class Press(Agent):
        def act(self, obs):
            return Action.RIGHT

    w = World()
    register(w.space, world_ref=w)
    # Ground floor under both
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    b = PushableBox(w, position=(200, 580), size=32, mass=0.5)
    w.add_entity(b)
    p = Player(agent=Press(), spawn_xy=(140, 580))
    w.add_entity(p)
    start_x = b.body.position.x
    for _ in range(120):
        w.step(1 / 60)
    assert b.body.position.x > start_x + 5
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/pushable_box.py`:

```python
"""PushableBox — dynamic body that the player can push laterally."""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class PushableBox(Entity):
    def __init__(self, world, position: tuple[float, float], size: float = 32, mass: float = 0.5) -> None:
        super().__init__()
        self.size = size
        moment = pymunk.moment_for_box(mass, (size, size))
        body = pymunk.Body(mass=mass, moment=moment)
        body.position = position
        hs = size / 2
        verts = [(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]
        self.shape = pymunk.Poly(body, verts)
        self.shape.friction = 0.6
        self.shape.collision_type = _col.CT_PUSHABLE
        self.bodies.append(body)
        self.shapes.append(self.shape)
        self.body = body
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/pushable_box.py`:

```python
"""pushable_box chunk — ground segment + a single PushableBox sitting on it."""

from __future__ import annotations

import pymunk

from ...entities.pushable_box import PushableBox
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("pushable_box")
class PushableBoxChunk(Chunk):
    difficulty: int = 2

    def __init__(self, width_tiles: int = 2, size_px: int = 32, mass: float = 0.5) -> None:
        self.width_tiles = width_tiles
        self.size_px = size_px
        self.mass = mass

    @classmethod
    def random_params(cls, rng) -> dict:
        return {"width_tiles": rng.randint(2, 3), "size_px": rng.choice([28, 32, 40]), "mass": round(rng.uniform(0.4, 0.8), 2)}

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        cx = x_offset + w / 2
        # Spawn slightly above ground so the box doesn't intersect the segment.
        world.add_entity(PushableBox(world, position=(cx, GROUND_Y - self.size_px / 2 - 1), size=self.size_px, mass=self.mass))
        return w
```

Add `pushable_box,` to `__init__.py`.

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/entities/pushable_box.py src/blueball/levels/chunks/pushable_box.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: pushable_box — dynamic body the player can push"
```

---

## Task 17: Key entity + chunk + handler

**Goal:** `Key` is a sensor; contact sets a bit in `player.keys_held` and removes itself.

**Files:**
- Create: `src/blueball/entities/key.py`
- Create: `src/blueball/levels/chunks/key.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_collision.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `Key(world, position, key_id, radius=18)` is a static sensor Circle with `CT_KEY` and `key_id` attribute.
- [ ] On player contact, `player.collect_key(key.key_id)` is called and `key._collected = True`. On the next tick, the entity removes its shape from the space.
- [ ] Chunk: `key`, `sampler_include = False`, difficulty 1.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_key_entity_is_sensor():
    from blueball.entities.key import Key
    from blueball import collision as col
    w = World()
    k = Key(w, position=(100, 540), key_id=2)
    assert k.shape.sensor is True
    assert k.shape.collision_type == col.CT_KEY
    assert k.key_id == 2


def test_key_contact_sets_player_bit_and_removes_entity():
    from blueball.entities.key import Key
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register
    from blueball.world import World

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    register(w.space, world_ref=w)
    k = Key(w, position=(150, 540), key_id=3)
    w.add_entity(k)
    p = Player(agent=Idle(), spawn_xy=(150, 540))
    w.add_entity(p)
    w.step(1 / 60)
    w.step(1 / 60)
    assert p.has_key(3) is True
    assert k._collected is True
    assert k.shape not in w.space.shapes
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/key.py`:

```python
"""Key — sensor pickup that sets a bit in player.keys_held on contact."""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class Key(Entity):
    def __init__(self, world, position: tuple[float, float], key_id: int = 0, radius: int = 18) -> None:
        super().__init__()
        self.key_id = key_id
        self.position = position
        self.radius = radius
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        self.shape = pymunk.Circle(body, radius)
        self.shape.sensor = True
        self.shape.collision_type = _col.CT_KEY
        self.bodies.append(body)
        self.shapes.append(self.shape)
        self._collected = False
        self._world_ref = world

    def update(self, dt: float) -> None:
        if self._collected and self.shape in self._world_ref.space.shapes:
            self._world_ref.space.remove(self.shape)
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/key.py`:

```python
"""key chunk — ground segment + Key sensor pickup."""

from __future__ import annotations

import pymunk

from ...entities.key import Key
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("key")
class KeyChunk(Chunk):
    difficulty: int = 1
    sampler_include: bool = False

    def __init__(self, width_tiles: int = 2, y_offset: int = 64, key_id: int = 0) -> None:
        self.width_tiles = width_tiles
        self.y_offset = y_offset
        self.key_id = key_id

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(Key(world, position=(x_offset + w / 2, GROUND_Y - self.y_offset), key_id=self.key_id))
        return w
```

Add `key,` to `__init__.py`.

- [ ] **Step 4: Wire collision handler**

In `collision.py` inside `register(...)`:

```python
    def on_key(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if hasattr(entity, "key_id") and shape.collision_type == CT_KEY:
                if player is not None and not entity._collected:
                    player.collect_key(entity.key_id)
                    entity._collected = True
        return False  # sensor

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_KEY, begin=on_key)
```

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/key.py src/blueball/levels/chunks/key.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: key sensor sets player.keys_held bit"
```

---

## Task 18: Door entity + chunk + handler

**Goal:** Solid vertical segment that opens when the player has the matching key.

**Files:**
- Create: `src/blueball/entities/door.py`
- Create: `src/blueball/levels/chunks/door.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_collision.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `Door(world, position, height, key_id)` exposes a static vertical Segment with `CT_DOOR`. `door.is_open` starts False.
- [ ] On first contact, if `player.has_key(door.key_id)`, the door's shape is queued for removal (`door._opening = True`); `door._opening` is consumed in `update()` to call `space.remove`.
- [ ] If player does NOT have the key, the contact is solid (handler returns True).
- [ ] Chunk: `door`, `sampler_include = False`, difficulty 0 (it's a gate, not a hazard).

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_door_blocks_player_without_key():
    from blueball.entities.door import Door
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register

    class Press(Agent):
        def act(self, obs):
            return Action.RIGHT

    w = World()
    register(w.space, world_ref=w)
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    d = Door(w, position=(300, 540), height=128, key_id=0)
    w.add_entity(d)
    p = Player(agent=Press(), spawn_xy=(200, 580))
    w.add_entity(p)
    for _ in range(120):
        w.step(1 / 60)
    # Player should be blocked by the door (didn't cross x=300)
    assert p.body.position.x < 300
    assert d.is_open is False


def test_door_opens_when_player_has_key():
    from blueball.entities.door import Door
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register

    class Press(Agent):
        def act(self, obs):
            return Action.RIGHT

    w = World()
    register(w.space, world_ref=w)
    floor = pymunk.Segment(w.space.static_body, (-2000, 600), (2000, 600), 5)
    floor.friction = 1.0
    w.space.add(floor)
    d = Door(w, position=(300, 540), height=128, key_id=0)
    w.add_entity(d)
    p = Player(agent=Press(), spawn_xy=(200, 580))
    p.collect_key(0)
    w.add_entity(p)
    for _ in range(180):
        w.step(1 / 60)
    assert d.is_open is True
    # And player has passed through
    assert p.body.position.x > 300
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/door.py`:

```python
"""Door — solid vertical barrier that opens when player has the matching key."""

from __future__ import annotations

import pymunk

from .. import collision as _col
from .base import Entity


class Door(Entity):
    def __init__(self, world, position: tuple[float, float], height: float, key_id: int = 0) -> None:
        super().__init__()
        self.position = position
        self.height = height
        self.key_id = key_id
        cx, cy = position
        hh = height / 2
        self.shape = pymunk.Segment(
            world.space.static_body,
            (cx, cy - hh), (cx, cy + hh), 5,
        )
        self.shape.collision_type = _col.CT_DOOR
        self.shape.friction = 1.0
        self.shapes.append(self.shape)
        self._world_ref = world
        self.is_open = False
        self._opening = False

    def update(self, dt: float) -> None:
        if self._opening and self.shape in self._world_ref.space.shapes:
            self._world_ref.space.remove(self.shape)
            self.is_open = True
            self._opening = False
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/door.py`:

```python
"""door chunk — vertical barrier that opens when player has key_id."""

from __future__ import annotations

import pymunk

from ...entities.door import Door
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("door")
class DoorChunk(Chunk):
    difficulty: int = 0
    sampler_include: bool = False

    def __init__(self, width_tiles: int = 2, height_tiles: int = 4, key_id: int = 0) -> None:
        self.width_tiles = width_tiles
        self.height_tiles = height_tiles
        self.key_id = key_id

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        h = self.height_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        cx = x_offset + w / 2
        world.add_entity(Door(world, position=(cx, GROUND_Y - h / 2), height=h, key_id=self.key_id))
        return w
```

Add `door,` to `__init__.py`.

- [ ] **Step 4: Wire collision handler**

In `collision.py` inside `register(...)`:

```python
    def on_door(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if hasattr(entity, "key_id") and shape.collision_type == CT_DOOR:
                if player is not None and not entity.is_open:
                    if player.has_key(entity.key_id):
                        entity._opening = True
                        return False  # pass through this contact
                    return True  # solid
        return True

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_DOOR, begin=on_door)
```

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/door.py src/blueball/levels/chunks/door.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: door opens on contact when player holds matching key"
```

---

## Task 19: SwingingHazard entity + chunk + handler

**Goal:** Pendulum: static anchor + dynamic bob + PinJoint. Bob is a sensor-killing shape (`CT_SWINGING`).

**Files:**
- Create: `src/blueball/entities/swinging_hazard.py`
- Create: `src/blueball/levels/chunks/swinging_hazard.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_collision.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `SwingingHazard(world, anchor_pos, rope_length, bob_mass, initial_angle_deg)` exposes a static anchor body, a dynamic bob body, a Circle shape on the bob with `CT_SWINGING`, and a `PinJoint` connecting them.
- [ ] The bob starts at `initial_angle_deg` from straight-down and swings under gravity.
- [ ] Player contact with the bob calls `player.die()`.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_swinging_hazard_anchor_is_static_bob_is_dynamic():
    from blueball.entities.swinging_hazard import SwingingHazard
    from blueball import collision as col
    w = World()
    sh = SwingingHazard(w, anchor_pos=(500, 400), rope_length=120, bob_mass=2.0, initial_angle_deg=45)
    assert sh.anchor_body.body_type == pymunk.Body.STATIC
    assert sh.bob_body.body_type == pymunk.Body.DYNAMIC
    assert sh.bob_shape.collision_type == col.CT_SWINGING


def test_swinging_hazard_kills_player_on_contact():
    from blueball.entities.swinging_hazard import SwingingHazard
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent
    from blueball.collision import register

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    register(w.space, world_ref=w)
    # Position bob right at the player so contact is immediate
    sh = SwingingHazard(w, anchor_pos=(200, 400), rope_length=60, bob_mass=2.0, initial_angle_deg=0)
    w.add_entity(sh)
    p = Player(agent=Idle(), spawn_xy=(200, 460))
    w.add_entity(p)
    for _ in range(10):
        w.step(1 / 60)
        if p.dead:
            break
    assert p.dead is True
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/swinging_hazard.py`:

```python
"""SwingingHazard — pendulum with a kinematic-static anchor and dynamic bob
joined by a pymunk PinJoint."""

from __future__ import annotations

import math

import pymunk

from .. import collision as _col
from .base import Entity


class SwingingHazard(Entity):
    def __init__(
        self,
        world,
        anchor_pos: tuple[float, float],
        rope_length: float = 128,
        bob_mass: float = 2.0,
        bob_radius: float = 14,
        initial_angle_deg: float = 30,
    ) -> None:
        super().__init__()
        self.anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_body.position = anchor_pos
        self.bodies.append(self.anchor_body)

        moment = pymunk.moment_for_circle(bob_mass, 0, bob_radius)
        self.bob_body = pymunk.Body(mass=bob_mass, moment=moment)
        theta = math.radians(initial_angle_deg)
        # Pymunk y-down: straight-down from anchor is +y
        bx = anchor_pos[0] + rope_length * math.sin(theta)
        by = anchor_pos[1] + rope_length * math.cos(theta)
        self.bob_body.position = (bx, by)
        self.bodies.append(self.bob_body)

        self.bob_shape = pymunk.Circle(self.bob_body, bob_radius)
        self.bob_shape.collision_type = _col.CT_SWINGING
        self.bob_shape.friction = 0.5
        self.shapes.append(self.bob_shape)

        self.joint = pymunk.PinJoint(self.anchor_body, self.bob_body, (0, 0), (0, 0))
        world.space.add(self.joint)

        self.rope_length = rope_length
        self.anchor_pos = anchor_pos
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/swinging_hazard.py`:

```python
"""swinging_hazard chunk — places a pendulum bob hanging from above."""

from __future__ import annotations

import pymunk

from ...entities.swinging_hazard import SwingingHazard
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("swinging_hazard")
class SwingingHazardChunk(Chunk):
    difficulty: int = 3

    def __init__(
        self,
        width_tiles: int = 4,
        anchor_y_offset: int = 192,
        rope_length: float = 128,
        bob_mass: float = 2.0,
        initial_angle_deg: float = 30,
    ) -> None:
        self.width_tiles = width_tiles
        self.anchor_y_offset = anchor_y_offset
        self.rope_length = rope_length
        self.bob_mass = bob_mass
        self.initial_angle_deg = initial_angle_deg

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(3, 5),
            "anchor_y_offset": rng.choice([160, 192, 224]),
            "rope_length": rng.choice([100, 128, 150]),
            "initial_angle_deg": rng.choice([20, 30, 40, -20, -30]),
        }

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        anchor = (x_offset + w / 2, GROUND_Y - self.anchor_y_offset)
        world.add_entity(SwingingHazard(
            world,
            anchor_pos=anchor,
            rope_length=self.rope_length,
            bob_mass=self.bob_mass,
            initial_angle_deg=self.initial_angle_deg,
        ))
        return w
```

Add `swinging_hazard,` to `__init__.py`.

- [ ] **Step 4: Wire collision handler**

In `collision.py` inside `register(...)`:

```python
    def on_swinging(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        if player is not None:
            player.die()
        return True

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_SWINGING, begin=on_swinging)
```

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/swinging_hazard.py src/blueball/levels/chunks/swinging_hazard.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: swinging_hazard pendulum kills player on bob contact"
```

---

## Task 20: Charger entity + chunk + handler

**Goal:** Kinematic enemy that patrols a bounded segment. When the player enters its FOV cone AND line-of-sight is unblocked (one segment query to static segments), it charges at the player. Top-stomp kills it; side contact kills the player.

**Files:**
- Create: `src/blueball/entities/charger.py`
- Create: `src/blueball/levels/chunks/charger_platform.py`
- Modify: `src/blueball/collision.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_entities.py`, `tests/test_collision.py`, `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `Charger(world, position, left_bound, right_bound, facing, sight_range, sight_arc_deg, charge_speed, patrol_speed)` exposes a kinematic body, a Circle shape with `CT_CHARGER`, and a state machine (`"patrol" | "charge"`).
- [ ] Patrol: walks at `patrol_speed`, reversing direction at bounds and on hitting a wall.
- [ ] FOV check: each `update(dt)` looks for the Player in the world; if the dx/dy vector is within `sight_arc_deg/2` of `facing` and within `sight_range`, and a `segment_query_first` from charger to player (filtered to static segments only) doesn't hit anything → state becomes "charge".
- [ ] Charge: velocity is `charge_speed` in the player's direction; reverts to patrol when LOS lost or bound reached.
- [ ] Collision handler: top-stomp (n.y ≥ TOL on player-side) kills the charger via `entity.die()`; otherwise kills the player.
- [ ] Chunk: `charger_platform`, difficulty 3.

**Verify:** `pytest -q tests/test_entities.py tests/test_collision.py tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_charger_patrols_when_player_absent():
    from blueball.entities.charger import Charger
    w = World()
    c = Charger(w, position=(300, 588), left_bound=200, right_bound=400, facing="right", sight_range=200, sight_arc_deg=60, charge_speed=180, patrol_speed=40)
    w.add_entity(c)
    start_x = c.body.position.x
    for _ in range(60):
        w.step(1 / 60)
    assert c.body.position.x != start_x
    assert c.state == "patrol"


def test_charger_switches_to_charge_when_player_in_cone():
    from blueball.entities.charger import Charger
    from blueball.entities.player import Player
    from blueball.agent import Action, Agent

    class Idle(Agent):
        def act(self, obs):
            return Action.IDLE

    w = World()
    c = Charger(w, position=(300, 588), left_bound=200, right_bound=600, facing="right", sight_range=300, sight_arc_deg=90, charge_speed=180, patrol_speed=40)
    w.add_entity(c)
    p = Player(agent=Idle(), spawn_xy=(400, 588))  # in the cone, to the right
    w.add_entity(p)
    for _ in range(10):
        w.step(1 / 60)
    assert c.state == "charge"
```

- [ ] **Step 2: Create entity**

`src/blueball/entities/charger.py`:

```python
"""Charger — kinematic enemy with directional FOV that charges at the player
when seen, otherwise patrols a bounded segment.
"""

from __future__ import annotations

import math

import pymunk

from .. import collision as _col
from .base import Entity


def _find_player(world):
    from .player import Player
    for e in world.entities:
        if isinstance(e, Player):
            return e
    return None


class Charger(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        left_bound: float,
        right_bound: float,
        facing: str = "right",
        sight_range: float = 200.0,
        sight_arc_deg: float = 60.0,
        charge_speed: float = 180.0,
        patrol_speed: float = 40.0,
        radius: int = 12,
    ) -> None:
        super().__init__()
        if facing not in ("left", "right"):
            raise ValueError(f"facing must be 'left' or 'right'; got {facing!r}")
        self.facing = facing
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.sight_range = sight_range
        self.sight_arc_cos = math.cos(math.radians(sight_arc_deg / 2))
        self.charge_speed = charge_speed
        self.patrol_speed = patrol_speed
        self.state = "patrol"
        self.alive = True

        moment = pymunk.moment_for_circle(1.0, 0, radius)
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        self.body = body
        self.bodies.append(body)
        self.shape = pymunk.Circle(body, radius)
        self.shape.collision_type = _col.CT_CHARGER
        self.shape.friction = 0.5
        self.shapes.append(self.shape)

        body.velocity = (patrol_speed if facing == "right" else -patrol_speed, 0)
        self._world_ref = world

    def die(self) -> None:
        self.alive = False
        if self.shape in self._world_ref.space.shapes:
            self._world_ref.space.remove(self.shape)
        if self.body in self._world_ref.space.bodies:
            self._world_ref.space.remove(self.body)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        player = _find_player(self._world_ref)
        if player is None:
            return self._patrol_tick()
        # FOV check
        dx = player.body.position.x - self.body.position.x
        dy = player.body.position.y - self.body.position.y
        dist = math.hypot(dx, dy)
        in_range = dist <= self.sight_range
        if in_range and dist > 0:
            facing_dir = 1.0 if self.facing == "right" else -1.0
            cos_to_player = (dx * facing_dir + dy * 0) / dist
            in_cone = cos_to_player >= self.sight_arc_cos
        else:
            in_cone = False
        # LOS: segment query from charger to player; if anything static is hit, LOS blocked
        los_clear = True
        if in_range and in_cone:
            hit = self._world_ref.space.segment_query_first(
                (self.body.position.x, self.body.position.y),
                (player.body.position.x, player.body.position.y),
                0.5,
                pymunk.ShapeFilter(),
            )
            if hit is not None and hit.shape.body.body_type == pymunk.Body.STATIC:
                los_clear = False
        if in_range and in_cone and los_clear:
            self.state = "charge"
            dir_x = 1.0 if dx > 0 else -1.0
            self.body.velocity = (self.charge_speed * dir_x, 0)
        else:
            self.state = "patrol"
            self._patrol_tick()
        # Always respect bounds
        if self.body.position.x <= self.left_bound:
            self.body.velocity = (abs(self.body.velocity.x) or self.patrol_speed, 0)
        elif self.body.position.x >= self.right_bound:
            self.body.velocity = (-(abs(self.body.velocity.x) or self.patrol_speed), 0)

    def _patrol_tick(self) -> None:
        # Maintain patrol speed magnitude in the current direction
        vx = self.body.velocity.x
        if vx >= 0:
            self.body.velocity = (self.patrol_speed, 0)
        else:
            self.body.velocity = (-self.patrol_speed, 0)
```

- [ ] **Step 3: Create chunk**

`src/blueball/levels/chunks/charger_platform.py`:

```python
"""charger_platform chunk — flat ground with a Charger patrolling it."""

from __future__ import annotations

import pymunk

from ... import config
from ...entities.charger import Charger
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("charger_platform")
class ChargerPlatformChunk(Chunk):
    difficulty: int = 3

    def __init__(
        self,
        length_tiles: int = 8,
        facing: str = "right",
        sight_range: float = config.CHARGER_DEFAULT_SIGHT_RANGE,
        sight_arc_deg: float = config.CHARGER_DEFAULT_SIGHT_ARC_DEG,
        charge_speed: float = config.CHARGER_DEFAULT_CHARGE_SPEED,
        patrol_speed: float = config.CHARGER_DEFAULT_PATROL_SPEED,
    ) -> None:
        self.length_tiles = length_tiles
        self.facing = facing
        self.sight_range = sight_range
        self.sight_arc_deg = sight_arc_deg
        self.charge_speed = charge_speed
        self.patrol_speed = patrol_speed

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "length_tiles": rng.randint(6, 10),
            "facing": rng.choice(["left", "right"]),
            "sight_range": rng.choice([160, 200, 240]),
            "charge_speed": rng.choice([140.0, 180.0, 220.0]),
        }

    def build(self, world, x_offset: float) -> float:
        w = self.length_tiles * TILE
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        left = x_offset + 16
        right = x_offset + w - 16
        world.add_entity(Charger(
            world,
            position=((left + right) / 2, GROUND_Y - 12),
            left_bound=left,
            right_bound=right,
            facing=self.facing,
            sight_range=self.sight_range,
            sight_arc_deg=self.sight_arc_deg,
            charge_speed=self.charge_speed,
            patrol_speed=self.patrol_speed,
        ))
        return w
```

Add `charger_platform,` to `__init__.py`.

- [ ] **Step 4: Wire collision handler**

In `collision.py` inside `register(...)`:

```python
    def on_charger(arbiter, space_, data):
        player = _find_player_entity(arbiter, world_ref)
        if player is None:
            return True
        n = arbiter.contact_point_set.normal
        for shape in arbiter.shapes:
            entity = _find_entity_for_shape(shape, world_ref)
            if entity is None or entity is player:
                continue
            if shape.collision_type != CT_CHARGER:
                continue
            if arbiter.shapes[0] is shape:
                if -n.y >= _TOP_NORMAL_COS:
                    entity.die()
                    return True
            else:
                if n.y >= _TOP_NORMAL_COS:
                    entity.die()
                    return True
            player.die()
            return True
        return True

    space.on_collision(collision_type_a=CT_PLAYER, collision_type_b=CT_CHARGER, begin=on_charger)
```

- [ ] **Step 5: Run, confirm pass**

- [ ] **Step 6: Commit**

```bash
git add src/blueball/entities/charger.py src/blueball/levels/chunks/charger_platform.py src/blueball/collision.py src/blueball/levels/chunks/__init__.py tests/
git commit -m "feat: charger enemy with directional FOV and LOS-gated charging"
```

---

## Task 21: `spike_wall` chunk

**Goal:** Variant of `spike_pit` that places spikes against a wall (ceiling, side) using the oriented `Spike` entity from Task 6.

**Files:**
- Create: `src/blueball/levels/chunks/spike_wall.py`
- Modify: `src/blueball/levels/chunks/__init__.py`
- Modify: `tests/test_chunks.py`

**Acceptance Criteria:**
- [ ] `spike_wall` registered; difficulty 2.
- [ ] `SpikeWall(width_tiles=3, spikes=3, orientation="down")` places N `Spike(orientation="down")` entities along the ceiling of the chunk slot (y = GROUND_Y - some configurable height).

**Verify:** `pytest -q tests/test_chunks.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_spike_wall_chunk_places_oriented_spikes():
    from blueball.levels.chunks.spike_wall import SpikeWall
    from blueball.entities.spike import Spike
    w = World()
    SpikeWall(width_tiles=4, spikes=4, orientation="down", ceiling_y_offset=160).build(w, x_offset=0)
    spikes = [e for e in w.entities if isinstance(e, Spike)]
    assert len(spikes) == 4
    for s in spikes:
        assert s.orientation == "down"
```

- [ ] **Step 2: Create chunk**

`src/blueball/levels/chunks/spike_wall.py`:

```python
"""spike_wall chunk — N oriented spikes along a wall (ceiling or side)."""

from __future__ import annotations

import pymunk

from ...entities.spike import Spike
from .base import Chunk, TILE, register_chunk
from .flat import GROUND_Y


@register_chunk("spike_wall")
class SpikeWall(Chunk):
    difficulty: int = 2

    def __init__(self, width_tiles: int = 3, spikes: int = 3, orientation: str = "down", ceiling_y_offset: int = 160) -> None:
        self.width_tiles = width_tiles
        self.spikes = spikes
        self.orientation = orientation
        self.ceiling_y_offset = ceiling_y_offset

    @classmethod
    def random_params(cls, rng) -> dict:
        return {
            "width_tiles": rng.randint(2, 4),
            "spikes": rng.randint(2, 4),
            "orientation": rng.choice(["down", "left", "right"]),
            "ceiling_y_offset": rng.choice([128, 160, 200]),
        }

    def build(self, world, x_offset: float) -> float:
        w = self.width_tiles * TILE
        # For "down" we place along a ceiling; for "left"/"right" we put them
        # along the side at midheight (just a row of spikes the player must
        # avoid). For "up" we behave like spike_pit (rare for spike_wall but
        # supported for completeness).
        if self.orientation == "up":
            y = GROUND_Y
        else:
            y = GROUND_Y - self.ceiling_y_offset
        seg = pymunk.Segment(world.space.static_body, (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        for i in range(self.spikes):
            cx = x_offset + (i + 0.5) * w / self.spikes
            world.add_entity(Spike(world, position=(cx, y), width=24, height=24, orientation=self.orientation))
        return w
```

Add `spike_wall,` to `__init__.py`.

- [ ] **Step 3: Run tests, confirm pass**

- [ ] **Step 4: Commit**

```bash
git add src/blueball/levels/chunks/spike_wall.py src/blueball/levels/chunks/__init__.py tests/test_chunks.py
git commit -m "feat: spike_wall chunk uses oriented Spike entities"
```

---

## Task 22: HitType enum + enriched Observation dataclass

**Goal:** Replace the existing `Observation` dataclass with the enriched shape. Add the `HitType` IntEnum and `_CT_TO_HITTYPE` lookup. `Player._observe` still returns the original shape (Task 23 rewrites it); this task only changes the dataclass.

**Files:**
- Modify: `src/blueball/agent.py`
- Modify: `src/blueball/entities/player.py` (update `_observe` to return new dataclass with placeholder values)
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] `HitType` IntEnum exported with values MISS=0, GROUND=1, HAZARD=2, PICKUP=3, GOAL=4, ENEMY=5, BLOCK=6, DOOR=7.
- [ ] `_CT_TO_HITTYPE: dict[int, HitType]` maps every collision-type constant to its category (per the spec).
- [ ] `Observation` dataclass has the new fields: `rays`, `ray_hit_types`, `vel`, `ang_vel`, `grounded`, `nearest_pickup`, `nearest_hazard`, `abilities`, `keys_held`.
- [ ] `Player._observe()` returns the new shape with zero arrays / None for not-yet-implemented fields (Task 23 fills them in).

**Verify:** `pytest -q tests/test_player.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_observation_has_enriched_fields():
    from blueball.agent import Observation, HitType
    import numpy as np
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    obs = p._observe()
    assert obs.rays.shape == (8,)
    assert obs.ray_hit_types.shape == (8,)
    assert obs.ray_hit_types.dtype == np.int8
    assert obs.abilities == 0
    assert obs.keys_held == 0
    assert obs.nearest_hazard is None
    # HitType enum complete
    assert HitType.MISS == 0
    assert HitType.DOOR == 7
```

- [ ] **Step 2: Rewrite `agent.py`**

Replace the file's `Observation` block:

```python
"""Agent interface. v1 ships HumanAgent; AI agents arrive later."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pygame

from . import collision as _col


class Action(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


class HitType(IntEnum):
    MISS = 0
    GROUND = 1
    HAZARD = 2
    PICKUP = 3
    GOAL = 4
    ENEMY = 5
    BLOCK = 6
    DOOR = 7


_CT_TO_HITTYPE: dict[int, HitType] = {
    _col.CT_PLAYER: HitType.GROUND,  # never hit, but safe default
    _col.CT_SPIKE: HitType.HAZARD,
    _col.CT_PATROLLER: HitType.ENEMY,
    _col.CT_COLLECTIBLE: HitType.PICKUP,
    _col.CT_GOAL: HitType.GOAL,
    _col.CT_BOOST_PAD: HitType.PICKUP,
    _col.CT_ABILITY_PICKUP: HitType.PICKUP,
    _col.CT_ONE_WAY: HitType.GROUND,
    _col.CT_SPRING: HitType.PICKUP,
    _col.CT_PUSHABLE: HitType.BLOCK,
    _col.CT_SWINGING: HitType.HAZARD,
    _col.CT_CHARGER: HitType.ENEMY,
    _col.CT_CHECKPOINT: HitType.PICKUP,
    _col.CT_KEY: HitType.PICKUP,
    _col.CT_DOOR: HitType.DOOR,
}


@dataclass(frozen=True)
class Observation:
    rays: np.ndarray              # shape (8,), float32, in [0, 1]; 1.0 = miss
    ray_hit_types: np.ndarray     # shape (8,), int8 HitType values
    vel: np.ndarray               # shape (2,), float32
    ang_vel: float
    grounded: bool
    nearest_pickup: Optional[tuple[float, float]]
    nearest_hazard: Optional[tuple[float, float]]
    abilities: int                # bitfield, ability enum ordinal
    keys_held: int                # bitfield


class Agent(abc.ABC):
    def reset(self, world) -> None:
        """Called at level start. Default no-op."""

    @abc.abstractmethod
    def act(self, observation: Observation) -> Action:
        ...


class HumanAgent(Agent):
    """Reads PyGame keyboard state and emits an Action."""

    def act(self, observation: Observation) -> Action:
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]

        if left and not right:
            return Action.LEFT_JUMP if jump else Action.LEFT
        if right and not left:
            return Action.RIGHT_JUMP if jump else Action.RIGHT
        if jump:
            return Action.JUMP
        return Action.IDLE
```

- [ ] **Step 3: Stub-update Player._observe**

In `entities/player.py`, replace `_observe`:

```python
    def _observe(self) -> Observation:
        return Observation(
            rays=np.ones(8, dtype=np.float32),
            ray_hit_types=np.zeros(8, dtype=np.int8),
            vel=np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32),
            ang_vel=self.body.angular_velocity,
            grounded=self.grounded,
            nearest_pickup=None,
            nearest_hazard=None,
            abilities=0,
            keys_held=self.keys_held,
        )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest -q tests/ -v` and verify no other tests broke (they shouldn't, since the old fields are gone and only test files that asserted on them would break — currently only test_player constructs Observation indirectly via `_observe`).

- [ ] **Step 5: Commit**

```bash
git add src/blueball/agent.py src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: HitType enum + enriched Observation dataclass (stub _observe)"
```

---

## Task 23: Player._observe — raycasts + scalars + bitfields

**Goal:** Populate the enriched Observation from real raycasts, nearest-entity scans, and an abilities bitfield.

**Files:**
- Modify: `src/blueball/entities/player.py`
- Modify: `src/blueball/config.py` (constants set in Task 9; verify present)
- Modify: `tests/test_player.py`

**Acceptance Criteria:**
- [ ] 8 raycasts evenly spaced 45° apart, starting from due-right counter-clockwise. Each ray uses `space.segment_query_first` with the player's `_ray_filter`. Distances normalized to `MAX_RAY_LEN` ∈ [0, 1]; 1.0 = miss.
- [ ] `ray_hit_types[i]` is the `HitType` derived from the hit shape's `collision_type` via `_CT_TO_HITTYPE`. Defaults to `HitType.GROUND` for unknown CTs.
- [ ] `nearest_pickup` and `nearest_hazard` returned as world-frame deltas `(target.x - player.x, target.y - player.y)`, or `None` if no entity in the bucket exists. Buckets per spec:
  - Pickups: `Collectible`, `AbilityPickup`, `BoostPad`, `Key`, `Spring`, `Checkpoint`.
  - Hazards: `Spike`, `FallingHazard`, `Patroller`, `SwingingHazard`, `Charger`.
- [ ] `abilities` bitfield indexed by `Ability` enum declaration order.

**Verify:** `pytest -q tests/test_player.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

```python
def test_observe_rays_hit_nearby_geometry():
    import pymunk
    w = World()
    # Place a wall 100 px to the right
    wall = pymunk.Segment(w.space.static_body, (200, 0), (200, 1000), 5)
    w.space.add(wall)
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    obs = p._observe()
    # Ray 0 (due-right) should hit the wall (alpha = 100/MAX_RAY_LEN)
    from blueball import config
    expected_alpha = 100 / config.MAX_RAY_LEN
    assert abs(obs.rays[0] - expected_alpha) < 0.05
    from blueball.agent import HitType
    assert obs.ray_hit_types[0] == HitType.GROUND  # static segment with default CT


def test_observe_nearest_pickup_finds_collectible():
    from blueball.entities.collectible import Collectible
    w = World()
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100))
    w.add_entity(p)
    c = Collectible(w, position=(200, 150))
    w.add_entity(c)
    obs = p._observe()
    assert obs.nearest_pickup == (100.0, 50.0)


def test_observe_abilities_bitfield():
    from blueball.abilities import Ability
    p = Player(agent=_ScriptedAgent([Action.IDLE]), spawn_xy=(100, 100), abilities={Ability.DOUBLE_JUMP})
    obs = p._observe()
    assert obs.abilities & 1  # bit 0 = DOUBLE_JUMP (declaration order)
```

- [ ] **Step 2: Implement raycast and scans**

In `entities/player.py`, add module constants near the top:

```python
import math as _math

_RAY_ANGLES = tuple((_math.cos(i * 2 * _math.pi / 8), _math.sin(i * 2 * _math.pi / 8)) for i in range(8))


def _abilities_to_bitfield(abilities) -> int:
    from ..abilities import Ability
    bits = 0
    for i, member in enumerate(Ability):
        if member in abilities:
            bits |= (1 << i)
    return bits


_PICKUP_TYPENAMES = {"Collectible", "AbilityPickup", "BoostPad", "Key", "Spring", "Checkpoint"}
_HAZARD_TYPENAMES = {"Spike", "FallingHazard", "Patroller", "SwingingHazard", "Charger"}
```

(The `import math as _math` line goes alongside the existing `import math`; if `math` is already imported, just reuse it.)

Replace `_observe()` with:

```python
    def _observe(self) -> Observation:
        from ..agent import HitType, _CT_TO_HITTYPE
        from .. import config

        pos = self.body.position
        rays = np.empty(8, dtype=np.float32)
        hit_types = np.empty(8, dtype=np.int8)
        for i, (cx, sx) in enumerate(_RAY_ANGLES):
            end = (pos.x + cx * config.MAX_RAY_LEN, pos.y + sx * config.MAX_RAY_LEN)
            hit = self._world.space.segment_query_first(
                (pos.x, pos.y), end, 0.5, self._ray_filter,
            )
            if hit is None:
                rays[i] = 1.0
                hit_types[i] = HitType.MISS
            else:
                rays[i] = float(hit.alpha)
                hit_types[i] = _CT_TO_HITTYPE.get(hit.shape.collision_type, HitType.GROUND)

        nearest_pickup = self._nearest_entity_delta(_PICKUP_TYPENAMES)
        nearest_hazard = self._nearest_entity_delta(_HAZARD_TYPENAMES)

        return Observation(
            rays=rays,
            ray_hit_types=hit_types,
            vel=np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32),
            ang_vel=self.body.angular_velocity,
            grounded=self.grounded,
            nearest_pickup=nearest_pickup,
            nearest_hazard=nearest_hazard,
            abilities=_abilities_to_bitfield(self.abilities),
            keys_held=self.keys_held,
        )

    def _nearest_entity_delta(self, type_names: set[str]):
        if self._world is None:
            return None
        best = None
        best_d2 = float("inf")
        px, py = self.body.position
        for e in self._world.entities:
            if type(e).__name__ not in type_names:
                continue
            ex, ey = getattr(e, "body", None).position if getattr(e, "body", None) else getattr(e, "position", (None, None))
            if ex is None:
                continue
            dx = ex - px
            dy = ey - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (dx, dy)
        return best
```

- [ ] **Step 3: Run tests, confirm pass**

Run: `pytest -q tests/ -v` — verify all pass.

- [ ] **Step 4: Commit**

```bash
git add src/blueball/entities/player.py tests/test_player.py
git commit -m "feat: Player._observe — real raycasts, scalars, bitfields"
```

---

## Task 24: Loader dict-mode

**Goal:** `load_level(source, world)` accepts a `Path | str | dict` and dispatches.

**Files:**
- Modify: `src/blueball/levels/loader.py`
- Modify: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] `load_level(path_str, world)` and `load_level(Path(...), world)` continue to work.
- [ ] `load_level({"name": "X", "background": "#000000", "ground": "#111111", "spawn": [0,0], "chunks": [...]}, world)` builds directly without file I/O.

**Verify:** `pytest -q tests/test_level_loader.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

```python
def test_load_level_accepts_dict():
    from blueball.levels.loader import load_level
    from blueball.world import World
    data = {
        "name": "Test",
        "background": "#000000",
        "ground": "#111111",
        "spawn": [80, 540],
        "chunks": [
            {"type": "flat", "width_tiles": 3},
            {"type": "goal"},
        ],
    }
    w = World()
    meta = load_level(data, w)
    assert meta.name == "Test"
    assert meta.total_width > 0
```

- [ ] **Step 2: Modify loader**

In `src/blueball/levels/loader.py`:

```python
def load_level(source: Path | str | dict, world) -> LevelMeta:
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(Path(source).read_text())
    chunks_list = data["chunks"]

    x = 0.0
    for entry in chunks_list:
        type_name = entry["type"]
        kwargs = {k: v for k, v in entry.items() if k != "type"}
        if type_name not in CHUNK_REGISTRY:
            available = ", ".join(sorted(CHUNK_REGISTRY))
            raise ValueError(f"Unknown chunk type {type_name!r}. Available: {available}")
        chunk = CHUNK_REGISTRY[type_name](**kwargs)
        width = chunk.build(world, x_offset=x)
        x += width

    spawn = tuple(data["spawn"])
    return LevelMeta(
        name=data["name"],
        spawn=spawn,
        background=_hex_to_rgb(data["background"]),
        ground=_hex_to_rgb(data["ground"]),
        total_width=x,
    )
```

(Make sure `from typing import Union` is imported if needed for older Python; the `|` syntax requires 3.10+, which the project already uses.)

- [ ] **Step 3: Run tests, confirm pass**

- [ ] **Step 4: Commit**

```bash
git add src/blueball/levels/loader.py tests/test_level_loader.py
git commit -m "feat: load_level accepts dict for in-memory sampler output"
```

---

## Task 25: ChunkSampler

**Goal:** Deterministic sampler emitting a chunk-dict sequence with a soft difficulty ramp, periodic checkpoints, and a goal terminator.

**Files:**
- Create: `src/blueball/levels/sampler.py`
- Create: `tests/test_sampler.py`

**Acceptance Criteria:**
- [ ] `ChunkSampler(seed, target_chunks=500, ramp_per_chunk=0.006, sigma=0.7, checkpoint_every=25)` is iterable; iterating to exhaustion yields exactly `target_chunks` non-terminal entries plus a final `{"type": "goal"}`.
- [ ] Two `ChunkSampler(seed=N)` instances yield identical sequences.
- [ ] Every emitted chunk except checkpoints and the goal has `sampler_include == True`.
- [ ] Average difficulty of the last quartile exceeds the first quartile (smoke check of the ramp).
- [ ] `checkpoint` entries appear at progress indices `25, 50, 75, ...`.

**Verify:** `pytest -q tests/test_sampler.py -v`

**Steps:**

- [ ] **Step 1: Failing test**

Create `tests/test_sampler.py`:

```python
import pytest

from blueball.levels.chunks.base import CHUNK_REGISTRY
# Importing the chunks package registers every chunk type
from blueball.levels import chunks  # noqa: F401
from blueball.levels.sampler import ChunkSampler


def test_sampler_is_deterministic_per_seed():
    a = list(ChunkSampler(seed=42, target_chunks=50))
    b = list(ChunkSampler(seed=42, target_chunks=50))
    assert a == b


def test_sampler_ends_with_goal():
    seq = list(ChunkSampler(seed=1, target_chunks=20))
    assert seq[-1] == {"type": "goal"}
    assert sum(1 for s in seq if s["type"] == "goal") == 1


def test_sampler_emits_only_sampler_included_chunks():
    seq = list(ChunkSampler(seed=1, target_chunks=50))
    for entry in seq:
        t = entry["type"]
        if t in ("goal", "checkpoint"):
            continue
        assert CHUNK_REGISTRY[t].sampler_include is True


def test_sampler_difficulty_ramps_with_progress():
    seq = [s for s in ChunkSampler(seed=7, target_chunks=200) if s["type"] not in ("goal", "checkpoint")]
    q = len(seq) // 4
    first_q = seq[:q]
    last_q = seq[-q:]
    avg_first = sum(CHUNK_REGISTRY[s["type"]].difficulty for s in first_q) / len(first_q)
    avg_last = sum(CHUNK_REGISTRY[s["type"]].difficulty for s in last_q) / len(last_q)
    assert avg_last > avg_first


def test_sampler_inserts_checkpoints_every_n_steps():
    seq = list(ChunkSampler(seed=2, target_chunks=100, checkpoint_every=20))
    # Find indices of checkpoints
    checkpoint_indices = [i for i, s in enumerate(seq) if s["type"] == "checkpoint"]
    # First checkpoint should be at index ~20 (after 20 emits)
    assert len(checkpoint_indices) >= 4
```

- [ ] **Step 2: Implement sampler**

Create `src/blueball/levels/sampler.py`:

```python
"""ChunkSampler — deterministic procedural chunk emitter."""

from __future__ import annotations

import math
import random
from typing import Iterator

# Importing the chunks package registers every chunk type
from . import chunks  # noqa: F401
from .chunks.base import CHUNK_REGISTRY, Chunk


class ChunkSampler:
    def __init__(
        self,
        seed: int,
        target_chunks: int = 500,
        ramp_per_chunk: float = 0.006,
        sigma: float = 0.7,
        checkpoint_every: int = 25,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.target = target_chunks
        self.ramp = ramp_per_chunk
        self.sigma = sigma
        self.checkpoint_every = checkpoint_every
        self.progress = 0
        # Stable-sorted pool of sampler-included chunks. Sorting by name removes
        # dict-ordering as a determinism risk.
        self._pool: list[tuple[str, type[Chunk]]] = sorted(
            ((name, cls) for name, cls in CHUNK_REGISTRY.items() if cls.sampler_include),
            key=lambda item: item[0],
        )

    def __iter__(self) -> Iterator[dict]:
        while True:
            entry = self.emit_next()
            if entry is None:
                return
            yield entry

    def emit_next(self) -> dict | None:
        if self.progress > self.target:
            return None
        if self.progress == self.target:
            self.progress += 1  # advance past so subsequent calls return None
            return {"type": "goal"}
        # Checkpoint every N (but not at index 0)
        if self.progress > 0 and self.progress % self.checkpoint_every == 0:
            cid = self.progress // self.checkpoint_every
            self.progress += 1
            return {"type": "checkpoint", "id": cid}
        # Weighted pick by closeness to target difficulty
        target_diff = min(3.0, self.progress * self.ramp)
        names = [n for n, _ in self._pool]
        weights = [
            math.exp(-((cls.difficulty - target_diff) ** 2) / (2 * self.sigma ** 2))
            for _, cls in self._pool
        ]
        idx = self._weighted_pick(weights)
        name, cls = self._pool[idx]
        params = cls.random_params(self.rng)
        self.progress += 1
        return {"type": name, **params}

    def _weighted_pick(self, weights: list[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1
```

- [ ] **Step 3: Run tests, confirm pass**

- [ ] **Step 4: Commit**

```bash
git add src/blueball/levels/sampler.py tests/test_sampler.py
git commit -m "feat: ChunkSampler — deterministic procedural chunk emitter"
```

---

## Task 26: MenuScene + main.py entry

**Goal:** New `MenuScene` with 5 entries; `main.py` starts in MenuScene; selecting an entry returns a `PlayScene`.

**Files:**
- Create: `src/blueball/scenes/menu.py`
- Modify: `main.py`
- Create: `tests/test_menu_scene.py`

**Acceptance Criteria:**
- [ ] `MenuScene(screen)` lists 5 entries (Tutorial Hill, Vertical Climb, Speed Run, Maze, Infinite Run).
- [ ] Up/Down (and W/S) move the cursor; bounded.
- [ ] Enter on a normal level returns a `PlayScene` with that `level_path`.
- [ ] Enter on Infinite Run returns a `PlayScene` with non-None `level_data` whose `chunks` ends with a `{"type": "goal"}` entry; `sampler_seed` is set on the returned scene.
- [ ] Esc returns `None` (app quit).
- [ ] `main.py` starts in `MenuScene(screen)` instead of constructing `PlayScene` directly.

**Verify:** `pytest -q tests/test_menu_scene.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

Create `tests/test_menu_scene.py`:

```python
import pygame
import pytest

from blueball.scenes.menu import MenuScene
from blueball.scenes.play import PlayScene


@pytest.fixture(autouse=True)
def _init_pygame():
    pygame.init()
    pygame.display.set_mode((800, 600))
    yield
    pygame.quit()


def _key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, {"key": key})


def test_menu_cursor_moves_down():
    m = MenuScene(pygame.display.get_surface())
    assert m.cursor == 0
    m.handle_events([_key_event(pygame.K_DOWN)])
    assert m.cursor == 1
    # Wrap-clamp at end
    for _ in range(20):
        m.handle_events([_key_event(pygame.K_DOWN)])
    assert m.cursor == len(m.entries) - 1


def test_menu_enter_on_normal_level_returns_playscene():
    m = MenuScene(pygame.display.get_surface())
    m.cursor = 0  # Tutorial Hill
    result = m.handle_events([_key_event(pygame.K_RETURN)])
    assert isinstance(result, PlayScene)
    assert result.level_path is not None


def test_menu_enter_on_infinite_run_returns_playscene_with_level_data():
    m = MenuScene(pygame.display.get_surface())
    # Infinite Run is the last entry
    m.cursor = len(m.entries) - 1
    result = m.handle_events([_key_event(pygame.K_RETURN)])
    assert isinstance(result, PlayScene)
    assert result.level_data is not None
    assert result.level_data["chunks"][-1]["type"] == "goal"
    assert result.sampler_seed is not None


def test_menu_esc_returns_none():
    m = MenuScene(pygame.display.get_surface())
    result = m.handle_events([_key_event(pygame.K_ESCAPE)])
    assert result is None
```

- [ ] **Step 2: Create MenuScene**

`src/blueball/scenes/menu.py`:

```python
"""MenuScene — level select with 5 entries."""

from __future__ import annotations

import time
from pathlib import Path

import pygame

from .. import config
from ..levels.sampler import ChunkSampler
from .base import Scene
from .play import PlayScene


class MenuScene(Scene):
    INFINITE_RUN = "__infinite__"

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        levels_dir = Path(__file__).parent.parent / "levels"
        self.entries: list[tuple[str, object]] = [
            ("Tutorial Hill", levels_dir / "tutorial_hill.json"),
            ("Vertical Climb", levels_dir / "vertical_climb.json"),
            ("Speed Run", levels_dir / "speed_run.json"),
            ("Maze", levels_dir / "maze.json"),
            ("Infinite Run", self.INFINITE_RUN),
        ]
        self.cursor: int = 0
        self._font = pygame.font.SysFont(None, 36)
        self._title_font = pygame.font.SysFont(None, 64)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.cursor = max(0, self.cursor - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.cursor = min(len(self.entries) - 1, self.cursor + 1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    _, target = self.entries[self.cursor]
                    if target == self.INFINITE_RUN:
                        seed = int(time.time() * 1000) & 0xFFFFFFFF
                        sampler = ChunkSampler(seed=seed)
                        level_data = {
                            "name": f"Infinite Run (seed={seed})",
                            "background": "#202028",
                            "ground": "#666c70",
                            "spawn": [80, 540],
                            "chunks": list(sampler),
                        }
                        return PlayScene(self.screen, level_data=level_data, sampler_seed=seed)
                    return PlayScene(self.screen, level_path=target)
        return self

    def update(self, frame_dt: float) -> None:
        pass

    def draw(self) -> None:
        self.screen.fill((20, 30, 50))
        title_surf = self._title_font.render("Blue Ball", True, (255, 255, 255))
        self.screen.blit(title_surf, ((self.screen.get_width() - title_surf.get_width()) // 2, 80))
        for i, (label, _) in enumerate(self.entries):
            color = (255, 220, 80) if i == self.cursor else (200, 200, 200)
            prefix = "> " if i == self.cursor else "  "
            surf = self._font.render(prefix + label, True, color)
            self.screen.blit(surf, (self.screen.get_width() // 2 - 100, 220 + i * 50))
        hint = self._font.render("Up/Down: select   Enter: play   Esc: quit", True, (140, 140, 160))
        self.screen.blit(hint, ((self.screen.get_width() - hint.get_width()) // 2, self.screen.get_height() - 60))
        pygame.display.flip()
```

- [ ] **Step 3: Update main.py**

Replace `main.py`:

```python
import sys

import pygame

from blueball import config
from blueball.scenes.menu import MenuScene


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball")
    clock = pygame.time.Clock()

    scene = MenuScene(screen)

    while scene is not None:
        events = pygame.event.get()
        scene = scene.handle_events(events)
        if scene is None:
            break
        frame_dt = clock.tick(config.TARGET_FPS) / 1000.0
        scene.update(frame_dt)
        scene.draw()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/blueball/scenes/menu.py main.py tests/test_menu_scene.py
git commit -m "feat: MenuScene with 5 entries + Infinite Run wiring"
```

---

## Task 27: PlayScene updates — level_data, Esc-to-menu, checkpoint respawn

**Goal:** `PlayScene` accepts either `level_path` or `level_data`; returns to `MenuScene` on Esc and on level-complete; restores from `respawn_xy` on death.

**Files:**
- Modify: `src/blueball/scenes/play.py`
- Modify: `tests/test_play_scene.py`

**Acceptance Criteria:**
- [ ] Constructor signature: `__init__(self, screen, level_path=None, level_data=None, sampler_seed=None)`. Exactly one of `level_path`/`level_data` must be non-None.
- [ ] `_reset()` passes whichever source through to `load_level`; if `self._last_respawn_xy` is set, overrides `player.body.position` with it.
- [ ] On `player.dead`, snapshots `self._last_respawn_xy = self.player.respawn_xy` (may be None) before `_reset()`.
- [ ] On `world.level_complete`, sets `self._exit_to_menu = True`, clears `self._last_respawn_xy = None`; `handle_events` then returns `MenuScene(self.screen)`.
- [ ] Esc key in `handle_events` returns `MenuScene(self.screen)` (not None). `pygame.QUIT` still returns None.

**Verify:** `pytest -q tests/test_play_scene.py -v`

**Steps:**

- [ ] **Step 1: Failing tests**

Append to `tests/test_play_scene.py`:

```python
def test_play_scene_accepts_level_data(monkeypatch, tmp_path):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    data = {
        "name": "Test", "background": "#000000", "ground": "#111111",
        "spawn": [80, 540],
        "chunks": [{"type": "flat", "width_tiles": 3}, {"type": "goal"}],
    }
    from blueball.scenes.play import PlayScene
    scene = PlayScene(screen, level_data=data, sampler_seed=12345)
    assert scene.sampler_seed == 12345
    assert scene.level_data is data
    pygame.quit()


def test_play_scene_esc_returns_menu_scene(monkeypatch, tmp_path):
    import pygame
    from pathlib import Path
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(tmp_path / "save.json"))
    level_path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "tutorial_hill.json"
    from blueball.scenes.play import PlayScene
    from blueball.scenes.menu import MenuScene
    scene = PlayScene(screen, level_path=level_path)
    result = scene.handle_events([pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})])
    assert isinstance(result, MenuScene)
    pygame.quit()
```

- [ ] **Step 2: Update PlayScene**

Replace `src/blueball/scenes/play.py`:

```python
"""PlayScene — gameplay loop. Accepts either a level path or in-memory data."""

from __future__ import annotations

from pathlib import Path

import pygame

from .. import config, save
from ..abilities import Ability
from ..agent import HumanAgent
from ..camera import FollowCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


class PlayScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        level_path: Path | None = None,
        level_data: dict | None = None,
        sampler_seed: int | None = None,
    ) -> None:
        if (level_path is None) == (level_data is None):
            raise ValueError("PlayScene requires exactly one of level_path or level_data")
        self.screen = screen
        self.level_path = level_path
        self.level_data = level_data
        self.sampler_seed = sampler_seed
        self.camera = FollowCamera(screen.get_width(), screen.get_height())
        self.renderer = Renderer(screen, self.camera)
        self._last_respawn_xy: tuple[float, float] | None = None
        self._exit_to_menu: bool = False
        self._reset()

    def _reset(self) -> None:
        self.world = World()
        register_collisions(self.world.space, world_ref=self.world)
        source = self.level_path if self.level_path is not None else self.level_data
        self.level_meta = load_level(source, self.world)
        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=unlocked,
        )
        if self._last_respawn_xy is not None:
            self.player.body.position = self._last_respawn_xy
        self.world.add_entity(self.player)
        self.camera.position = (self.player.body.position.x, self.player.body.position.y)

    def handle_events(self, events):
        if self._exit_to_menu:
            from .menu import MenuScene
            return MenuScene(self.screen)
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                from .menu import MenuScene
                return MenuScene(self.screen)
        return self

    def update(self, frame_dt: float) -> None:
        self.renderer.begin_frame(self.world)
        self.world.step(frame_dt)
        if self.player.dead:
            self._last_respawn_xy = self.player.respawn_xy
            self._reset()
            return
        if self.world.level_complete:
            for ability in self.player.abilities:
                save.add_ability(ability.value)
            print(f"Level complete! Collectibles: {self.player.collectibles_collected}")
            self._last_respawn_xy = None
            self._exit_to_menu = True
            return
        self.camera.update(
            target=(self.player.body.position.x, self.player.body.position.y),
            dt=frame_dt,
        )

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        pygame.display.flip()
```

- [ ] **Step 3: Run tests, confirm pass**

- [ ] **Step 4: Commit**

```bash
git add src/blueball/scenes/play.py tests/test_play_scene.py
git commit -m "feat: PlayScene supports level_data, esc-to-menu, checkpoint respawn"
```

---

## Task 28: Renderer additions for new entities

**Goal:** Each new entity that needs visual output gets a draw method in `Renderer`. Each entity's `draw` method on the entity class calls into the renderer. Flat-primitive style, no asset pipeline.

**Files:**
- Modify: `src/blueball/render/renderer.py`
- Modify: each new entity module that didn't already implement `draw`

**Acceptance Criteria:**
- [ ] All new entities (`MovingPlatform`, `Spring`, `Checkpoint`, `CrumblingPlatform`, `Key`, `Door`, `PushableBox`, `SwingingHazard`, `OneWayPlatform`, `Charger`) have a `draw(renderer, alpha)` method that delegates to a corresponding renderer method.
- [ ] `Spike.draw` honors `self.orientation` (renders tip in the correct direction).
- [ ] App boots end-to-end through MenuScene → PlayScene → draw without raising.

**Verify:** `pytest -q tests/ -v` + manual smoke (launch app, see all 4 levels render).

**Steps:**

- [ ] **Step 1: Add draw methods to Renderer**

In `src/blueball/render/renderer.py`, add (consolidated):

```python
_MOVING_PLATFORM_COLOR = (200, 160, 110)
_SPRING_COLOR = (170, 170, 220)
_CHECKPOINT_COLOR = (90, 220, 140)
_CHECKPOINT_ACTIVE_COLOR = (255, 220, 80)
_CRUMBLING_COLOR = (160, 110, 90)
_KEY_COLOR = (255, 200, 60)
_DOOR_COLOR = (140, 110, 200)
_DOOR_OPEN_COLOR = (90, 80, 130)
_BOX_COLOR = (200, 170, 130)
_SWINGING_COLOR = (220, 80, 80)
_ONE_WAY_COLOR = (140, 200, 220)
_CHARGER_PATROL_COLOR = (200, 80, 80)
_CHARGER_CHARGE_COLOR = (255, 120, 120)


def draw_moving_platform(self, body, alpha: float, length: float) -> None:
    px, py = body.position
    cx, cy = self._w2s((px, py))
    half_len_screen = int(length / 2 * self._camera.zoom if hasattr(self._camera, 'zoom') else length / 2)
    pygame.draw.rect(self.screen, _MOVING_PLATFORM_COLOR, (cx - half_len_screen, cy - 5, half_len_screen * 2, 10))


def draw_spring(self, pos, width, t) -> None:
    cx, cy = self._w2s(pos)
    half = int(width / 2)
    pygame.draw.rect(self.screen, _SPRING_COLOR, (cx - half, cy - 4, half * 2, 8))


def draw_checkpoint(self, pos, radius, t, active: bool) -> None:
    cx, cy = self._w2s(pos)
    color = _CHECKPOINT_ACTIVE_COLOR if active else _CHECKPOINT_COLOR
    pygame.draw.polygon(self.screen, color, [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)])


def draw_crumbling_platform(self, body, alpha, width, t_progress: float) -> None:
    cx, cy = self._w2s(body.position) if hasattr(body, 'position') else self._w2s(body)
    half = int(width / 2)
    g = max(40, int(110 - 70 * t_progress))
    pygame.draw.rect(self.screen, (160, g, 90), (cx - half, cy - 5, half * 2, 10))


def draw_key(self, pos, radius, key_id: int) -> None:
    cx, cy = self._w2s(pos)
    pygame.draw.circle(self.screen, _KEY_COLOR, (cx, cy), radius)


def draw_door(self, pos, height: float, open_: bool) -> None:
    cx, cy = self._w2s(pos)
    half = int(height / 2)
    color = _DOOR_OPEN_COLOR if open_ else _DOOR_COLOR
    if open_:
        pygame.draw.rect(self.screen, color, (cx - 4, cy - half, 8, half * 2), width=2)
    else:
        pygame.draw.rect(self.screen, color, (cx - 6, cy - half, 12, half * 2))


def draw_pushable_box(self, body, alpha, size: float) -> None:
    cx, cy = self._w2s(body.position)
    half = int(size / 2)
    pygame.draw.rect(self.screen, _BOX_COLOR, (cx - half, cy - half, half * 2, half * 2))


def draw_swinging_hazard(self, anchor_pos, bob_body, alpha, bob_radius=14) -> None:
    ax, ay = self._w2s(anchor_pos)
    bx, by = self._w2s(bob_body.position)
    pygame.draw.line(self.screen, (180, 180, 180), (ax, ay), (bx, by), 2)
    pygame.draw.circle(self.screen, _SWINGING_COLOR, (bx, by), int(bob_radius))


def draw_one_way_platform(self, pos, width: float) -> None:
    cx, cy = self._w2s(pos)
    half = int(width / 2)
    pygame.draw.rect(self.screen, _ONE_WAY_COLOR, (cx - half, cy - 4, half * 2, 8))
    # chevron arrow indicating one-way (down)
    pygame.draw.polygon(self.screen, _ONE_WAY_COLOR, [(cx - 8, cy + 4), (cx + 8, cy + 4), (cx, cy + 14)])


def draw_charger(self, body, alpha, state: str, radius=12) -> None:
    cx, cy = self._w2s(body.position)
    color = _CHARGER_CHARGE_COLOR if state == "charge" else _CHARGER_PATROL_COLOR
    pygame.draw.circle(self.screen, color, (cx, cy), int(radius))
```

(Adapt `self._w2s` to whatever world-to-screen helper your renderer already exposes; reuse the existing pattern from `draw_ball` / `draw_collectible`.)

- [ ] **Step 2: Add `draw` methods to each new entity**

For each entity, add:

```python
# MovingPlatform
def draw(self, renderer, alpha):
    renderer.draw_moving_platform(self.body, alpha, self.length)

# Spring
def draw(self, renderer, alpha):
    import pygame as _pg
    t = _pg.time.get_ticks() / 1000.0
    renderer.draw_spring(self.position, self.width, t)

# Checkpoint
def draw(self, renderer, alpha):
    import pygame as _pg
    t = _pg.time.get_ticks() / 1000.0
    renderer.draw_checkpoint(self.position, 18, t, self.activated)

# CrumblingPlatform
def draw(self, renderer, alpha):
    t_progress = min(1.0, self._contact_timer / self.crumble_delay_s) if self._contacted else 0.0
    if self._removed:
        return
    renderer.draw_crumbling_platform(self.position, alpha, self.width, t_progress)

# Key
def draw(self, renderer, alpha):
    if self._collected:
        return
    renderer.draw_key(self.position, self.radius, self.key_id)

# Door
def draw(self, renderer, alpha):
    renderer.draw_door(self.position, self.height, self.is_open)

# PushableBox
def draw(self, renderer, alpha):
    renderer.draw_pushable_box(self.body, alpha, self.size)

# SwingingHazard
def draw(self, renderer, alpha):
    renderer.draw_swinging_hazard(self.anchor_pos, self.bob_body, alpha)

# OneWayPlatform
def draw(self, renderer, alpha):
    renderer.draw_one_way_platform(self.position, self.width)

# Charger
def draw(self, renderer, alpha):
    if not self.alive:
        return
    renderer.draw_charger(self.body, alpha, self.state)
```

- [ ] **Step 3: Update Spike.draw to honor orientation**

In `src/blueball/entities/spike.py`, the existing `draw` method should rotate the triangle to match `self.orientation`. Update the renderer's `draw_spike` (or wherever the triangle is drawn) to read `entity.orientation` and place the tip accordingly.

- [ ] **Step 4: Run tests + manual smoke**

```bash
pytest -q tests/ -v
python main.py  # boot — confirm MenuScene renders, all 4 levels selectable
```

- [ ] **Step 5: Commit**

```bash
git add src/blueball/render/renderer.py src/blueball/entities/
git commit -m "feat: renderer + draw methods for all Phase 3 entities"
```

---

## Task 29: Vertical Climb level JSON

**Goal:** Hand-authored `vertical_climb.json` exercising vertical_column, platform, spring, moving_platform (axis "y"), one_way_platform, crumbling_platform, checkpoint, and the swinging_hazard.

**Files:**
- Create: `src/blueball/levels/vertical_climb.json`
- Modify: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] File loads via `load_level` without raising.
- [ ] Level is clearable in manual playtest (subjective; not asserted by tests).
- [ ] Test `test_load_vertical_climb_smoke` constructs the level into a World and confirms `level_meta.name == "Vertical Climb"`.

**Verify:** `pytest -q tests/test_level_loader.py -v` and manual playtest.

**Steps:**

- [ ] **Step 1: Create the JSON**

`src/blueball/levels/vertical_climb.json`:

```json
{
  "name": "Vertical Climb",
  "background": "#1f2540",
  "ground": "#444a5c",
  "spawn": [80, 540],
  "chunks": [
    {"type": "flat", "width_tiles": 3},
    {"type": "ability_pickup", "width_tiles": 2, "ability": "double_jump"},
    {"type": "flat", "width_tiles": 2},
    {"type": "spring", "width_tiles": 2, "impulse": 600},
    {"type": "vertical_column", "width_tiles": 6, "steps": 4, "step_height": 80, "bottom_offset": 96, "platform_tiles": 2},
    {"type": "platform", "width_tiles": 3, "y_offset": 256},
    {"type": "moving_platform", "width_tiles": 4, "length_tiles": 2, "axis": "y", "range_px": 160, "speed": 60, "y_offset": 200},
    {"type": "platform", "width_tiles": 3, "y_offset": 288},
    {"type": "checkpoint", "width_tiles": 2, "y_offset": 64, "id": 0},
    {"type": "swinging_hazard", "width_tiles": 4, "anchor_y_offset": 224, "rope_length": 128, "initial_angle_deg": 40},
    {"type": "one_way_platform", "width_tiles": 4, "y_offset": 96},
    {"type": "vertical_column", "width_tiles": 6, "steps": 5, "step_height": 80, "bottom_offset": 96, "platform_tiles": 2},
    {"type": "crumbling_platform", "width_tiles": 2, "y_offset": 0, "crumble_delay_s": 0.4},
    {"type": "flat", "width_tiles": 3},
    {"type": "goal"}
  ]
}
```

- [ ] **Step 2: Failing test**

```python
def test_load_vertical_climb_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "vertical_climb.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Vertical Climb"
    assert meta.total_width > 0
```

- [ ] **Step 3: Run, confirm pass**

- [ ] **Step 4: Manual playtest**

Launch `python main.py`, select Vertical Climb, play through. Iterate on the chunk sequence as needed for feel. **Do NOT commit during feel-tuning** — wait for an explicit "commit" from the user.

- [ ] **Step 5: Commit (after user signoff)**

```bash
git add src/blueball/levels/vertical_climb.json tests/test_level_loader.py
git commit -m "feat: vertical_climb level"
```

---

## Task 30: Speed Run level JSON

**Goal:** Hand-authored `speed_run.json` — momentum-focused, boost pads, ice floor, crumbling platforms over gaps, no vertical_column.

**Files:**
- Create: `src/blueball/levels/speed_run.json`
- Modify: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] Loads cleanly.
- [ ] Mostly horizontal traversal; uses boost_pad, ice_floor, crumbling_platform, moving_platform (axis "x"), bump, gap.

**Verify:** `pytest -q tests/test_level_loader.py -v` + manual playtest.

**Steps:**

- [ ] **Step 1: Create the JSON**

`src/blueball/levels/speed_run.json`:

```json
{
  "name": "Speed Run",
  "background": "#fed28b",
  "ground": "#a17a3a",
  "spawn": [80, 540],
  "chunks": [
    {"type": "flat", "width_tiles": 4},
    {"type": "boost_pad", "width_tiles": 4, "multiplier": 2.0},
    {"type": "flat", "width_tiles": 4},
    {"type": "ice_floor", "width_tiles": 5},
    {"type": "gap", "width_tiles": 3},
    {"type": "crumbling_platform", "width_tiles": 2, "y_offset": 0, "crumble_delay_s": 0.4},
    {"type": "crumbling_platform", "width_tiles": 2, "y_offset": 0, "crumble_delay_s": 0.4},
    {"type": "gap", "width_tiles": 3},
    {"type": "flat", "width_tiles": 3},
    {"type": "bump", "width_tiles": 2, "height": 32},
    {"type": "flat", "width_tiles": 2},
    {"type": "boost_pad", "width_tiles": 3, "multiplier": 2.2},
    {"type": "flat", "width_tiles": 4},
    {"type": "spike_pit", "width_tiles": 3, "spikes": 3},
    {"type": "flat", "width_tiles": 4},
    {"type": "moving_platform", "width_tiles": 6, "length_tiles": 3, "axis": "x", "range_px": 200, "speed": 100, "y_offset": 0},
    {"type": "falling_hazard", "width_tiles": 4, "hazard_height": 220},
    {"type": "flat", "width_tiles": 4},
    {"type": "patrol_platform", "length_tiles": 6, "patroller_speed": 80},
    {"type": "boost_pad", "width_tiles": 4, "multiplier": 2.0},
    {"type": "flat", "width_tiles": 4},
    {"type": "goal"}
  ]
}
```

- [ ] **Step 2: Add loader smoke test**

```python
def test_load_speed_run_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "speed_run.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Speed Run"
```

- [ ] **Step 3: Manual playtest + iterate**

Tune chunk sequence by hand until the level plays right. No auto-commits during tuning.

- [ ] **Step 4: Commit (after user signoff)**

```bash
git add src/blueball/levels/speed_run.json tests/test_level_loader.py
git commit -m "feat: speed_run level"
```

---

## Task 31: Maze level JSON

**Goal:** Hand-authored `maze.json` — routing puzzle with keys, doors, a charger, a pushable_box on a spring, and ceiling spike_walls.

**Files:**
- Create: `src/blueball/levels/maze.json`
- Modify: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] Loads cleanly.
- [ ] Uses at least 2 `key`+`door` pairs, `pushable_box`, `charger_platform`, `boost_pad`, `one_way_platform`, `checkpoint`, `spike_wall`.

**Verify:** `pytest -q tests/test_level_loader.py -v` + manual playtest.

**Steps:**

- [ ] **Step 1: Create the JSON**

`src/blueball/levels/maze.json`:

```json
{
  "name": "Maze",
  "background": "#27435a",
  "ground": "#5a6878",
  "spawn": [80, 540],
  "chunks": [
    {"type": "flat", "width_tiles": 4},
    {"type": "key", "width_tiles": 2, "y_offset": 64, "key_id": 0},
    {"type": "flat", "width_tiles": 3},
    {"type": "spike_wall", "width_tiles": 3, "spikes": 3, "orientation": "down", "ceiling_y_offset": 140},
    {"type": "flat", "width_tiles": 3},
    {"type": "charger_platform", "length_tiles": 8, "facing": "right", "sight_range": 200, "charge_speed": 180},
    {"type": "flat", "width_tiles": 3},
    {"type": "door", "width_tiles": 2, "height_tiles": 4, "key_id": 0},
    {"type": "flat", "width_tiles": 3},
    {"type": "checkpoint", "width_tiles": 2, "y_offset": 64, "id": 0},
    {"type": "pushable_box", "width_tiles": 3, "size_px": 32, "mass": 0.5},
    {"type": "spring", "width_tiles": 2, "impulse": 700},
    {"type": "flat", "width_tiles": 2},
    {"type": "one_way_platform", "width_tiles": 4, "y_offset": 96},
    {"type": "vertical_column", "width_tiles": 6, "steps": 3, "step_height": 96, "bottom_offset": 64, "platform_tiles": 2},
    {"type": "key", "width_tiles": 2, "y_offset": 256, "key_id": 1},
    {"type": "flat", "width_tiles": 3},
    {"type": "boost_pad", "width_tiles": 3, "multiplier": 1.8},
    {"type": "flat", "width_tiles": 3},
    {"type": "door", "width_tiles": 2, "height_tiles": 4, "key_id": 1},
    {"type": "flat", "width_tiles": 4},
    {"type": "goal"}
  ]
}
```

- [ ] **Step 2: Add loader smoke test**

```python
def test_load_maze_smoke():
    from pathlib import Path
    from blueball.levels.loader import load_level
    from blueball.world import World
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "maze.json"
    w = World()
    meta = load_level(path, w)
    assert meta.name == "Maze"
```

- [ ] **Step 3: Manual playtest + iterate**

Tune until satisfying. No auto-commits during tuning.

- [ ] **Step 4: Commit (after user signoff)**

```bash
git add src/blueball/levels/maze.json tests/test_level_loader.py
git commit -m "feat: maze level"
```

---

## Task 32: Determinism re-runs — speed_run + sampler-built level

**Goal:** Extend `tests/test_world_determinism.py` to cover Phase 3: re-run determinism on `speed_run.json` and on a sampler-built level with a fixed seed. Catches any non-determinism introduced by new entities.

**Files:**
- Modify: `tests/test_world_determinism.py`

**Acceptance Criteria:**
- [ ] `test_speed_run_world_determinism` constructs two Worlds, loads `speed_run.json` into each, runs identical action streams for N ticks, asserts identical Player positions and velocities.
- [ ] `test_sampler_level_world_determinism` constructs `ChunkSampler(seed=12345)`, builds the resulting level into two Worlds, runs identical action streams, asserts identical state.

**Verify:** `pytest -q tests/test_world_determinism.py -v`

**Steps:**

- [ ] **Step 1: Write tests**

```python
import pymunk

from blueball.world import World
from blueball.entities.player import Player
from blueball.agent import Action, Agent
from blueball.collision import register
from blueball.levels.loader import load_level
from blueball.levels.sampler import ChunkSampler


class _Scripted(Agent):
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, obs):
        a = self.actions[self.i] if self.i < len(self.actions) else Action.IDLE
        self.i += 1
        return a


def _run(level_source, actions, n_ticks=300):
    w = World(seed=1)
    register(w.space, world_ref=w)
    meta = load_level(level_source, w)
    p = Player(agent=_Scripted(actions), spawn_xy=tuple(meta.spawn))
    w.add_entity(p)
    for _ in range(n_ticks):
        w.step(1 / 60)
    return (p.body.position.x, p.body.position.y, p.body.velocity.x, p.body.velocity.y)


def test_speed_run_world_determinism():
    from pathlib import Path
    actions = [Action.RIGHT] * 600
    path = Path(__file__).parent.parent / "src" / "blueball" / "levels" / "speed_run.json"
    a = _run(path, actions)
    b = _run(path, actions)
    assert a == b


def test_sampler_level_world_determinism():
    actions = [Action.RIGHT] * 600
    seq1 = list(ChunkSampler(seed=12345, target_chunks=80))
    seq2 = list(ChunkSampler(seed=12345, target_chunks=80))
    data = {
        "name": "Det", "background": "#000000", "ground": "#111111",
        "spawn": [80, 540], "chunks": seq1,
    }
    a = _run(data, actions)
    data2 = {**data, "chunks": seq2}
    b = _run(data2, actions)
    assert a == b
```

- [ ] **Step 2: Run, confirm pass**

If a determinism test fails, the most likely culprit is a non-deterministic source in one of the new entities (e.g. `Charger` calling `random` outside the world's seeded RNG, or wall-clock-based timing). Trace via `git bisect` against the per-entity commits.

- [ ] **Step 3: Commit**

```bash
git add tests/test_world_determinism.py
git commit -m "test: determinism guards for speed_run and sampler-built levels"
```

---

## Final smoke pass

After all tasks complete:

- [ ] Run the full suite: `pytest -q tests/ -v`. All tests pass.
- [ ] Run the game: `python main.py`. Menu appears with 5 entries. Each level loads. Infinite Run generates a fresh procedural level each selection.
- [ ] Update `~/.claude/projects/-home-ddgg0-projects-BlueBall/memory/project_blue_ball_phase_2.md` → rename to `project_blue_ball_phase_3.md` (or update in place) per the Phase 3 handoff note in the user's brief. Record:
  - What shipped (14 chunks, 10 entities, 3 levels, sampler, menu, enriched Observation).
  - Final Observation shape (or just point at the spec).
  - Where the AI session should wire raycasts (it doesn't need to — they're wired). The AI session just reads `Observation` fields.
  - Streaming infinite mode and sampler-difficulty hand-tuning deferred to Phase 4.

