# Phase 3 — Level Design & Content Expansion — Design Spec

**Date:** 2026-05-26
**Status:** Approved for planning
**Phase:** 3, level-design branch (AI/Genetic Algorithm (GA) scaffolding runs in a separate worktree session and is out of scope for this spec)

## Summary

Phase 3 is split into two parallel branches, each driven by its own Claude session:

1. **Level-design branch (this spec)** — expand the chunk library, ship three new hand-built levels, add a level-select menu, and enrich the agent Observation so the AI session can wire raycasts to it after merge.
2. **AI/GA scaffolding branch (separate session)** — scaffolds Fixed-Topology Neural Network (FTNN) agents, the GA trainer, and `TrainScene` against the current `Observation` (rays as a zero array). It does not touch anything this spec owns.

The level-design slice ships:

- **14 new chunk types**, bringing the library from 10 to 24.
- **3 new hand-built levels** — Vertical Climb, Speed Run, Maze — each with distinct feel and a mandatory subset of new chunks.
- A **`MenuScene`** for level selection; all four levels (Tutorial Hill + the three new ones) open from the start.
- **Observation enrichment**: the 8 raycast slots actually contain raycast results, a parallel `ray_hit_types` array categorizes what each ray hit, scalar fields surface the nearest pickup and nearest hazard, and bitfields expose unlocked abilities and collected keys.

Chunk Sampler / infinite mode is **deferred to a follow-up phase**; the chunk library shipped here is sized so that Phase 4 can stitch it procedurally without further chunk authoring.

## Motivation

The Phase 2 handoff memory committed Phase 3 to **levels-first** rather than the original AI-first ordering. Three reasons:

- The AI session needs diverse training environments to generalize. A GA trained on one hand-built level memorizes muscle memory; one trained on a rich chunk library generalizes through the raycast Observation. Building chunks before GA wiring stops the AI work from over-fitting on day one.
- Hand-built levels are more fun to play with than untrained agents — they unlock testing of the abilities framework and boost pads in varied contexts.
- The AI session is explicitly told to scaffold on `tutorial_hill` with the v1 Observation (rays = zeros), and to defer raycast wiring until this branch merges. The level-design branch is the unblocker for that follow-up.

The chunk catalog is sized to support both the three hand-built levels and a future procedural sampler. Each chunk is a self-contained build-and-register module so Phase 4 can add a `ChunkSampler` without touching this slice.

## Behavior

### Level select & progression

| When | What happens |
|---|---|
| App start | `MenuScene` is shown with cursor on Tutorial Hill. All four levels are listed and selectable. |
| Player presses Up/Down (or W/S) | Cursor moves; clamps at list ends. |
| Player presses Enter/Space | `MenuScene` returns a fresh `PlayScene(screen, level_path=selected)`. |
| Player presses Esc in `MenuScene` | Scene loop receives `None` and the app exits. |
| Player presses Esc in `PlayScene` | `PlayScene` returns a fresh `MenuScene(screen)`; player is back at the menu. |
| Level cleared (`world.level_complete`) | `PlayScene` returns a fresh `MenuScene(screen)`; current behavior of posting `pygame.QUIT` is replaced. |

Save state is unchanged from Phase 2 — only unlocked abilities persist across runs. There is no `levels_cleared` set; all four levels are always available.

### Checkpoints

| When | What happens |
|---|---|
| Player overlaps a `Checkpoint` sensor | `player.respawn_xy = (checkpoint.x, GROUND_Y - BALL_RADIUS - 4)`. In-memory only — never persisted to the save file. |
| Player dies | `PlayScene._reset()` reads `self._last_respawn_xy` (snapshotted from the dying player); the new Player's `body.position` is overridden to that point instead of `level_meta.spawn`. |
| Level cleared | `PlayScene` clears `self._last_respawn_xy` so a re-played level starts from the level's spawn. |
| `PlayScene` is constructed fresh from `MenuScene` | `self._last_respawn_xy = None`; first life starts at the level's spawn. |

### Keys & doors

| When | What happens |
|---|---|
| Player overlaps a `Key(key_id=N)` sensor | Bit `N` is set in `player.keys_held`. Entity removes itself. Never persisted. |
| Player contacts a `Door(key_id=N)` segment for the first time | Collision handler checks `player.has_key(N)`. If true, the door's solid segment is removed (entity stays for visuals); if false, the door behaves as a solid wall. |
| Door is already open | Subsequent contacts pass through (sensor or removed-shape, depending on implementation). |
| Player dies | Keys-held bitfield is reset (rebuilt Player). Doors that were opened during the failed run reset to closed when the level rebuilds. |

### Springs vs boost pads

| Mechanic | Direction | Affects | Trigger model |
|---|---|---|---|
| Boost pad (existing) | Horizontal — multiplies horizontal speed cap | Player only | Take-the-max multiplier, in-memory until next landing |
| Spring (new) | Vertical — direct upward impulse | Any dynamic body (Player, pushable boxes) | Per-contact impulse, no consume, no cooldown beyond pymunk's contact rules |

### Crumbling platforms

| When | What happens |
|---|---|
| Static platform with no contacts | Renders normally; no timer. |
| Player (or any dynamic body) lands on it | Per-platform timer starts. Configurable `crumble_delay_s`, default 0.5s. |
| Timer expires | Entity marks itself for removal; next `update()` tick calls `space.remove(shape, body)` (avoids removing shapes inside collision callbacks). |
| Platform removed | Renders nothing thereafter; gone for the rest of the run. Rebuilt on level reload. |

### Charger enemy

| State | Behavior |
|---|---|
| Patrolling | Walks back and forth on a platform at `patrol_speed`, identical to `Patroller`. Each tick, checks Field of View (FOV) cone for player presence. |
| Player inside FOV cone AND Line of Sight (LOS) clear | Switches to charge state. |
| Charging | Moves at `charge_speed` toward the player until LOS lost or it reaches the platform's edge bound. Then reverts to patrol. |
| Player contact from above | Charger dies (like Patroller). |
| Player contact from side | Player dies. |

FOV cone is parameterized: `facing` ("left" or "right"), `sight_range` (default 200 px), `sight_arc_deg` (default 60°). LOS check is one `space.segment_query_first` from charger position to player position, filtered to static segments only. The charger's "vision" is asymmetric — it only sees in its facing direction; the player can sneak up from behind.

### Pushable boxes

| When | What happens |
|---|---|
| Player rolls into a `PushableBox` | Pymunk's default solid collision pushes the box laterally (friction-driven). No custom handler needed in the v1 implementation. |
| Box lands on a `Spring` | Box receives a mass-scaled upward impulse so it gains the same delta-v as the player would. |
| Box overlaps a hazard | Box is unaffected (only Player and Charger die on hazards). |
| Box falls below `FALL_DEATH_Y` | Box is removed from the world (out-of-bounds cleanup); player is unaffected. |

### Moving platforms

| When | What happens |
|---|---|
| Each tick | Kinematic body advances along its axis ("x" or "y") between two waypoints (`-range_px/2`, `+range_px/2` from spawn) at `speed`. Reverses on hitting either bound. |
| Player stands on it | Carried via pymunk's surface-friction. No explicit carry impulse in this slice — if playtest shows slipping, a follow-up adds a contact-velocity carry pass. |
| Box lands on it | Same friction-driven carry. |

### One-way platforms

| When | What happens |
|---|---|
| Player rising into a platform from below | Pymunk `pre_solve` callback returns `False`, disabling the contact. Player passes through. |
| Player falling onto the platform from above | Callback returns `True`; normal solid landing. |
| Dynamic box approaches from below | Same callback; boxes also pass through from below. |

The "rising from below" test uses the player's `body.velocity.y < 0` plus contact normal direction (the contact normal points downward from the platform when the player is below).

### Ice floor

A `flat`-equivalent chunk with a single tunable difference: `friction = ICE_FLOOR_FRICTION` (default 0.05). Otherwise identical geometry. The ball's torque-driven roll spins it up to top speed faster but ground-stop reversals lose grip — momentum-management mechanic.

### Spike walls

The existing `Spike` entity gains an `orientation` parameter ("up" | "down" | "left" | "right"). Default "up" preserves current behavior. The chunk `spike_wall` exposes the orientation parameter to level JSONs and places spikes against ceilings or corridor walls instead of floors.

### Swinging hazards

| Component | Behavior |
|---|---|
| Anchor | Static body at `(chunk.center_x, GROUND_Y - anchor_y_offset)`. Invisible. |
| Rope | Pymunk `PinJoint` connecting the anchor to the bob. Length = `rope_length`. |
| Bob | Dynamic body (default mass 2.0), Circle shape with `collision_type = CT_SWINGING`. Killing collision with player on any contact. |
| Initial state | Bob starts at `initial_angle_deg` from vertical, released; gravity drives the swing. |

The bob is deterministic given the world seed (no random component). World-determinism tests will exercise this.

## Components

### `levels/chunks/` (new modules)

Each registers via `@register_chunk(name)` and implements `build(world, x_offset) -> width`. Listed with their key constructor arguments:

| File | Chunk name | Constructor args |
|---|---|---|
| `platform.py` | `platform` | `width_tiles: int = 4`, `y_offset: int = 96` |
| `vertical_column.py` | `vertical_column` | `width_tiles: int = 6`, `steps: int = 5`, `step_height: int = 80`, `bottom_offset: int = 96`, `platform_tiles: int = 2` |
| `moving_platform.py` | `moving_platform` | `width_tiles: int = 4`, `length_tiles: int = 2`, `axis: str = "x"`, `range_px: float = 160`, `speed: float = 80`, `y_offset: int = 96` |
| `spring.py` | `spring` | `width_tiles: int = 2`, `impulse: float = config.SPRING_DEFAULT_IMPULSE` |
| `checkpoint.py` | `checkpoint` | `width_tiles: int = 2`, `y_offset: int = 64`, `id: int = 0` |
| `one_way_platform.py` | `one_way_platform` | `width_tiles: int = 4`, `y_offset: int = 96` |
| `crumbling_platform.py` | `crumbling_platform` | `width_tiles: int = 2`, `y_offset: int = 96`, `crumble_delay_s: float = 0.5` |
| `key.py` | `key` | `width_tiles: int = 2`, `y_offset: int = 64`, `key_id: int = 0` |
| `door.py` | `door` | `width_tiles: int = 2`, `height_tiles: int = 4`, `key_id: int = 0` |
| `pushable_box.py` | `pushable_box` | `width_tiles: int = 2`, `size_px: int = 32`, `mass: float = 0.5` |
| `spike_wall.py` | `spike_wall` | `width_tiles: int = 3`, `spikes: int = 3`, `orientation: str = "down"` |
| `swinging_hazard.py` | `swinging_hazard` | `width_tiles: int = 4`, `anchor_y_offset: int = 192`, `rope_length: float = 128`, `bob_mass: float = 2.0`, `initial_angle_deg: float = 30` |
| `ice_floor.py` | `ice_floor` | `width_tiles: int = 4` |
| `charger_platform.py` | `charger_platform` | `length_tiles: int = 8`, `facing: str = "right"`, `sight_range: float = 200`, `sight_arc_deg: float = 60`, `charge_speed: float = 180`, `patrol_speed: float = 40` |

All chunks place a ground segment at `GROUND_Y` if and only if the chunk's player-traversal path is along the ground line (e.g. `charger_platform`, `ice_floor`, `spring`, `crumbling_platform` when `y_offset == 0`). Floating chunks (`platform`, `vertical_column`, `one_way_platform` with `y_offset > 0`) skip the ground segment so vertical sections work.

### `entities/` (new modules)

Each entity follows the existing pattern: `__init__(world, position, ...)`, populates `self.bodies` and `self.shapes`, exposes an `update(dt)` method when needed.

| File | Class | Body type | Collision type |
|---|---|---|---|
| `moving_platform.py` | `MovingPlatform` | Kinematic | no new CT — the kinematic body's shape uses the default collision type and collides solidly with the player via pymunk's default contact rules, exactly like a static segment |
| `spring.py` | `Spring` | Static + sensor shape | `CT_SPRING = 9` |
| `checkpoint.py` | `Checkpoint` | Static + sensor shape | `CT_CHECKPOINT = 13` |
| `crumbling_platform.py` | `CrumblingPlatform` | Static | reuses ground collision (no CT); detects contact via per-tick `each_arbiter` scan inside its `update()` |
| `key.py` | `Key` | Static + sensor shape | `CT_KEY = 14` |
| `door.py` | `Door` | Static (segment) | `CT_DOOR = 15` |
| `pushable_box.py` | `PushableBox` | Dynamic | `CT_PUSHABLE = 10` |
| `swinging_hazard.py` | `SwingingHazard` | Static anchor + dynamic bob + PinJoint | `CT_SWINGING = 11` |
| `one_way_platform.py` | `OneWayPlatform` | Static (segment) | `CT_ONE_WAY = 8` |
| `charger.py` | `Charger` | Kinematic | `CT_CHARGER = 12` |

The existing `entities/spike.py` is modified to add an `orientation` parameter; its `__init__` rotates the vertex set accordingly. `entities/patroller.py` and other existing entities are unchanged.

### `entities/player.py` (modified)

Add fields and methods, no changes to motion/physics:

```python
self.keys_held: int = 0
self.respawn_xy: tuple[float, float] | None = None

def collect_key(self, key_id: int) -> None:
    self.keys_held |= (1 << key_id)

def has_key(self, key_id: int) -> bool:
    return bool(self.keys_held & (1 << key_id))

def receive_spring(self, impulse: float) -> None:
    # Vertical impulse, mass-scaled so the resulting upward delta-v is the
    # same regardless of body mass (a light box and the player bounce to the
    # same height). Pymunk y-down so "up" is negative y.
    self.body.apply_impulse_at_local_point((0, -impulse * self.body.mass), (0, 0))
```

`Player.update()` is not modified. `_observe()` is rewritten to produce the enriched Observation (see §Observation).

`Player.push()` from the design sections is **not** added — pymunk's default solid collision handles box pushing without a custom handler.

### `agent.py` (modified)

```python
class HitType(IntEnum):
    MISS = 0
    GROUND = 1
    HAZARD = 2
    PICKUP = 3
    GOAL = 4
    ENEMY = 5
    BLOCK = 6
    DOOR = 7

@dataclass(frozen=True)
class Observation:
    rays: np.ndarray              # shape (8,), float32, in [0, 1]; 1.0 = miss
    ray_hit_types: np.ndarray     # shape (8,), int8 (HitType values)
    vel: np.ndarray               # shape (2,), float32
    ang_vel: float
    grounded: bool
    nearest_pickup: Optional[tuple[float, float]]
    nearest_hazard: Optional[tuple[float, float]]
    abilities: int
    keys_held: int
```

`HitType` is exported. A module-level constant `_CT_TO_HITTYPE: dict[int, HitType]` maps every collision type to its hit category. Sensors register as hits (`segment_query_first` includes them by default).

Module constants:
- `MAX_RAY_LEN = 300.0`
- `NUM_RAYS = 8`
- Ray angles: `[i * 2π / NUM_RAYS for i in range(NUM_RAYS)]`, starting at 0 (due-right) and going counter-clockwise.

`Action` enum is unchanged.

### `Player._observe()` (rewritten)

```python
def _observe(self) -> Observation:
    pos = self.body.position
    rays = np.empty(NUM_RAYS, dtype=np.float32)
    hit_types = np.empty(NUM_RAYS, dtype=np.int8)
    for i, angle in enumerate(RAY_ANGLES):
        dx = math.cos(angle) * MAX_RAY_LEN
        dy = math.sin(angle) * MAX_RAY_LEN
        end = (pos.x + dx, pos.y + dy)
        hit = self._world.space.segment_query_first(
            (pos.x, pos.y), end, radius=1, shape_filter=self._ray_filter,
        )
        if hit is None:
            rays[i] = 1.0
            hit_types[i] = HitType.MISS
        else:
            rays[i] = hit.alpha  # alpha is fraction of segment, already in [0,1]
            hit_types[i] = _CT_TO_HITTYPE.get(hit.shape.collision_type, HitType.GROUND)

    nearest_pickup = self._nearest_entity_delta(_PICKUP_TYPES)
    nearest_hazard = self._nearest_entity_delta(_HAZARD_TYPES)
    abilities_bits = _abilities_to_bitfield(self.abilities)

    return Observation(
        rays=rays,
        ray_hit_types=hit_types,
        vel=np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32),
        ang_vel=self.body.angular_velocity,
        grounded=self.grounded,
        nearest_pickup=nearest_pickup,
        nearest_hazard=nearest_hazard,
        abilities=abilities_bits,
        keys_held=self.keys_held,
    )
```

`self._world` is set by `World.add_entity` (a new line: `entity._world = self`). The Player needs a world reference for the segment query. Existing entities are unaffected; the attribute is set unconditionally and just unused on entities that don't need it.

`self._ray_filter` is a `pymunk.ShapeFilter` excluding the player's own shape from raycast hits. Implementation: assign the player's shape `filter = ShapeFilter(group=_PLAYER_RAY_GROUP)` at construction; ray queries use `ShapeFilter(group=_PLAYER_RAY_GROUP)` which excludes shapes in the same group. `_PLAYER_RAY_GROUP` is a non-zero constant on Player (e.g. `1`).

`_nearest_entity_delta(type_set)` iterates `self._world.entities`, partitions by `type(entity).__name__ in type_set`, computes squared distances, returns the smallest as `(dx, dy)` or `None`. O(N) per tick; N stays under ~50 for any level.

`_abilities_to_bitfield(set[Ability]) -> int` maps each ability to its enum-declaration-order bit.

### `collision.py` (modified)

New collision-type constants in declaration order:

```python
CT_PLAYER = 1
CT_SPIKE = 2
CT_PATROLLER = 3
CT_COLLECTIBLE = 4
CT_GOAL = 5
CT_BOOST_PAD = 6
CT_ABILITY_PICKUP = 7
CT_ONE_WAY = 8
CT_SPRING = 9
CT_PUSHABLE = 10
CT_SWINGING = 11
CT_CHARGER = 12
CT_CHECKPOINT = 13
CT_KEY = 14
CT_DOOR = 15
```

New handlers in `register(space, world_ref)`:

- `on_spring` — for each non-spring shape in the arbiter: if its entity is a Player, call `player.receive_spring(spring.impulse)`; otherwise, if the body is dynamic, apply `(0, -spring.impulse * body.mass)` at the body's local center. Mass-scaled in both branches so the delta-v is identical across body masses. Sensor, returns False.
- `on_checkpoint` — sets `player.respawn_xy = (checkpoint.body.position.x, GROUND_Y - BALL_RADIUS - 4)`. Sensor.
- `on_key` — calls `player.collect_key(key.key_id)`; key marks itself collected (`_collected = True` like ability pickup).
- `on_door` — first contact: if `player.has_key(door.key_id)`, mark door open and remove its solid shape next tick; else solid contact (return True). Subsequent contacts on open doors: pass through.
- `on_charger` — mirrors `on_patroller`: top-stomp kills charger, side contact kills player.
- `on_one_way` — pre-solve callback (NOT begin): `return arbiter.shapes[?].body.velocity.y >= 0` for the dynamic body; the dynamic-body identification logic is the same arbiter-shape-ordering pattern already used in `on_patroller`.
- `on_pushable` — no custom handler in v1; pymunk's default solid handler does the pushing.
- `on_swinging` — first-contact handler that calls `player.die()`. Sensor returns False (the bob never lands on the player except via the contact normal, and it's a 1-hit kill anyway).
- `on_crumbling` — no collision-time handler; `CrumblingPlatform.update()` polls `body.each_arbiter` to detect first contact and starts its timer.

The existing handlers (`on_spike`, `on_collectible`, `on_goal`, `on_patroller`, `on_ability_pickup`, `on_boost_pad`) are unchanged.

### `scenes/menu.py` (new)

```python
class MenuScene(Scene):
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        levels_dir = Path(__file__).parent.parent / "levels"
        self.entries: list[tuple[str, Path]] = [
            ("Tutorial Hill", levels_dir / "tutorial_hill.json"),
            ("Vertical Climb", levels_dir / "vertical_climb.json"),
            ("Speed Run", levels_dir / "speed_run.json"),
            ("Maze", levels_dir / "maze.json"),
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
                    _, path = self.entries[self.cursor]
                    return PlayScene(self.screen, level_path=path)
        return self

    def update(self, frame_dt: float) -> None:
        pass

    def draw(self) -> None:
        # title centered; entries listed with > marker on cursor; footer hint
        ...
```

`PlayScene` is imported at the bottom of `menu.py` or inside `handle_events` to avoid a circular import.

### `scenes/play.py` (modified)

- Constructor signature unchanged.
- Add `self._last_respawn_xy: tuple[float, float] | None = None` and `self._exit_to_menu: bool = False` fields.
- `_reset()` overrides `player.body.position = self._last_respawn_xy` when non-None.
- `update()` on death snapshots `self._last_respawn_xy = self.player.respawn_xy` before calling `_reset()`.
- `update()` on `world.level_complete` clears `self._last_respawn_xy = None`, sets `self._exit_to_menu = True`, and returns early without further camera/physics updates that tick.
- `handle_events()`: if `self._exit_to_menu`, returns `MenuScene(self.screen)`. Else processes events normally. Esc returns `MenuScene(self.screen)` instead of `None`. `pygame.QUIT` still returns `None`.

The existing `main.py` already drives scene swaps via the `handle_events` return value — no changes to the main loop's scene-swap logic.

### `main.py` (modified)

Currently instantiates `PlayScene` directly. Change: instantiate `MenuScene(screen)` first; the main loop already handles scene transitions via the `handle_events` return value.

### `world.py` (modified — minimal)

`World.add_entity(entity)` sets `entity._world = self` at the start. This single attribute attach lets `Player._observe()` reach the space without breaking encapsulation. Existing entities ignore it.

### `render/renderer.py` (modified)

New draw methods for the new visual entity types:

- `draw_moving_platform(body, alpha, width, length)` — same primitive as `flat` but at the body's interpolated position.
- `draw_spring(pos, width, t)` — coil shape; pulses subtly using `t`. Color from a new `_SPRING_COLOR = (170, 170, 220)` constant.
- `draw_checkpoint(pos, radius, t, active: bool)` — flag-like triangle. `active=True` once touched, brighter color.
- `draw_crumbling_platform(body, alpha, width, crumble_progress: float)` — same as `flat`, tinted darker as `crumble_progress` approaches 1.0; gone when removed.
- `draw_key(pos, radius, t, key_id)` — small key icon; color indexed by `key_id`.
- `draw_door(pos, width, height, key_id, open: bool)` — vertical rectangle when closed; outline only when open.
- `draw_pushable_box(body, alpha, size)` — square at interpolated position.
- `draw_swinging_hazard(anchor_pos, bob_body, alpha, rope_length)` — line from anchor to bob, plus a spiky circle for the bob.
- `draw_one_way_platform(body, alpha, width)` — thin segment with a downward arrow chevron.
- `draw_ice_floor(start, end)` — same primitive as `flat` with a tinted color.
- `draw_charger(body, alpha, facing, state)` — patroller-like sprite with a vision-cone overlay when patrolling, recolored when charging.
- `draw_spike(pos, orientation, width, height)` — modified to honor orientation.

Renderer additions stay within the existing flat-primitive style — no asset pipeline.

### `config.py` (modified — additive)

```python
# Phase 3 chunks
ICE_FLOOR_FRICTION = 0.05
SPRING_DEFAULT_IMPULSE = 600.0  # px/s impulse magnitude, applied to bodies entering a spring sensor
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

### Level JSONs

#### `levels/vertical_climb.json` (new)

- **Spawn:** bottom-left.
- **Mood:** dark — palette `background: "#1f2540"` (dusk indigo), `ground: "#444a5c"` (slate gray).
- **Mandatory chunks:** `vertical_column`, `platform`, `spring`, `moving_platform` (axis "y"), `one_way_platform`, `crumbling_platform`, `checkpoint`, `ability_pickup` (double_jump, placed early), `goal`.
- **Signature beat:** spring → moving_platform (vertical) → ledge above a swinging_hazard pit.
- **Length:** ~60–80 tiles horizontal; height varies internally via vertical_columns.

Exact chunk sequence is finalized during feel-tuning, not in this spec.

#### `levels/speed_run.json` (new)

- **Spawn:** left, ground level.
- **Mood:** bright — palette `background: "#fed28b"` (warm sun), `ground: "#a17a3a"` (sandy).
- **Mandatory chunks:** `boost_pad` (multiple), `ice_floor`, `crumbling_platform` (over gaps), `moving_platform` (axis "x"), `bump`, `gap`, `falling_hazard`, `patrol_platform`, `spike_pit`, `goal`.
- **Signature beat:** long ice_floor → crumbling-platform sequence over a gap pit.
- **Length:** ~120–160 tiles wide, mostly horizontal.

#### `levels/maze.json` (new)

- **Spawn:** central-left.
- **Mood:** cool — palette `background: "#27435a"` (steel blue), `ground: "#5a6878"` (gray-blue).
- **Mandatory chunks:** at least 2 `key` + `door` pairs, `pushable_box`, `charger_platform`, `boost_pad`, `one_way_platform`, `checkpoint`, `spike_wall` (orientation "down" and side-facing), `swinging_hazard`, `vertical_column` (for branching), `goal` (behind a door).
- **Signature beat:** push a pushable_box onto a spring to launch it through a one_way_platform, blocking a charger's line of sight so the player can pass.
- **Length:** ~80–100 tiles wide with multiple vertical passages.

## Testing

### New test files

- **`tests/test_menu_scene.py`** — cursor moves on key events; Enter returns a `PlayScene` whose `level_path` matches the cursor; Esc returns `None`.

### Modified test files

- **`tests/test_chunks.py`** — registry asserts all 24 chunk types (10 existing + 14 new). Per-chunk smoke test that `Chunk.build` runs without raising and populates the expected entity types.
- **`tests/test_entities.py`** — per-entity unit tests:
  - `MovingPlatform` oscillates between waypoints when stepped (advance N substeps, assert position bounded by `range_px`).
  - `Spring` is a sensor with `CT_SPRING`; applies the configured impulse to a test dynamic body on contact.
  - `Checkpoint` is a sensor with `CT_CHECKPOINT`; sets `player.respawn_xy` on contact; never writes to save (assert save file unchanged).
  - `CrumblingPlatform` self-removes after `crumble_delay_s` of post-contact stepping; remains while uncontacted.
  - `Key` is a sensor; self-removes on contact; sets `player.keys_held` bit matching `key_id`.
  - `Door` is solid before key collection; collision is allowed after; passes through when open.
  - `PushableBox` is a dynamic body with finite mass and the configured friction.
  - `SwingingHazard` bob stays anchored (anchor body position unchanged); bob moves under gravity; bob contact triggers `player.die()`.
  - `Charger` patrols at `patrol_speed` until a test player is placed in its FOV cone; then accelerates to `charge_speed`; reverts when LOS broken (place a static segment between them).
  - `Spike` orientation parameter produces correct vertex sets for "up", "down", "left", "right".
  - `OneWayPlatform` pre-solve disables collision when test body has upward velocity and contact normal points down; allows otherwise.
- **`tests/test_collision.py`** — `test_spring_launches_player`, `test_spring_launches_pushable_box`, `test_checkpoint_updates_respawn`, `test_key_sets_held_bit`, `test_door_blocks_until_key`, `test_door_opens_after_key`, `test_charger_kills_on_side_contact`, `test_charger_dies_on_stomp`, `test_crumbling_starts_timer_on_contact`, `test_pushable_box_moves_under_player_contact`.
- **`tests/test_player.py`** — `test_player_starts_with_no_keys`, `test_player_collect_key_sets_bit_and_is_idempotent`, `test_player_has_key_reads_bit`, `test_player_respawn_xy_defaults_to_none`, `test_player_observation_includes_new_fields` (rays nonzero when geometry nearby; hit types correct; abilities bitfield matches `Player.abilities`; keys_held matches; `nearest_pickup`/`nearest_hazard` selected correctly from a world with multiple entities).
- **`tests/test_play_scene.py`** — `test_play_scene_respawn_uses_checkpoint_after_death`, `test_play_scene_clears_checkpoint_on_level_complete`, `test_play_scene_escape_returns_menu_scene`, `test_play_scene_level_complete_returns_menu_scene`.
- **`tests/test_level_loader.py`** — smoke-load each of the three new level JSONs.
- **`tests/test_world_determinism.py`** — re-run with `speed_run.json` (no Charger means no FOV-driven branching; cleanest determinism canary). Two `World`s with the same seed and same action stream produce identical Player positions after N ticks.

### Coverage gaps deliberately left

- Charger LOS edge cases (player hidden behind a one_way platform's underside) — manual playtest.
- Pushable box stacking / chain reactions — not exercised by level designs.
- Swinging-hazard ↔ pushable-box collision — not a designed mechanic; whatever pymunk decides is fine.
- Performance regression suite — current ~960 ray queries/second comfortably fits Pymunk; revisit only if playtest reports a slowdown.

## What's deliberately out of scope

- **ChunkSampler / infinite mode.** Phase 4 builds on this slice's chunk library; this slice does not add the sampler, difficulty curve, or `InfiniteScene`.
- **AI/GA scaffolding.** Owned by the parallel session per the Phase 2 handoff. This slice ships the Observation enrichment so the AI session has something concrete to wire to when it picks up the work.
- **Sprites / asset pipeline.** All new entities draw with flat PyGame primitives, matching the v1 visual style.
- **Audio.** Same deferral as v1.
- **HUD for keys / checkpoints / abilities.** Visual feedback is implicit (key icon disappears, checkpoint flag changes color, door visually opens). A HUD overlay is a polish follow-up.
- **Save-file changes.** No `levels_cleared` set, no checkpoint persistence, no per-level high score. The save file still only tracks unlocked abilities.
- **Pushable-box puzzles beyond what Maze ships.** No multi-block stacking puzzles, no weighted switches; the box exists to support the Maze signature beat.
- **Level editor / external authoring tool.** Level JSONs are hand-edited.
- **Save-file migration / versioning.** Same deferral as Phase 2.
- **Player API additions beyond `collect_key`, `has_key`, `receive_spring`, and the `respawn_xy` field.** Notably `Player.push()` is dropped — pymunk's default solid collision suffices for pushable boxes.
- **Carry-velocity on moving platforms.** Pymunk's surface-friction is the v1 carry mechanism. If playtest shows slipping, a follow-up pass adds an explicit contact-velocity bias.

## Implementation notes for the AI session handoff

When the AI session picks up after this slice merges, the following are guaranteed:

- `Observation.rays` contains actual raycast distances in [0, 1] (1.0 = miss). 8 rays, evenly spaced starting from due-right.
- `Observation.ray_hit_types` contains `HitType` integers in [0, 7] (per the enum above).
- `Observation.nearest_pickup` and `Observation.nearest_hazard` return world-frame deltas to the closest entity in those categories, or `None` if no such entity exists.
- `Observation.abilities` and `Observation.keys_held` are bitfields. Ability bit indices match `Ability` enum declaration order (only bit 0 = `DOUBLE_JUMP` is meaningful today).
- All ray queries exclude the player's own shape via a pymunk `ShapeFilter` group, configured on the Player at construction.
- Determinism is preserved — `test_world_determinism.py` runs against `speed_run.json` after this slice lands.

The AI session does not need to re-implement any Observation logic. It just reads the dataclass.

## Acronyms used in this document

- **CT** — Collision Type (pymunk's numeric collision-type tag)
- **FOV** — Field of View (the charger enemy's vision cone)
- **FTNN** — Fixed-Topology Neural Network (the AI agent variant the AI session is scaffolding)
- **GA** — Genetic Algorithm
- **JSON** — JavaScript Object Notation
- **LOS** — Line of Sight (the segment query used by the charger to check whether geometry blocks its view of the player)
- **QoL** — Quality of Life
