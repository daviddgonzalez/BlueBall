# Abilities Framework — Design Spec

**Date:** 2026-05-25
**Status:** Approved for planning
**Phase:** 2, slice 1 of 2 (slice 2 = boost pads, already specced 2026-05-23)

## Summary

Add an **ability framework** to the Player so that mechanics beyond "move + jump" can be unlocked mid-game and persisted across runs. The framework ships with three pieces:

1. A **slot system on `Player`** that tracks which abilities are active and routes per-ability logic.
2. A **simple JSON save file** that records unlocked abilities and is read at level start / written when a new ability is picked up.
3. A **level-side unlock trigger** — a new sensor entity (`AbilityPickup`) and matching chunk — that grants an ability on contact.

The first ability shipped is **double jump**.

## Motivation

The v1 design spec already telegraphed unlockables ("the ability system is structured as composable verbs from the start"), and the Phase 2 handoff approved abilities as slice #1 specifically because the *framework* is the load-bearing infrastructure — once it exists, future abilities (wall jump, ground pound, etc.) are small additions. Picking double jump as the first concrete ability keeps the surface area minimal: it slots cleanly into the existing `JumpController` state machine and needs no new input.

The save file gets included in this slice rather than deferred because without persistence the level designer can't place "unlock X here" pickups meaningfully — the player would re-unlock every run.

## Behavior

### Ability framework

| When | What happens |
|---|---|
| `PlayScene._reset()` runs | Save file is read; the unlocked-ability set is passed into `Player.__init__`. |
| Player overlaps an `AbilityPickup` sensor | `player.unlock(pickup.ability)` is called. |
| Inside `unlock(ability)` | If already unlocked → no-op. Otherwise add to the slot set, persist via `save.add_ability(ability)`, and tear down the pickup entity. |
| Player dies / respawns | Already-unlocked abilities **persist** (they're on disk). The pickup entity does not respawn if it was consumed in a previous attempt. |

### Double jump (first ability)

| When | What happens |
|---|---|
| `JumpController.tick(...)` is called and `double_jump` is in the player's ability set | The controller may fire **one** extra jump while the player is airborne and the primary jump has already been spent. |
| Player becomes grounded | The air-jump counter resets to 1 (one extra jump available next time the player leaves the ground). |
| Player fires an air jump | Counter decrements to 0. Subsequent jump presses while airborne do nothing. |
| Coyote-window jump | Counts as the primary (grounded) jump. Air-jump counter is untouched. |
| Jump cut | Applies to air jumps exactly as it does to ground jumps. |
| Player dies / respawns | New `JumpController` is constructed (Player is rebuilt by `_reset()`), so counter starts fresh. |

**Edge case — jumping off a ledge without pressing jump first:** The player walked off (didn't jump). They are airborne with the air-jump counter still at 1. Pressing jump fires the air jump. This is intentional — it's identical to how most platformers handle "double jump from a fall" and avoids a special case.

## Components

### `save.py` (new, top-level under `blueball/`)

```python
from pathlib import Path
import json, os

SAVE_PATH = Path(os.environ.get("BLUEBALL_SAVE_PATH",
                                Path.home() / ".blueball" / "save.json"))

def load() -> set[str]:
    if not SAVE_PATH.exists():
        return set()
    data = json.loads(SAVE_PATH.read_text())
    return set(data.get("unlocked_abilities", []))

def add_ability(name: str) -> None:
    abilities = load()
    if name in abilities:
        return
    abilities.add(name)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps({"unlocked_abilities": sorted(abilities)}, indent=2))
```

- `BLUEBALL_SAVE_PATH` env var lets tests redirect to a tmp file without monkey-patching.
- Atomic-write is deferred (single-user local file; corruption tolerance is fine).
- No schema versioning yet — a `"version": 1` field can be added the first time the schema changes.

### `abilities.py` (new, top-level under `blueball/`)

```python
from enum import StrEnum

class Ability(StrEnum):
    DOUBLE_JUMP = "double_jump"
    # Future: WALL_JUMP = "wall_jump", GROUND_POUND = "ground_pound"
```

`StrEnum` so the value used on disk, in level JSON, and in code is the same string. New abilities are added here.

### `entities/player.py` (modified)

- New constructor parameter:
  ```python
  def __init__(self, agent, spawn_xy, abilities: set[Ability] | None = None) -> None:
      ...
      self.abilities: set[Ability] = set(abilities or ())
      self.jump_ctrl = JumpController(abilities=self.abilities)
  ```
- New method:
  ```python
  def unlock(self, ability: Ability) -> None:
      if ability in self.abilities:
          return
      self.abilities.add(ability)
      self.jump_ctrl.abilities = self.abilities  # share by reference
      save.add_ability(ability.value)
  ```
- The shared-by-reference set means `JumpController` sees subsequent unlocks without `Player` having to push them. (`JumpController` already reads `self.abilities` per `tick`.)

### `input_feel.py` (modified)

`JumpController` gains:
- A new constructor parameter `abilities: set[Ability]`. Stored as `self.abilities`.
- A new instance field `self._air_jumps_remaining: int = 0`.
- Updated `tick(...)` logic:
  - On the *grounded* → *airborne* transition (i.e. `self._was_grounded and not grounded`), set `self._air_jumps_remaining = 1 if Ability.DOUBLE_JUMP in self.abilities else 0`.
  - Also reset `self._air_jumps_remaining` to the same value when a primary jump fires from grounded or coyote (the player has now used the ground jump; the air jump becomes available).
  - Add a fourth `fire = True` branch: fresh press, airborne, no live buffer, no coyote window left, and `self._air_jumps_remaining > 0`. On fire, decrement.

Existing branches (grounded fresh press, coyote fresh press, landing-with-buffer) are unchanged. Existing buffer / coyote / cut behavior is unchanged.

### `entities/ability_pickup.py` (new)

```python
class AbilityPickup(Entity):
    def __init__(self, world, position, ability: Ability, radius: int = 18) -> None: ...
```

- Static body at `position`; `Circle` shape, `sensor=True`, `collision_type=CT_ABILITY_PICKUP`.
- Stores `self.ability` for the collision handler.
- Stores `self._collected: bool = False`; once true, `draw` is a no-op and `update` removes the body/shape from the space on the next tick (mirrors `Collectible`).

### `collision.py` (modified)

- New constant `CT_ABILITY_PICKUP = 7` (CT_BOOST_PAD is 6 in the boost-pads spec; reserve 6 first when both ship).
- New begin handler `on_ability_pickup`:
  ```python
  def on_ability_pickup(arbiter, space_, data):
      player = _find_player_entity(arbiter, world_ref)
      for shape in arbiter.shapes:
          entity = _find_entity_for_shape(shape, world_ref)
          if entity is not None and entity is not player and hasattr(entity, "ability"):
              if player is not None and not entity._collected:
                  player.unlock(entity.ability)
                  entity._collected = True
      return False  # sensor
  ```
- Registered alongside the other v1 handlers.

### `levels/chunks/ability_pickup.py` (new)

```python
@register_chunk("ability_pickup")
class AbilityPickupChunk(Chunk):
    def __init__(self, width_tiles=2, ability="double_jump", height=64): ...
    def build(self, world, x_offset):
        w = self.width_tiles * TILE
        # Ground segment beneath the pickup
        seg = pymunk.Segment(world.space.static_body,
                             (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        world.add_entity(AbilityPickup(
            world,
            position=(x_offset + w/2, GROUND_Y - self.height),
            ability=Ability(self.ability),
        ))
        return w
```

Registered via `chunks/__init__.py`.

### `scenes/play.py` (modified)

`_reset()` reads the save file once at construction time:

```python
def _reset(self) -> None:
    self.world = World()
    register_collisions(self.world.space, world_ref=self.world)
    self.level_meta = load_level(self.level_path, self.world)
    unlocked = {Ability(name) for name in save.load() if name in Ability._value2member_map_}
    self.player = Player(
        agent=HumanAgent(),
        spawn_xy=tuple(self.level_meta.spawn),
        abilities=unlocked,
    )
    self.world.add_entity(self.player)
    self.camera.position = (self.player.body.position.x, self.player.body.position.y)
```

The filter against `Ability._value2member_map_` makes the loader forward-compatible: a save file with an ability we don't know about yet is silently ignored rather than crashing.

### `render/renderer.py` (modified)

```python
def draw_ability_pickup(self, pos, radius, ability: str, t: float):
    # Pulsing diamond. Color is per-ability so the player can recognize
    # which power they're about to grab.
    color = _ABILITY_COLORS.get(ability, _ABILITY_COLORS_DEFAULT)
    cx, cy = self._w2s(pos)
    pulse = 1.0 + 0.15 * math.sin(t * 4.0)
    r = int(radius * pulse)
    pygame.draw.polygon(self.screen, color, [
        (cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy),
    ])
```

New module-level constants:
- `_ABILITY_COLORS = {"double_jump": (255, 220, 80)}` (warm yellow)
- `_ABILITY_COLORS_DEFAULT = (220, 220, 220)`

Animation time `t` comes from `pygame.time.get_ticks() / 1000.0`, matching the pulse pattern used for `Collectible`.

### `config.py` (modified)

```python
# Abilities
ABILITY_PICKUP_DEFAULT_HEIGHT = 64    # px above ground; matches typical jump arc
```

(No new tunables for double jump itself — the air jump reuses `JUMP_IMPULSE` and `JUMP_CUT_FACTOR`. If the feel is wrong, a `DOUBLE_JUMP_IMPULSE_FACTOR` tunable lands during feel-tuning, not in this spec.)

### `levels/tutorial_hill.json` (modified)

Insert one ability pickup early enough that the player has the double jump available for most of the level. Proposed: right after the first `flat` so the tutorial teaches "grab thing, jump farther" immediately. The chunk after the pickup should be a `gap` whose width is barely beyond a single jump's range so the new ability is obviously useful.

```diff
   {"type": "flat", "width": 8},
+  {"type": "ability_pickup", "width_tiles": 2, "ability": "double_jump"},
+  {"type": "gap", "width": 6},
   {"type": "spike_pit", "width": 3, "spikes": 4},
```

The exact `gap` width is part of feel-tuning, not spec.

## Testing

### `tests/test_save.py` (new)

- `test_load_returns_empty_set_when_file_missing` — point `BLUEBALL_SAVE_PATH` at a nonexistent tmp path, assert `save.load() == set()`.
- `test_add_ability_creates_file_and_persists` — call `save.add_ability("double_jump")` with a tmp save path, assert the file exists and `save.load() == {"double_jump"}`.
- `test_add_ability_is_idempotent` — call `add_ability("double_jump")` twice; assert the file's JSON list contains exactly one entry.
- `test_add_ability_preserves_existing_unlocks` — seed file with `{"unlocked_abilities": ["wall_jump"]}`; call `add_ability("double_jump")`; assert both present, sorted.

### `tests/test_player.py` additions

- `test_player_starts_with_no_abilities_by_default` — `Player(...)` then assert `p.abilities == set()`.
- `test_player_unlock_adds_to_set_and_persists` — construct Player with empty abilities, call `p.unlock(Ability.DOUBLE_JUMP)` (using tmp save path), assert the set updates and `save.load()` reflects it.
- `test_player_unlock_is_idempotent` — calling `unlock` twice doesn't double-write or re-trigger anything.

### `tests/test_input_feel.py` additions

- `test_double_jump_disabled_when_ability_missing` — `JumpController(abilities=set())`; simulate ground jump, then air press; only the first fires.
- `test_double_jump_fires_one_extra_air_jump_when_unlocked` — `JumpController(abilities={Ability.DOUBLE_JUMP})`; ground jump → press jump in air → second press in air; first air press fires, second doesn't.
- `test_double_jump_resets_on_landing` — ground jump → air jump → land (grounded=True tick) → ground jump → air jump again; both air jumps fire.
- `test_double_jump_available_after_walk_off_ledge` — never call a ground jump; transition grounded→airborne by setting grounded=False; press jump in air; it fires.
- `test_double_jump_does_not_break_jump_buffer_or_coyote` — re-run two existing buffer/coyote tests with `abilities={Ability.DOUBLE_JUMP}`; behavior is identical to base.

### `tests/test_entities.py` additions

- `test_ability_pickup_is_sensor_with_correct_collision_type` — constructs `AbilityPickup`, asserts `shape.sensor is True` and `shape.collision_type == collision.CT_ABILITY_PICKUP`.
- `test_ability_pickup_stores_ability` — `AbilityPickup(..., ability=Ability.DOUBLE_JUMP).ability == Ability.DOUBLE_JUMP`.

### `tests/test_chunks.py` additions

- `"ability_pickup"` added to the existing `test_registry_has_all_v1_chunks` list (or rename it; in either case `ability_pickup` must be in the registry assertion).
- `test_ability_pickup_chunk_adds_one_pickup_entity` — build chunk, assert one `AbilityPickup` is in `world.entities`, and that its `ability` matches the JSON-supplied string.

### `tests/test_collision.py` additions

- `test_player_unlocks_ability_on_pickup_contact` — set up world with player and an overlapping `AbilityPickup(ability=DOUBLE_JUMP)`, step once, assert `Ability.DOUBLE_JUMP in player.abilities`, assert `pickup._collected is True`, and (with tmp save path) assert the save file was written.

### `tests/test_play_scene.py` additions

- `test_play_scene_loads_unlocked_abilities_from_save` — write a save file with `{"unlocked_abilities": ["double_jump"]}` (via tmp path), construct `PlayScene`, assert `scene.player.abilities == {Ability.DOUBLE_JUMP}`.
- `test_play_scene_ignores_unknown_abilities_in_save` — save file contains `"frobnicate"`; PlayScene constructs without error and the bogus ability is filtered out.

## What's deliberately out of scope

- **Second ability** (wall jump, ground pound). The framework is built to absorb them; each one is its own follow-up.
- **In-game UI showing which abilities are unlocked.** Visual feedback for double jump is implicit (the second jump just works). A HUD/inventory panel is a future cosmetic pass.
- **Level-local ability disabling** ("you have double jump, but this level forbids it"). No level so far needs it; revisit when designing a level that does.
- **Cosmetic effects on the player when an ability is active** (e.g. a glow during the airborne window where double jump is available). Pure polish; deferred.
- **Save-file migration / versioning.** A `version` field gets added when the schema first changes. Until then, the loader is tolerant of unknown ability strings.
- **Concurrent-process safety on the save file.** Single-user local game; not a concern.
- **Audio cue on pickup or on double-jump fire.** No audio system yet (v1 deferred it).

All of these can be layered on later without restructuring this design.

## Acronyms used in this document

- **CT** — Collision Type (numeric tag pymunk uses to dispatch contact handlers; existing convention from v1 `collision.py`)
- **HUD** — Heads-Up Display
- **JSON** — JavaScript Object Notation
