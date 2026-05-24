# Boost Pads — Design Spec

**Date:** 2026-05-23
**Status:** Approved

## Summary

Add **boost pads**: floor-strip sensors that raise the player's max-linear-speed cap (and matching max-angular-velocity cap) until the next time the player transitions from airborne to grounded. Multiple pads stack take-the-max; touching a weaker pad while a stronger one is active is a no-op.

## Motivation

The base feel locks the ball at `MAX_LINEAR_SPEED = 600 px/s`. Boost pads give the level designer a per-segment "go faster here" tool that pairs naturally with the jump/gap mechanics — roll over a pad, sprint into a gap, fly farther. The boost is consumed on landing, so the rush is short and committed.

## Behavior

| When | What happens |
|---|---|
| Player overlaps a `BoostPad` sensor | `player.receive_boost(pad.multiplier)` is called. |
| Inside `receive_boost(m)` | If `m > _boost_multiplier`, set `_boost_multiplier = m`; arm "ends on next landing" tracking. Otherwise no-op (take-the-max). |
| Each physics tick, player is airborne | `_aerial_since_pickup = True`. |
| Each physics tick, player is grounded AND `_aerial_since_pickup` AND `_boost_multiplier > 1.0` | Boost ends: `_boost_multiplier = 1.0`, `_aerial_since_pickup = False`. |
| Player dies / respawns | Boost cleared (`_reset()` rebuilds the Player). |

**Edge case — touching a pad while grounded and never jumping:** The boost stays active indefinitely until the player jumps and lands. This is consistent with "until you land" and avoids a 1-tick boost for pads on flat ground.

**Effective caps** (used by Player's existing velocity-cap and angular-velocity clamps):
- `effective_max_speed = config.MAX_LINEAR_SPEED * _boost_multiplier`
- `effective_max_ang_vel = config.MAX_ANGULAR_VEL * _boost_multiplier`

Scaling both keeps `effective_max_ang_vel * BALL_RADIUS ≈ effective_max_speed`, so the ball never spins faster than it can translate (avoids visual slipping).

## Components

### `entities/boost_pad.py` (new)

```python
class BoostPad(Entity):
    def __init__(self, world, position, width=128, multiplier=2.0): ...
```

- Pymunk `Body(body_type=STATIC)` at `position`, `Poly` rectangle (width × ~16px tall), `sensor=True`, `collision_type=CT_BOOST_PAD`.
- Stores `self.multiplier` for the collision handler to read.
- `draw(renderer, alpha)` → `renderer.draw_boost_pad(self.position, self.width)`.

### `collision.py` (modified)

- New constant `CT_BOOST_PAD = 6`.
- New begin handler `on_boost_pad`:
  ```python
  def on_boost_pad(arbiter, space_, data):
      player = _find_player_entity(arbiter, world_ref)
      for shape in arbiter.shapes:
          entity = _find_entity_for_shape(shape, world_ref)
          if entity is not None and entity is not player and hasattr(entity, "multiplier"):
              if player is not None:
                  player.receive_boost(entity.multiplier)
      return False  # sensor
  ```
- Registered alongside the other v1 handlers.

### `entities/player.py` (modified)

- New instance state in `__init__`:
  ```python
  self._boost_multiplier: float = 1.0
  self._aerial_since_pickup: bool = False
  ```
- New method `receive_boost(self, multiplier: float)`:
  ```python
  if multiplier > self._boost_multiplier:
      self._boost_multiplier = multiplier
      self._aerial_since_pickup = not self.grounded
  ```
- New private method `_update_boost(self, grounded: bool)` called from `update()` after `_refresh_contact_normals()`:
  ```python
  if self._boost_multiplier <= 1.0:
      return
  if not grounded:
      self._aerial_since_pickup = True
  elif self._aerial_since_pickup:
      self._boost_multiplier = 1.0
      self._aerial_since_pickup = False
  ```
- Existing speed-cap and angular-cap reads multiplied by `self._boost_multiplier`.

### `levels/chunks/boost_pad.py` (new)

```python
@register_chunk("boost_pad")
class BoostPadChunk(Chunk):
    def __init__(self, width_tiles=4, multiplier=2.0): ...
    def build(self, world, x_offset):
        w = self.width_tiles * TILE
        # Ground segment under the pad
        seg = pymunk.Segment(world.space.static_body,
                             (x_offset, GROUND_Y), (x_offset + w, GROUND_Y), 5)
        seg.friction = 1.0
        world.space.add(seg)
        # BoostPad sensor centered on the segment, sitting flush with the ground
        world.add_entity(BoostPad(world, position=(x_offset + w/2, GROUND_Y - 8),
                                  width=w, multiplier=self.multiplier))
        return w
```

Registered via `chunks/__init__.py`.

### `render/renderer.py` (modified)

```python
def draw_boost_pad(self, pos, width):
    # Flat cyan strip with a forward-pointing chevron
    x, y = pos
    hw = width / 2
    pad_h = 8
    p1 = self._w2s((x - hw, y - pad_h))
    p2 = self._w2s((x + hw, y - pad_h))
    p3 = self._w2s((x + hw, y + pad_h))
    p4 = self._w2s((x - hw, y + pad_h))
    pygame.draw.polygon(self.screen, _BOOST_PAD_COLOR, [p1, p2, p3, p4])
    # Center chevron pointing right
    cx, cy = self._w2s((x, y))
    pygame.draw.polygon(self.screen, _BOOST_PAD_EDGE,
                        [(cx - 8, cy - 6), (cx + 6, cy), (cx - 8, cy + 6)])
```

New module-level color constants:
- `_BOOST_PAD_COLOR = (80, 220, 240)` (cyan)
- `_BOOST_PAD_EDGE = (30, 150, 180)` (deeper cyan)

### `config.py` (modified)

```python
# Boost pad
BOOST_PAD_DEFAULT_MULTIPLIER = 2.0
```

(Color constants live in `renderer.py` next to the other entity colors, not in config — matches existing pattern.)

### `levels/tutorial_hill.json` (modified)

Insert one boost-pad chunk in a place where the speed gain is meaningful. Proposed: between the patrol_platform and the stairs_up — gives the player a chance to feel the boost on flat ground, then keep it through the stair climb if they jump.

```diff
   {"type": "patrol_platform", "length_tiles": 6, "patroller_speed": 60},
+  {"type": "boost_pad", "width_tiles": 3, "multiplier": 2.0},
   {"type": "stairs_up", "steps": 3, "step_height": 32},
```

## Testing

### `tests/test_entities.py` additions

- `test_boost_pad_is_sensor_with_correct_collision_type` — constructs `BoostPad`, asserts `shape.sensor is True` and `shape.collision_type == collision.CT_BOOST_PAD`.
- `test_boost_pad_stores_multiplier` — `BoostPad(..., multiplier=1.7).multiplier == 1.7`.

### `tests/test_player.py` additions

- `test_player_receive_boost_raises_multiplier` — call `p.receive_boost(2.0)`, assert `p._boost_multiplier == 2.0`.
- `test_player_receive_boost_takes_max` — call `(1.5)` then `(1.2)`, assert still `1.5`. Then `(2.0)`, assert `2.0`.
- `test_player_boost_clears_on_air_to_ground_transition` — manually toggle the contact-normal state (or just call `_update_boost(grounded=False)` then `_update_boost(grounded=True)`), assert multiplier returns to 1.0.
- `test_player_boost_persists_while_grounded_until_jump_land_cycle` — call `receive_boost` while `grounded=True`, then several `_update_boost(grounded=True)` ticks, assert still active. Toggle to `grounded=False`, then `grounded=True`, assert cleared.

### `tests/test_chunks.py` additions

- `"boost_pad"` added to the existing `test_registry_has_all_v1_chunks` list.
- `test_boost_pad_chunk_adds_one_boost_pad_entity` — build chunk, assert one `BoostPad` is in `world.entities`.

### `tests/test_collision.py` additions

- `test_player_gets_boost_on_pad_contact` — set up world with player and a `BoostPad` overlapping the player, step once, assert `player._boost_multiplier > 1.0`.

## What's deliberately out of scope

- **Directional/oriented pads** that impulse you horizontally.
- **Instant velocity bump** to match the new cap (player has to accelerate into it).
- **Time-based boost decay** (only landing ends the boost).
- **Audio cues / particles** (no audio system yet; particles aren't in v1).
- **Persistent boost across deaths** (respawn rebuilds the Player; boost is lost).
- **Multiplying multiple active boosts** (take-the-max only).

All of these can be layered on later without restructuring this design.
