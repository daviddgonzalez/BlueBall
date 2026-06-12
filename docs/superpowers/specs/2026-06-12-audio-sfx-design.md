# Basic audio + 4 SFX — design

**Date:** 2026-06-12
**Status:** Approved (design)

## Goal

Add basic, original (non-copyrighted) sound effects: a **whoosh** on boost pads,
a **spring** boing, a **key** collect/unlock sound, and a **fanfare** at the
goal. No third-party audio — all four are synthesized procedurally.

Locked during brainstorming: approach A (synthesize → committed WAVs); a config
flag for enable/disable (no in-game mute key in v1).

## Architecture (decoupled, headless-safe)

Physics/entities stay pygame-free and deterministic; only the visual `PlayScene`
touches `pygame.mixer`. Events flow through a tiny queue on `World`:

```
collision handler (begin) -> world.emit_sound(name)   [physics layer, pygame-free]
PlayScene.update()        -> drains world.sound_events -> SoundManager.play(name)
```

This mirrors how `PlayScene` already turns player state into particles. The
headless trainer never drains the queue (the World is discarded per episode);
appending a string does not touch physics, so trainer runs stay byte-identical.

## Components / files

### 1. `tools/gen_sfx.py` (new, dev tool)
Synthesizes the four SFX with numpy and writes 16-bit mono WAVs (44.1 kHz) to
`src/blueball/assets/sfx/`. Run once; the WAVs are committed. Deterministic
(seeded noise). Sounds (all short, ~0.2–0.6 s):
- **whoosh.wav** — white noise through a fast attack/decay envelope with a
  downward amplitude+lowpass-ish sweep (boost rush).
- **spring.wav** — sine with an upward frequency glide (~300→900 Hz) and a quick
  decay (a "boing").
- **key.wav** — two short tones: a high tick (~1200 Hz) then a lower thunk
  (~500 Hz) — "click-clunk" unlock.
- **fanfare.wav** — ascending arpeggio C5–E5–G5–C6 (triangle wave) with a short
  per-note decay — classic win jingle.

### 2. `src/blueball/audio.py` (new)
`SoundManager`:
- `__init__`: if audio is enabled (see config), lazily `pygame.mixer.init()` and
  load the four WAVs into `pygame.mixer.Sound` objects, all wrapped in
  `try/except` — any failure (no audio device, CI/headless, mixer error) leaves
  the manager in a silent-disabled state.
- `play(name)`: plays the named sound if loaded; otherwise a no-op (never raises).
- Honors `config.AUDIO_ENABLED` and the `BLUEBALL_NO_AUDIO` env var — either
  off → the manager loads nothing and `play` is a no-op.

### 3. `src/blueball/world.py`
- `World.sound_events: list[str]` (init empty).
- `World.emit_sound(name: str)` → appends `name`. Pure data; no pygame.

### 4. `src/blueball/collision.py`
Emit from the existing one-shot `begin` handlers (all already have `world_ref`):
- `on_boost_pad` → `world_ref.emit_sound("whoosh")` (after `receive_boost`).
- `on_spring` → `world_ref.emit_sound("spring")` in the **player** branch (where
  `player.receive_spring(...)` is called), not the pushable-box branch.
- `on_key` → `world_ref.emit_sound("key")` when a key is newly collected.
- `on_goal` → `world_ref.emit_sound("fanfare")` (on `complete_level`).

### 5. `src/blueball/scenes/play.py`
- Construct a `SoundManager` in `PlayScene.__init__`.
- In `update`, each frame: drain `self.world.sound_events` (play each, then
  clear the list). Placed alongside the existing particle-emission logic.
- `_reset()` clears any pending events so a respawn doesn't replay stale sounds.

### 6. `src/blueball/config.py`
- `AUDIO_ENABLED = True` — master switch. `SoundManager` also checks
  `os.environ.get("BLUEBALL_NO_AUDIO")`.

## Data flow

```
boost-pad contact → on_boost_pad → world.emit_sound("whoosh")
                                       → PlayScene drains → SoundManager.play("whoosh")
spring contact (player) → on_spring → "spring"
key contact (new)       → on_key    → "key"
goal contact            → on_goal   → "fanfare"
```

## Testing (headless; `SDL_AUDIODRIVER=dummy`)

- `gen_sfx.py` produces four readable WAV files (assert each exists, is a valid
  WAV with >0 frames). Run in the test (writes to a tmp dir) to verify the
  generator, independent of the committed assets.
- `World.emit_sound` appends to `sound_events`; fresh World has an empty queue.
- Collision handlers emit the right name: drive a real World with a player and
  each entity (boost pad / spring / key / goal), step until contact, assert the
  expected string is in `world.sound_events`.
- `SoundManager`: with the dummy audio driver, constructs and `play("whoosh")`
  without error; with `BLUEBALL_NO_AUDIO=1`, loads nothing and `play` is a no-op
  (`enabled` is False).
- `PlayScene` drains the queue: after a frame in which an event was emitted,
  `world.sound_events` is empty (consumed).
- Determinism guard: a headless trainer evaluation is unaffected (existing
  trainer tests still pass — emit only appends to a list).

## Out of scope (v1)

- Background music, per-sound volume mixing, spatial/stereo audio.
- In-game mute key (config flag + env var only).
- Re-whoosh on a boost re-triggered mid-boost (begin-handler fires per contact;
  acceptable).
```
