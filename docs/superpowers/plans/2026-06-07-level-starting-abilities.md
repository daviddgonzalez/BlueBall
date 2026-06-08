# Level-Declared Starting Abilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the training player the abilities a real player has when arriving at a level — declared by the level itself (`maze` → double jump) — so the GA trains on the same game players actually play, then retrain the box-lava specialist to confirm it now reaches the goal.

**Architecture:** A new defaulted `LevelMeta.starting_abilities` field parsed from an optional `"starting_abilities"` array in the level JSON. Every training player-construction (curriculum evaluator, static `trainer.evaluate`, the cosmetic `TrainScene` players) grants `meta.starting_abilities`; the real game (`PlayScene`) unions it with save-unlocked abilities. Default-empty frozenset keeps every existing path and all 420 tests byte-identical; the abilities bit is already an FTNN input, so `GENOME_SIZE` is unchanged and saved genomes stay compatible.

**Tech Stack:** Python 3.12, NumPy, pymunk, pygame-ce (headless via `SDL_VIDEODRIVER=dummy`), pytest. The repo's editable install points at the *main* repo, so this worktree's code is run with `PYTHONPATH=<worktree>/src`.

---

## Grounding facts (verified by reading the branch, 2026-06-07)

- **Run tests with the worktree on the path.** The venv (`/home/ddgg0/projects/BlueBall/.venv`) has an editable install pointing at the **main** repo's `src`. Run every command as:
  `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest …` from the worktree root. Baseline: **420 tests, all green.**
- **`LevelMeta`** (`src/blueball/levels/loader.py:16`): frozen dataclass `name, spawn, background, ground, total_width`. `load_level` builds `starting_abilities` nowhere yet. Two *other* `LevelMeta(...)` constructions (keyword-only, omit the new field → default kicks in): `scenes/play.py` `_init_streaming_level` and `scenes/train.py` `_start_generation` Infinite branch.
- **`Ability`** (`src/blueball/abilities.py:13`): `DOUBLE_JUMP = "double_jump"`. String→enum is `Ability("double_jump")`; an unknown string raises `ValueError`. The `ability_pickup` chunk already uses this.
- **`Player`** (`src/blueball/entities/player.py:55`): `Player(agent, spawn_xy=..., abilities: set[Ability] | None = None)`. It stores `self.abilities = abilities if abilities is not None else set()` and **shares it by reference** with `JumpController`; `unlock()` does `self.abilities.add(...)`. So callers MUST pass a **mutable `set`**, not a `frozenset` (a frozenset would break `unlock()`). `_max_air_jumps()` returns `1` iff `Ability.DOUBLE_JUMP` present, else `0`.
- **Training player constructions** (all currently grant NO abilities):
  - `make_curriculum_player` (`curriculum.py:158`) → used by `evaluate_curriculum`; this is the box-lava specialist's path.
  - `trainer.evaluate` static path (`trainer.py:93`) — authoritative level-training fitness. `Player` is imported as `from ..entities.player import Player` (`trainer.py:39`), so tests can `monkeypatch.setattr(trainer, "Player", spy)`.
  - `trainer.evaluate_infinite` (`trainer.py:130`) — **NOT changed** (Infinite Run, no level meta).
  - `scenes/train.py` `_start_generation` (`train.py:133`) — cosmetic display players (the real eval goes through `trainer.evaluate` via `_make_args`). `__init__` builds `_players`, so a constructed `TrainScene` already has them.
- **`PlayScene._reset`** (`scenes/play.py:60`): loads `unlocked` from `save.load()`, then `Player(..., abilities=unlocked)`. The union must run for both streaming and non-streaming (both set `self.level_meta`).
- **Test scaffolding that already exists:**
  - `tests/test_level_loader.py`: imports `json`, `pytest`, `World`, `load_level`; pattern is inline level dict → `load_level(path, World())` → assert on `meta`.
  - `tests/test_ai_curriculum.py`: imports `numpy as np`; uses `resolve_level_paths(["maze"])[0]`, `random_genome(np.random.default_rng(...))`.
  - `tests/test_ai_smoke.py`: has `_level_path()` (→ **tutorial_hill.json**, which has NO `starting_abilities`), `headless_pygame` fixture, `_SyncPool`, and imports `numpy as np`. `from blueball.ai.trainer import evaluate` works.
  - `tests/test_play_scene.py`: has `headless_pygame` and `tmp_save` fixtures; `PlayScene(headless_pygame, level_path)` builds `scene.player`. Existing `test_play_scene_starts_with_no_abilities_when_save_missing` uses tutorial_hill (stays green — tutorial_hill declares no starting abilities).

## File Structure

- **Modify** `src/blueball/levels/loader.py` — import `Ability`; add `starting_abilities: frozenset[Ability] = frozenset()` to `LevelMeta`; parse it in `load_level` (Task 1).
- **Modify** `src/blueball/levels/maze.json` — add `"starting_abilities": ["double_jump"]` (Task 1).
- **Modify** `src/blueball/ai/curriculum.py` — `make_curriculum_player` gains an `abilities` param; `evaluate_curriculum` passes `meta.starting_abilities` (Task 2).
- **Modify** `src/blueball/ai/trainer.py` — static `evaluate` passes `set(meta.starting_abilities)` (Task 3).
- **Modify** `src/blueball/scenes/train.py` — cosmetic players get `set(self.level_meta.starting_abilities)` (Task 3).
- **Modify** `src/blueball/scenes/play.py` — union save-unlocked with `set(self.level_meta.starting_abilities)` (Task 4).
- **Tests:** `tests/test_level_loader.py` (Task 1), `tests/test_ai_curriculum.py` (Task 2), `tests/test_ai_smoke.py` (Task 3), `tests/test_play_scene.py` (Task 4).

`trainer.evaluate_infinite` and the GA operators/encoding/fitness terms are **NOT** touched. No new source files.

---

### Task 1: `LevelMeta.starting_abilities` + maze declaration

**Goal:** `LevelMeta` gains a defaulted `starting_abilities: frozenset[Ability]` parsed from an optional `"starting_abilities"` JSON array; `maze.json` declares `["double_jump"]`. Levels omitting the field default to an empty frozenset (back-compat).

**Files:**
- Modify: `src/blueball/levels/loader.py` (import + `LevelMeta` field + `load_level` parse)
- Modify: `src/blueball/levels/maze.json`
- Test: `tests/test_level_loader.py`

**Acceptance Criteria:**
- [ ] A level JSON without `starting_abilities` → `meta.starting_abilities == frozenset()`.
- [ ] A level JSON with `"starting_abilities": ["double_jump"]` → `meta.starting_abilities == frozenset({Ability.DOUBLE_JUMP})`.
- [ ] `load_level("maze.json")` → `Ability.DOUBLE_JUMP in meta.starting_abilities`.
- [ ] An unknown ability string raises `ValueError`.
- [ ] All pre-existing loader tests stay green.

**Verify:** `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_level_loader.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_level_loader.py`:

```python
def test_starting_abilities_defaults_empty(tmp_path):
    level = {
        "name": "NoAbilities", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "lvl.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.starting_abilities == frozenset()


def test_starting_abilities_parsed_from_level(tmp_path):
    from blueball.abilities import Ability
    level = {
        "name": "WithDJ", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "starting_abilities": ["double_jump"],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "dj.json"
    path.write_text(json.dumps(level))
    meta = load_level(path, World())
    assert meta.starting_abilities == frozenset({Ability.DOUBLE_JUMP})


def test_maze_declares_double_jump_starting_ability():
    from pathlib import Path
    import blueball
    from blueball.abilities import Ability
    maze = Path(blueball.__file__).parent / "levels" / "maze.json"
    meta = load_level(maze, World())
    assert Ability.DOUBLE_JUMP in meta.starting_abilities


def test_unknown_starting_ability_raises(tmp_path):
    level = {
        "name": "BadAbility", "background": "#000000", "ground": "#000000",
        "spawn": [0, 0], "starting_abilities": ["triple_jump"],
        "chunks": [{"type": "flat", "width_tiles": 2}, {"type": "goal"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(level))
    with pytest.raises(ValueError):
        load_level(path, World())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_level_loader.py::test_starting_abilities_parsed_from_level -q`
Expected: FAIL — `AttributeError: 'LevelMeta' object has no attribute 'starting_abilities'`.

- [ ] **Step 3: Add the import + `LevelMeta` field** — `src/blueball/levels/loader.py`. Add the import near the other entity imports (after `from ..entities.lava import Lava`):

```python
from ..abilities import Ability
```

Add the field at the END of the `LevelMeta` dataclass (after `total_width: float`):

```python
@dataclass(frozen=True)
class LevelMeta:
    name: str
    spawn: tuple[float, float]
    background: tuple[int, int, int]
    ground: tuple[int, int, int]
    total_width: float
    starting_abilities: frozenset[Ability] = frozenset()
```

- [ ] **Step 4: Parse the field in `load_level`** — `src/blueball/levels/loader.py`. Just before the `spawn = tuple(data["spawn"])` line at the end, add:

```python
    # Abilities the player is assumed to already have on arrival (declared by
    # the level; e.g. double jump unlocked in an earlier level). Default empty.
    starting_abilities = frozenset(
        Ability(a) for a in data.get("starting_abilities", [])
    )
```

Then pass it into the return:

```python
    spawn = tuple(data["spawn"])
    return LevelMeta(
        name=data["name"],
        spawn=spawn,
        background=_hex_to_rgb(data["background"]),
        ground=_hex_to_rgb(data["ground"]),
        total_width=x,
        starting_abilities=starting_abilities,
    )
```

- [ ] **Step 5: Declare maze's starting ability** — `src/blueball/levels/maze.json`. Add the top-level key right after `"spawn"` (keep valid JSON — note the trailing comma):

```json
  "spawn": [80, 540],
  "starting_abilities": ["double_jump"],
```

- [ ] **Step 6: Run the loader tests to verify they pass**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_level_loader.py -q`
Expected: PASS (existing + 4 new).

- [ ] **Step 7: Run the full suite (back-compat guard)**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest -q`
Expected: PASS, count = 424 (420 + 4).

- [ ] **Step 8: Commit**

```bash
git add src/blueball/levels/loader.py src/blueball/levels/maze.json tests/test_level_loader.py
git commit -m "feat(levels): LevelMeta.starting_abilities; maze declares double_jump

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Grant starting abilities in curriculum training

**Goal:** `make_curriculum_player` gains a defaulted `abilities` param and grants it to the `Player`; `evaluate_curriculum` passes `meta.starting_abilities`. This is the box-lava specialist's path — after this, a maze curriculum player has double jump.

**Files:**
- Modify: `src/blueball/ai/curriculum.py:158` (`make_curriculum_player`) and the `make_curriculum_player(...)` call inside `evaluate_curriculum`
- Test: `tests/test_ai_curriculum.py`

**Acceptance Criteria:**
- [ ] `make_curriculum_player(world, g, spawn, keys)` with no `abilities` → `player.abilities == set()` (back-compat).
- [ ] `make_curriculum_player(..., frozenset({Ability.DOUBLE_JUMP}))` → `Ability.DOUBLE_JUMP in player.abilities` and `in player.jump_ctrl.abilities`.
- [ ] `evaluate_curriculum` on maze passes the level's `starting_abilities` (containing `DOUBLE_JUMP`) into `make_curriculum_player`.
- [ ] The player's `abilities` is a mutable `set` (so `unlock()` still works), not a frozenset.

**Verify:** `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_curriculum.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_curriculum.py`:

```python
def test_make_curriculum_player_default_no_abilities():
    from blueball.world import World
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.ai.curriculum import make_curriculum_player
    from blueball.ai.genome import random_genome
    from blueball.ai.episodes import resolve_level_paths
    path = resolve_level_paths(["maze"])[0]
    world = World(seed=1)
    register_collisions(world.space, world_ref=world)
    load_level(path, world)
    g = random_genome(np.random.default_rng(0))
    p = make_curriculum_player(world, g, (80.0, 540.0), 0)
    assert p.abilities == set()


def test_make_curriculum_player_grants_abilities():
    from blueball.world import World
    from blueball.collision import register as register_collisions
    from blueball.levels.loader import load_level
    from blueball.ai.curriculum import make_curriculum_player
    from blueball.ai.genome import random_genome
    from blueball.ai.episodes import resolve_level_paths
    from blueball.abilities import Ability
    path = resolve_level_paths(["maze"])[0]
    world = World(seed=1)
    register_collisions(world.space, world_ref=world)
    load_level(path, world)
    g = random_genome(np.random.default_rng(0))
    p = make_curriculum_player(world, g, (80.0, 540.0), 0,
                               frozenset({Ability.DOUBLE_JUMP}))
    assert Ability.DOUBLE_JUMP in p.abilities
    assert Ability.DOUBLE_JUMP in p.jump_ctrl.abilities
    assert isinstance(p.abilities, set)  # mutable, so unlock() still works


def test_evaluate_curriculum_grants_level_starting_abilities(monkeypatch):
    import blueball.ai.curriculum as cur
    from blueball.ai.genome import random_genome
    from blueball.ai.episodes import resolve_level_paths
    from blueball.abilities import Ability
    captured = {}
    real = cur.make_curriculum_player

    def spy(world, genome, spawn_xy, granted_keys, abilities=frozenset()):
        captured["abilities"] = abilities
        return real(world, genome, spawn_xy, granted_keys, abilities)

    monkeypatch.setattr(cur, "make_curriculum_player", spy)
    path = resolve_level_paths(["maze"])[0]
    g = random_genome(np.random.default_rng(0))
    cur.evaluate_curriculum((0, g, 1, path, 5, (3250.0, 540.0), 0b11))
    assert Ability.DOUBLE_JUMP in captured["abilities"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_curriculum.py::test_make_curriculum_player_grants_abilities -q`
Expected: FAIL — `TypeError: make_curriculum_player() takes 4 positional arguments but 5 were given`.

- [ ] **Step 3: Add the `abilities` param to `make_curriculum_player`** — `src/blueball/ai/curriculum.py`. Replace the function with:

```python
def make_curriculum_player(world, genome, spawn_xy, granted_keys: int,
                           abilities=frozenset()) -> Player:
    """Spawn a Player at `spawn_xy`, add it to `world`, and grant `granted_keys`
    (OR'd into keys_held) so doors behind the spawn are openable. `abilities`
    are the level's assumed starting abilities (e.g. double jump). Passed as a
    fresh mutable set so the player's shared abilities set stays mutable. Shared
    by the evaluator and tests."""
    player = Player(agent=FTNNAgent(genome),
                    spawn_xy=(float(spawn_xy[0]), float(spawn_xy[1])),
                    abilities=set(abilities))
    world.add_entity(player)
    player.keys_held |= int(granted_keys)
    return player
```

- [ ] **Step 4: Pass the level's abilities from `evaluate_curriculum`** — `src/blueball/ai/curriculum.py`. In `evaluate_curriculum`, replace the existing `player = make_curriculum_player(world, genome, spawn_xy, granted_keys)` line with:

```python
    player = make_curriculum_player(world, genome, spawn_xy, granted_keys,
                                    meta.starting_abilities)
```

- [ ] **Step 5: Run the curriculum tests to verify they pass**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_curriculum.py -q`
Expected: PASS (existing + 3 new).

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest -q`
Expected: PASS, count = 427 (424 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/curriculum.py tests/test_ai_curriculum.py
git commit -m "feat(curriculum): grant level starting_abilities to curriculum player

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Grant starting abilities in static `trainer.evaluate` + `TrainScene`

**Goal:** The authoritative static-level training path (`trainer.evaluate`) and the cosmetic `TrainScene` display players both grant `set(meta.starting_abilities)`, so maze level-training trains (and shows) a double-jumping agent. `evaluate_infinite` is left unchanged.

**Files:**
- Modify: `src/blueball/ai/trainer.py:93` (static `evaluate` player spawn)
- Modify: `src/blueball/scenes/train.py:133` (cosmetic display players)
- Test: `tests/test_ai_smoke.py`

**Acceptance Criteria:**
- [ ] `trainer.evaluate` on maze constructs its `Player` with `abilities` containing `Ability.DOUBLE_JUMP`.
- [ ] `trainer.evaluate` on a level without starting abilities (tutorial_hill) constructs its `Player` with `abilities == set()` (back-compat).
- [ ] A `TrainScene` built on maze has every `_players[i].abilities` containing `Ability.DOUBLE_JUMP`.
- [ ] `evaluate_infinite` is unchanged (no abilities granted).

**Verify:** `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ai_smoke.py`:

```python
def test_evaluate_grants_level_starting_abilities_on_maze(monkeypatch):
    """trainer.evaluate spawns the player with the level's declared starting
    abilities — maze grants double jump."""
    import blueball.ai.trainer as trainer
    from blueball.ai.genome import random_genome
    from blueball.abilities import Ability
    from pathlib import Path
    import blueball
    maze = Path(blueball.__file__).parent / "levels" / "maze.json"
    captured = {}
    RealPlayer = trainer.Player

    def spy(*args, **kwargs):
        captured["abilities"] = kwargs.get("abilities")
        return RealPlayer(*args, **kwargs)

    monkeypatch.setattr(trainer, "Player", spy)
    g = random_genome(np.random.default_rng(0))
    trainer.evaluate((0, g, 1, maze, 5))
    assert Ability.DOUBLE_JUMP in captured["abilities"]


def test_evaluate_no_abilities_on_level_without_starting_abilities(monkeypatch):
    """tutorial_hill declares no starting_abilities, so the evaluate player is
    ability-less — back-compat."""
    from blueball import config
    import blueball.ai.trainer as trainer
    from blueball.ai.genome import random_genome
    captured = {}
    RealPlayer = trainer.Player

    def spy(*args, **kwargs):
        captured["abilities"] = kwargs.get("abilities")
        return RealPlayer(*args, **kwargs)

    monkeypatch.setattr(trainer, "Player", spy)
    g = random_genome(np.random.default_rng(0))
    trainer.evaluate((0, g, config.DEFAULT_SEED, _level_path(), 5))
    assert captured["abilities"] == set()


def test_train_scene_players_get_level_abilities_on_maze(headless_pygame):
    """The cosmetic display players match what's trained: on maze they carry
    the level's double jump."""
    from blueball.scenes.train import TrainScene
    from blueball.abilities import Ability
    from pathlib import Path
    import blueball
    maze = Path(blueball.__file__).parent / "levels" / "maze.json"
    scene = TrainScene(headless_pygame, level_path=maze, pop_size=4,
                       n_visible=2, generations=2, max_steps=20, pool=_SyncPool())
    assert len(scene._players) == 2
    for p in scene._players:
        assert Ability.DOUBLE_JUMP in p.abilities
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py::test_evaluate_grants_level_starting_abilities_on_maze -q`
Expected: FAIL — `captured["abilities"]` is `None` (the current `Player(...)` passes no `abilities` kwarg).

- [ ] **Step 3: Grant abilities in static `trainer.evaluate`** — `src/blueball/ai/trainer.py`. In `evaluate` (NOT `evaluate_infinite`), replace:

```python
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y))
```

with:

```python
    player = Player(agent=FTNNAgent(genome), spawn_xy=(spawn_x, spawn_y),
                    abilities=set(meta.starting_abilities))
```

- [ ] **Step 4: Grant abilities to the cosmetic `TrainScene` players** — `src/blueball/scenes/train.py`, in `_start_generation`. Replace:

```python
            p = Player(agent=FTNNAgent(self.population[i]), spawn_xy=self._spawn_xy)
```

with:

```python
            p = Player(agent=FTNNAgent(self.population[i]), spawn_xy=self._spawn_xy,
                       abilities=set(self.level_meta.starting_abilities))
```

(`self.level_meta` is set just above in both the level and Infinite branches; the Infinite `LevelMeta` has the default empty `starting_abilities`, so Infinite Run is unchanged.)

- [ ] **Step 5: Run the smoke tests to verify they pass**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -q`
Expected: PASS (existing + 3 new).

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest -q`
Expected: PASS, count = 430 (427 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/blueball/ai/trainer.py src/blueball/scenes/train.py tests/test_ai_smoke.py
git commit -m "feat(train): grant level starting_abilities in evaluate + TrainScene

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Real-game parity — `PlayScene` unions level + save abilities

**Goal:** `PlayScene` constructs the player with `save-unlocked ∪ level starting_abilities`, so loading a level directly (even with an empty save) honors the level's declared abilities. Maze player has double jump regardless of save state.

**Files:**
- Modify: `src/blueball/scenes/play.py` (`_reset`, the `Player(...)` construction)
- Test: `tests/test_play_scene.py`

**Acceptance Criteria:**
- [ ] Loading maze with an empty save → `scene.player.abilities` contains `Ability.DOUBLE_JUMP`.
- [ ] Loading maze with a save unlocking double jump → still contains `Ability.DOUBLE_JUMP` (union, no duplication error).
- [ ] Loading tutorial_hill with an empty save → `scene.player.abilities == set()` (existing test stays green; tutorial_hill declares no starting abilities).

**Verify:** `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_play_scene.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `tests/test_play_scene.py`:

```python
def _maze_path() -> Path:
    return Path(blueball.__file__).parent / "levels" / "maze.json"


def test_play_scene_maze_grants_double_jump_from_level(headless_pygame, tmp_save):
    """maze declares starting_abilities=[double_jump]; even with an empty save
    the player arrives with double jump (level union)."""
    _path, _save_mod = tmp_save
    scene = PlayScene(headless_pygame, _maze_path())
    assert Ability.DOUBLE_JUMP in scene.player.abilities


def test_play_scene_unions_save_and_level_abilities(headless_pygame, tmp_save):
    """Save-unlocked abilities union with the level's starting abilities."""
    path, _save_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["double_jump"]}))
    scene = PlayScene(headless_pygame, _maze_path())
    assert Ability.DOUBLE_JUMP in scene.player.abilities
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_play_scene.py::test_play_scene_maze_grants_double_jump_from_level -q`
Expected: FAIL — empty save → `scene.player.abilities` is empty, so `DOUBLE_JUMP` is absent.

- [ ] **Step 3: Union level abilities in `PlayScene._reset`** — `src/blueball/scenes/play.py`. Replace:

```python
        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=unlocked,
        )
```

with:

```python
        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        # The level may declare abilities the player is assumed to arrive with
        # (e.g. double jump unlocked earlier); union so a direct load is fair.
        abilities = unlocked | set(self.level_meta.starting_abilities)
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=abilities,
        )
```

- [ ] **Step 4: Run the play-scene tests to verify they pass**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_play_scene.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest -q`
Expected: PASS, count = 432 (430 + 2).

- [ ] **Step 6: Commit**

```bash
git add src/blueball/scenes/play.py tests/test_play_scene.py
git commit -m "feat(play): union level starting_abilities with save-unlocked abilities

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Retrain the box-lava specialist & observe the verdict (USER-ORDERED GATE)

**Goal:** With the double-jump fix in place, retrain the box-lava specialist and observe whether it now reaches the goal — the user-ordered "retrain first" check that decides whether the reserved cradle + on-box-jump reward are still needed.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation ("grant double jump, retrain first… add cradle only if it still struggles"). It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after the retrain has actually run end-to-end and the verdict line + `run.json` `cracked` have been captured.

**Files:** none (empirical run; no code/tests).

**Acceptance Criteria:**
- [ ] The retrain runs to completion (exit 0) on the box-lava specialist at spec defaults (pop 80 × gens 200, world=1, ga=0).
- [ ] The stdout verdict line `Verdict @ 'box_lava': reached_goal=<bool> fitness=<float>` is captured.
- [ ] The new run's `run.json` `curriculum.cracked` value is recorded.
- [ ] Result compared against the single-jump baseline (`reached_goal=False`, fitness 1020.5, `cracked=false`). **Decision:** if `reached_goal=True` → double jump cracked box-lava, feature done; if still `False` → escalate to the reserved cradle + on-box-jump reward (separate spec/plan).

**Verify:** from the worktree root —
`PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python train_maze_curriculum.py --box-lava`
→ exits 0; capture the `Verdict @ 'box_lava': reached_goal=…` line and the newest `genomes/mazeboxlavacurr_w1_*/run.json` `cracked` field.

**Steps:**

- [ ] **Step 1: Confirm the full suite is green** (all of Tasks 1–4 committed)

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest -q`
Expected: PASS, count = 432.

- [ ] **Step 2: Run the box-lava specialist retrain** (≈5–6 min; deterministic given seeds)

Run: `PYTHONPATH=$PWD/src /home/ddgg0/projects/BlueBall/.venv/bin/python train_maze_curriculum.py --box-lava`
Expected: exit 0; a `genomes/mazeboxlavacurr_w1_<ts>/` dir with `final_best.npy` + `run.json`.

- [ ] **Step 3: Capture & compare the verdict**

Read the stdout `Verdict @ 'box_lava': reached_goal=… fitness=…` line and `run.json`'s `curriculum.cracked`. Compare to baseline (False / 1020.5 / cracked=false). Report the result and the decision (done vs escalate to cradle + on-box reward).

---

## Self-Review

**Spec coverage** (`docs/superpowers/specs/2026-06-07-level-starting-abilities-design.md`):
- Goal: level-declared starting abilities → **Task 1** (`LevelMeta.starting_abilities` + maze.json).
- "All maze training" grant → **Task 2** (curriculum/specialist), **Task 3** (static `trainer.evaluate` + cosmetic `TrainScene`). `evaluate_infinite` intentionally excluded (no level meta) — stated in both spec and Task 3.
- Real-game parity (play.py union) → **Task 4**.
- Encoding stability (GENOME_SIZE unchanged) → no encoding touched; abilities bit already an FTNN input (noted in header).
- Back-compat (default-empty frozenset) → guarded by the default-path tests in Tasks 1–4 and the full-suite step ending every task.
- Empirical follow-up (retrain, observe `reached_goal`) → **Task 5** (user-ordered gate).

**Placeholder scan:** none — every step shows exact code/commands.

**Type consistency:** `starting_abilities` is `frozenset[Ability]` on `LevelMeta` (Task 1) and is converted to a mutable `set(...)` at every `Player(abilities=…)` construction (Tasks 2–4) per the Player-shares-by-reference / `unlock()` constraint in the grounding facts. `make_curriculum_player`'s new param is named `abilities` consistently in the impl and the `evaluate_curriculum` spy test. `meta.starting_abilities` / `self.level_meta.starting_abilities` are the only read sites.
