# AI / GA Scaffolding — Design Spec

**Date:** 2026-05-26
**Status:** Approved for planning
**Phase:** 4 prep, slice 1 (AI/GA infrastructure)

## Summary

Land the scaffolding for the Genetic Algorithm (GA) training pipeline so a fixed-topology neural network (FTNN) population can be evolved end-to-end on a single level. Five pieces:

1. A new `ai/` package with the FTNN, the genome representation, the GA operators (mutation, crossover, tournament selection), an `Observation` → input-vector adapter, the fitness function, and the generation-loop trainer.
2. An `FTNNAgent` subclass of `Agent` (in the existing `agent.py`) that wraps a genome and emits an `Action` per tick.
3. Minimal additive touches to `Player` (shared collision-filter group; per-player `reached_goal` flag) so N agents can coexist non-interactively in one `World` and the trainer can score them individually.
4. A `TrainScene` that runs the trainer in-process, renders the visible population live on the tutorial level, and lets the developer pan/zoom a free camera while training proceeds.
5. A smoke test pinning the end-to-end loop: a 5-generation run on a tiny population finishes without crashing and produces well-formed fitnesses.

This is the prep slice. Real training and learning live in a follow-up that lands after the level-design branch ships its `Observation` enrichment (raycasts, populated `nearest_collectible`).

## Motivation

The v1 design spec already specifies the FTNN topology, the GA operator menu, the fitness shape, the parallelism strategy (`multiprocessing.Pool`), and the explicit user request for `TrainScene` with a free camera. Phase 4 cashes in those forward-looking sections. The scaffolding is the load-bearing piece: once it exists and the smoke test pins it, the level-design branch's raycast wiring is a near-zero-risk follow-up that turns on real learning. Holding off on the scaffolding until raycasts land would serialize work; landing it now in parallel halves the wall-clock to first trained agent.

The other parallel session owns `Observation` enrichment. This slice **must not** modify the `Observation` dataclass, add raycast computation, modify level JSON, or add chunk types — those belong to the other branch.

## Behavior

| When | What happens |
|---|---|
| `FTNNAgent(genome).act(observation)` is called | The observation's relevant fields are concatenated to a 14-float input vector, run through a two-layer FTNN (14 → 12 tanh → 6), and the output's `argmax` selects an `Action`. |
| Two or more `Player` entities exist in the same `World` | They never collide with each other (shared non-zero pymunk `ShapeFilter.group`). They still collide with ground, spikes, patrollers, collectibles, ability pickups, boost pads, and the goal. |
| A `Player` overlaps the goal sensor in TrainScene | That player's `reached_goal` becomes `True`. The shared `world.level_complete` flag still gets set too (PlayScene depends on it). |
| `trainer.evaluate(args)` runs in a worker | Builds a fresh headless `World`, registers collisions, loads the level, spawns one `Player(FTNNAgent(genome))` at the level's spawn, steps physics at `PHYS_DT` up to `max_steps` (or until the player dies / reaches the goal), and returns `(idx, fitness)`. |
| `trainer.train(...)` runs | Initializes a population of random genomes from a `ga_seed`, evaluates each via `map_fn` (default `map`, opt-in to `multiprocessing.Pool(...).imap`), records per-generation stats, breeds the next generation with elitism + tournament selection + crossover + mutation, and returns a `TrainingResult` containing per-gen history and the best genome. |
| `TrainScene` is entered | One shared headless `World` is built; N visible players are spawned with distinct FTNN genomes (all sharing the player collision group). Each physics tick advances the shared `World`. At gen boundary, the world is rebuilt with the next generation. The user can pan with arrow keys and zoom with `+` / `-`. A HUD strip shows `gen N | best fit X | live M/N`. |

## Components

### `src/blueball/ai/` (new package, 7 files)

Top-level boundary: this package is the only place that knows about genomes, FTNN topology, GA operators, the fitness shape, and the trainer loop. It depends on `World`, `Player`, `Action`, and `Observation` — but **not** the other way around. Nothing outside `ai/` imports anything from `ai/` except the scene/CLI that drives training.

```
src/blueball/ai/
├── __init__.py
├── ftnn.py
├── genome.py
├── ga.py
├── observation.py
├── fitness.py
└── trainer.py
```

#### `ai/ftnn.py`

```python
FTNN_INPUTS = 14
FTNN_HIDDEN = 12
FTNN_OUTPUTS = 6   # one per Action

class FTNN:
    """A 14 → 12 tanh → 6 fully-connected network. Pure numpy."""
    def __init__(self, genome: np.ndarray) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...
```

- The genome layout is `[W1 (14*12) | b1 (12) | W2 (12*6) | b2 (6)]` = `14*12 + 12 + 12*6 + 6 = 282` float32s. `GENOME_SIZE = 282` is exported from `genome.py` (derived from the constants here so changing topology updates both).
- `FTNN.__init__` slices the flat genome into its weight/bias arrays once; `forward` does two matmuls + one `tanh`.
- No batch dimension. One observation per call.

#### `ai/genome.py`

```python
GENOME_SIZE = FTNN_INPUTS * FTNN_HIDDEN + FTNN_HIDDEN + FTNN_HIDDEN * FTNN_OUTPUTS + FTNN_OUTPUTS  # 282

def random_genome(rng: np.random.Generator) -> np.ndarray:
    """Sample a fresh genome from N(0, 1). Returns float32 ndarray of shape (GENOME_SIZE,)."""
```

Trivial. Single helper to keep population init out of `ga.py`.

#### `ai/ga.py`

```python
def mutate(genome: np.ndarray, rng: np.random.Generator, *,
           rate: float = 0.1, sigma: float = 0.1) -> np.ndarray: ...
def crossover(parent_a: np.ndarray, parent_b: np.ndarray,
              rng: np.random.Generator) -> np.ndarray: ...
def tournament_select(fitnesses: np.ndarray, rng: np.random.Generator,
                      k: int = 4) -> tuple[int, int]: ...
def breed(population: list[np.ndarray], fitnesses: np.ndarray,
          rng: np.random.Generator, *, elitism: int = 1) -> list[np.ndarray]: ...
```

- `mutate` returns a *new* array. For each weight, with probability `rate`, add `rng.normal(0, sigma)`.
- `crossover` is per-gene uniform: each gene from A or B with 50/50.
- `tournament_select` samples `k` indices uniformly, returns the two highest-fitness indices.
- `breed` produces a next-generation list of size `len(population)`. The top `elitism` genomes (default 1) pass through unchanged; the rest are produced by `crossover(parents) → mutate(child)`.

All operators are pure functions of their arguments. No module-level state.

#### `ai/observation.py`

```python
RAY_COUNT = 8

def observation_to_inputs(obs: Observation) -> np.ndarray:
    """Pack an Observation into the 14-float vector the FTNN expects.

    Layout (indices):
      0-7:  obs.rays                                (8 floats; currently all 0 — populated when level-design branch ships)
      8-9:  obs.vel[0], obs.vel[1]                  (2 floats)
       10:  obs.ang_vel                             (1 float)
       11:  1.0 if obs.grounded else 0.0            (1 float)
      12-13: nearest_collectible offset             (2 floats; (0, 0) when obs.nearest_collectible is None)

    DEPENDENCY: if the level-design branch's Observation enrichment changes
    rays.shape from (8,) to a different size, RAY_COUNT and FTNN_INPUTS in
    ai/ftnn.py must be updated in lockstep. Asserted below so the failure
    mode is a clean message, not a numpy broadcast error.
    """
    assert obs.rays.shape == (RAY_COUNT,), (
        f"observation_to_inputs expects rays of shape ({RAY_COUNT},), "
        f"got {obs.rays.shape} — update RAY_COUNT and FTNN_INPUTS together."
    )
    ...
```

The whole point of this helper is to seal off the FTNN from `Observation`'s shape. When raycasts get populated, **no code in `ai/` changes** beyond possibly bumping `RAY_COUNT`.

#### `ai/fitness.py`

```python
@dataclass(frozen=True)
class FitnessInputs:
    progress_x: float           # player.body.position.x - spawn_x
    collectibles: int           # player.collectibles_collected
    reached_goal: bool          # player.reached_goal
    died: bool                  # player.dead
    steps_taken: int            # the loop counter from evaluate()

def fitness(inputs: FitnessInputs) -> float:
    """Matches the v1 spec's starting fitness function."""
    return (
        inputs.progress_x
        + 50.0  * inputs.collectibles
        + 200.0 * (1.0 if inputs.reached_goal else 0.0)
        -   0.01 * inputs.steps_taken
        - 100.0 * (1.0 if inputs.died else 0.0)
    )
```

A snapshot dataclass (no live entity references) so the fitness function is pure and testable. The trainer extracts these from the `Player` at end-of-evaluation; the function itself doesn't import `Player`.

#### `ai/trainer.py`

```python
@dataclass(frozen=True)
class TrainingResult:
    history: list[dict]                # per-gen: {"gen": int, "best": float, "mean": float, "min": float}
    best_genome: np.ndarray            # shape (GENOME_SIZE,)
    final_population: list[np.ndarray] # for follow-up runs / TrainScene re-entry

def evaluate(args: tuple) -> tuple[int, float]:
    """Worker function. Args is (idx, genome, world_seed, level_path, max_steps).
    Builds a fresh headless World inside the worker (so it's picklable-input,
    picklable-output across multiprocessing.Pool). Returns (idx, fitness)."""

def train(*,
          pop_size: int,
          generations: int,
          level_path: Path,
          ga_seed: int = 0,
          world_seed: int = DEFAULT_SEED,
          max_steps: int = MAX_STEPS,
          map_fn: Callable = map,
          on_generation: Callable[[int, np.ndarray, list[np.ndarray]], None] | None = None,
          ) -> TrainingResult: ...
```

- `map_fn` defaults to the builtin `map` (serial, in-process). Real training callers pass `multiprocessing.Pool(...).imap`. This default keeps the smoke test bulletproof and avoids fork-on-import surprises (e.g. `agent.py` transitively imports `pygame`).
- `ga_seed` controls all GA randomness (population init, mutation, crossover, tournament). `world_seed` controls the physics world. Separate knobs.
- `on_generation(gen, best_genome_so_far, population)` is the callback `TrainScene` uses to refresh its HUD and rebuild the visible world between generations. `None` in the headless path.

### `src/blueball/agent.py` (modified: one new class)

Add below `HumanAgent`:

```python
class FTNNAgent(Agent):
    def __init__(self, genome: np.ndarray) -> None:
        from .ai.ftnn import FTNN
        from .ai.observation import observation_to_inputs
        self._net = FTNN(genome)
        self._to_inputs = observation_to_inputs

    def act(self, observation: Observation) -> Action:
        x = self._to_inputs(observation)
        y = self._net.forward(x)
        return Action(int(np.argmax(y)))
```

Lazy import inside `__init__` so importing `agent` doesn't pull in the `ai/` package transitively for code paths (PlayScene, tests) that don't need it.

**Observation is not touched.** This is the entire diff to `agent.py`.

### `src/blueball/entities/player.py` (modified)

Two additive touches:

1. Always set a shared collision-filter group on the player's shape so multiple `Player` entities in one `World` don't collide with each other:

   ```python
   self.shape.filter = pymunk.ShapeFilter(group=PLAYER_GROUP)
   ```

   `PLAYER_GROUP = 99` lives in `collision.py` alongside the existing `CT_*` constants. PlayScene is unaffected — with one player, "same group" is trivially satisfied.

2. Add a per-player goal-reached flag:

   ```python
   self.reached_goal: bool = False
   ```

   Initialized in `__init__`; mutated by the goal collision handler (next section).

Motion and physics tuning code is untouched.

### `src/blueball/collision.py` (modified)

Two changes:

1. Add `PLAYER_GROUP = 99` alongside the existing `CT_*` constants.
2. In `on_goal(arbiter, ...)`, set `player.reached_goal = True` on whichever Player entity is in the arbiter (using the existing `_find_player_entity` helper), in addition to the current `world_ref.complete_level()` call. PlayScene's level-end flow is unchanged because `world.level_complete` still flips.

### `src/blueball/camera.py` (modified)

Add a uniform-scale field to `Camera` and a `FreeCamera` subclass:

```python
class Camera:
    def __init__(self, viewport_w, viewport_h) -> None:
        ...
        self.scale: float = 1.0

    def world_to_screen(self, world_xy):
        wx, wy = world_xy
        cx, cy = self.position
        s = self.scale
        return ((wx - cx) * s + self.viewport_w / 2,
                (wy - cy) * s + self.viewport_h / 2)


class FreeCamera(Camera):
    """Pan with arrow keys; zoom with +/-. Used by TrainScene."""
    PAN_SPEED   = 500.0   # px/s in world units
    ZOOM_STEP   = 1.1     # multiplicative per +/- press
    ZOOM_MIN    = 0.1
    ZOOM_MAX    = 4.0

    def handle_events(self, events) -> None: ...   # consume zoom keypresses
    def update(self, keys_pressed, dt) -> None: ... # apply pan from held arrow keys
```

`FollowCamera` already works correctly with `scale = 1.0` (no behavior change).

### `src/blueball/render/renderer.py` (not modified)

The renderer's `_w2s` already routes through `camera.world_to_screen`, which under the new `camera.scale` multiplies positions by the zoom factor. So zoom flows through for free without any renderer-side edit. **Primitive sizes (ball radius, ground line thickness, spike heights, etc.) deliberately stay in screen-space** — they don't multiply by `camera.scale`. The visual result at zoom = 0.5 is a half-sized world layout populated by correctly-sized markers. Good enough for a developer debug overview; a future polish pass can scale primitives if we ever want a beauty zoom.

### `src/blueball/scenes/train.py` (new)

```python
class TrainScene(Scene):
    """Run the GA trainer in-process and render the visible population live.

    Owns one shared World. N visible players are spawned at the level's spawn
    with one FTNN genome each, all sharing the player collision group (so
    they're non-interactive with each other but still collide with the world).
    """

    def __init__(self, screen, level_path, *,
                 pop_size: int = config.TRAIN_POP_SIZE,
                 n_visible: int = 16,
                 generations: int = config.TRAIN_GENERATIONS,
                 ga_seed: int = 0,
                 max_steps: int = config.MAX_STEPS) -> None: ...

    def handle_events(self, events): ...   # consume QUIT/ESC; pass others to FreeCamera
    def update(self, frame_dt): ...        # step world; advance generation when due
    def draw(self): ...                    # background, ground, entities, HUD
```

Behavior:

- **Generation lifecycle.** Each generation runs for up to `max_steps` ticks. The scene steps the shared `World` per frame using the existing fixed-substep accumulator (so visualization runs at real-time pace). When `max_steps` is reached or all visible players are dead-or-finished, fitnesses are computed, the trainer breeds the next generation, the `World` is rebuilt (fresh `World` object + re-`load_level` + re-spawn N players with the new genomes), and the next gen starts.
- **N visible.** `n_visible` is clamped to `min(n_visible, pop_size)` at scene construction time. The full `pop_size` (default `TRAIN_POP_SIZE = 80`) is what gets bred each generation; the first `n_visible` indices of the current population are the ones drawn. Generation 0 has no fitness yet so "first" is just population order. (Could be refined later to "best of previous gen drawn at the front" once the trainer is producing meaningful fitnesses.)
- **HUD.** Top-left text strip rendered with `pygame.font.Font(None, 20)`: `gen N | best X.X | mean Y.Y | live M/N`.
- **Camera.** A `FreeCamera`. Arrow keys pan, `+`/`-` zoom.
- **Exit.** `ESC` returns `None` from `handle_events` to quit. `Q` is *not* bound (avoids stealing keys from a future QWERTY-typed level editor).

### `src/blueball/config.py` (modified — append new section)

```python
# AI / GA training
TRAIN_POP_SIZE     = 80       # spec default for real training
TRAIN_GENERATIONS  = 200      # spec default for real training
MAX_STEPS          = 3000     # ~25s of simulated time at PHYS_HZ=120
GA_MUTATION_RATE   = 0.1
GA_MUTATION_SIGMA  = 0.1
GA_TOURNAMENT_K    = 4
GA_ELITISM         = 1
```

`MAX_STEPS` is the per-evaluation timeout. Smoke test overrides these to small values per-call; nothing in the config needs a "smoke" variant.

### `main.py` (not modified in this slice)

`TrainScene` is reachable via a separate `train_main.py` entry script (also new) so this slice doesn't touch the live-game entry path:

```python
# train_main.py
from pathlib import Path
import pygame
from blueball import config
from blueball.scenes.train import TrainScene

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball — Train")
    level_path = Path(__file__).parent / "src" / "blueball" / "levels" / "tutorial_hill.json"
    scene = TrainScene(screen, level_path)
    clock = pygame.time.Clock()
    while scene is not None:
        events = pygame.event.get()
        scene = scene.handle_events(events)
        if scene is None:
            break
        frame_dt = clock.tick(config.TARGET_FPS) / 1000.0
        scene.update(frame_dt)
        scene.draw()
    pygame.quit()
```

Two entry scripts (`main.py` for play, `train_main.py` for train) is the simplest split until a menu scene exists.

## Testing

### `tests/test_ai_smoke.py` (new)

The pinning test for the whole slice. Crash + shape/range sanity, no improvement claim — that lands when raycasts ship.

| Test | What it asserts |
|---|---|
| `test_ftnn_forward_pass_shape` | `FTNN(random_genome(rng)).forward(zeros(14)).shape == (6,)` and dtype is float32. |
| `test_random_genome_shape_and_dtype` | `random_genome(rng).shape == (GENOME_SIZE,)` and dtype is float32. |
| `test_mutate_returns_new_array_same_shape` | `mutate(g, rng).shape == g.shape`, returns a different object, and at least one weight changed (with overwhelming probability at rate=1.0, sigma=1.0). |
| `test_crossover_inherits_from_both_parents` | All-zero parent and all-one parent → child has both 0s and 1s (deterministic ga_seed; assert the proportion is within (0.3, 0.7) for a long-enough genome). |
| `test_tournament_select_returns_two_distinct_top_indices` | Given a fitness array with a known top-2, `tournament_select` returns those indices when `k == pop_size`. |
| `test_observation_to_inputs_shape_and_layout` | Build an Observation manually (using the existing v1 fields), call the helper, assert `shape == (14,)`, dtype float32, and that each slot matches the input field (zeros for None collectible, 1.0/0.0 for grounded). |
| `test_observation_to_inputs_rejects_wrong_ray_count` | Pass an Observation with `rays.shape == (7,)`; assert AssertionError with the documented message. |
| `test_evaluate_runs_one_genome_to_completion` | Build a random genome, call `evaluate((0, genome, DEFAULT_SEED, level_path, 200))`, assert it returns `(0, float)` with a finite result. |
| `test_trainer_smoke_5gens_no_crash` | `train(pop_size=8, generations=5, level_path=tutorial_hill, max_steps=600, ga_seed=0)` returns a `TrainingResult` whose `history` has length 5, `best_genome.shape == (GENOME_SIZE,)`, every history entry's `best` / `mean` / `min` is finite, and `final_population` has 8 entries each with the right shape. No improvement assertion. |

Smoke-test budget: pop=8, gens=5, max_steps=600 → ≈ 24,000 physics steps + 24,000 FTNN forward passes. Empirically ~3–10 seconds depending on hardware. Acceptable for pytest.

### `tests/test_player.py` (modified)

| New test | What it asserts |
|---|---|
| `test_player_has_player_group_collision_filter` | A freshly-constructed Player's shape has `filter.group == collision.PLAYER_GROUP`. |
| `test_two_players_in_one_world_do_not_collide` | Build a World, register collisions, spawn two Players at the same position, step a few frames, assert neither has moved due to mutual collision (positions should be free-fall identical to a control single-player World). |
| `test_player_reached_goal_defaults_false` | `Player(...).reached_goal is False`. |

### `tests/test_collision.py` (modified)

| New test | What it asserts |
|---|---|
| `test_goal_handler_sets_player_reached_goal` | Place a Player overlapping a Goal sensor, step one frame, assert `player.reached_goal is True` and `world.level_complete is True` (both must keep working). |

### Out-of-scope tests deliberately not in this slice

- **Renderer tests for TrainScene HUD.** Visual output, hard to assert. We rely on running it.
- **Multi-process integration test for `train(map_fn=multiprocessing.Pool(...).imap)`.** Lands with the real-training follow-up once we know what we want to assert about it.
- **Learning assertions ("after N gens the best fitness is > X").** Meaningless until raycasts arrive — there's no signal to learn from.

## Branch and worktree

- Worktree path: `.worktrees/ai-scaffolding`
- Branch: `feature/ai-scaffolding`, forked from `master` (the project's working trunk; `main` is the PR target per project convention).
- `.worktrees/` is already gitignored.

## Files touched, summary

```
src/blueball/ai/                              (+ new package, 7 files)
src/blueball/agent.py                         (~ add FTNNAgent class only; Observation untouched)
src/blueball/entities/player.py               (~ PLAYER_GROUP filter + reached_goal flag; motion unchanged)
src/blueball/collision.py                     (~ PLAYER_GROUP constant + reached_goal write in on_goal)
src/blueball/camera.py                        (~ Camera.scale field + FreeCamera subclass)
src/blueball/scenes/train.py                  (+ new TrainScene)
src/blueball/config.py                        (~ TRAIN_* and GA_* and MAX_STEPS defaults)
train_main.py                                 (+ new entry script for TrainScene)
tests/test_ai_smoke.py                        (+ new)
tests/test_player.py                          (~ PLAYER_GROUP + reached_goal cases)
tests/test_collision.py                       (~ goal-handler reached_goal case)
```

`renderer.py`, `levels/`, `chunks/`, level JSON, and the existing `Observation` are **not** touched.

## What's deliberately out of scope

- **Real raycast input.** `Observation.rays` stays zeros until the level-design branch's enrichment lands. Documented as the load-bearing follow-up: when it merges, no `ai/` code needs to change beyond bumping `RAY_COUNT` in `observation.py` if the dimension changes. The currently-zero rays are still fed into the network so the layout is final.
- **`ReplayAgent` and FTNN-side recording.** Race mode is the consumer; race mode doesn't exist yet. This slice's `Agent` interface is stable enough that a future `ReplayAgent` is a single new file with no upstream changes.
- **On-disk genome persistence.** `TrainingResult` is held in memory; callers do their own `numpy.save` if they want it. When the project moves from "scaffolding smoke" to "actually run training," we'll create a `genomes/` folder and have the trainer write per-generation and final snapshots. Tracked in agent memory as a project-level directive.
- **NEAT (NeuroEvolution of Augmenting Topologies) escape hatch.** The architectural alignment is already in place: `train()` produces a `TrainingResult` keyed on genome arrays; a NEAT swap would replace `ftnn.py` + `genome.py` + `ga.py` and pass a different `agent_factory` through `evaluate`. We do not write the NEAT wrapper in this slice.
- **`multiprocessing.Pool` integration test.** Default trainer parallelism is serial; the multiprocessing path is documented in the `train()` docstring as a one-liner for callers. We'll wire and pin it when real training runs.
- **Headless `pygame` avoidance.** Importing `blueball.agent` transitively imports `pygame` (because `HumanAgent` uses `pygame.key.get_pressed()`). On Linux this is fine — `pygame.display.init()` is never called in headless workers. If we later move to `multiprocessing` with `spawn`-mode (or to platforms where `fork` is unsafe), the fix is a small `HumanAgent` lazy-import refactor — not a re-architecture.
- **TrainScene polish.** No charts, no per-agent fitness overlays, no live-rendered network activations. Just the population balls moving and a one-line HUD.
- **TrainScene "primitives scale with zoom."** Primitives stay in screen-space; only positions zoom. Documented above.
- **Render-budget guardrails.** `n_visible` cap is fixed at 16. Above that we'd want to revisit.
- **Save-file integration.** `TrainScene` does not unlock abilities for trained agents; it constructs `Player` with the default empty abilities set. The trainer evaluates the bare physics-and-jump player. Unlocks for AI agents are a Phase 4 design conversation, not a scaffolding decision.

All deferred items can be layered on without restructuring this slice.

## Acronyms used in this document

- **CT** — Collision Type (numeric tag pymunk uses to dispatch contact handlers; existing convention from v1 `collision.py`)
- **FTNN** — Fixed-Topology Neural Network
- **GA** — Genetic Algorithm
- **HUD** — Heads-Up Display
- **NEAT** — NeuroEvolution of Augmenting Topologies
- **PR** — Pull Request
- **RNG** — Random Number Generator
