# Blue Ball — Design Spec

**Date:** 2026-05-23
**Status:** Approved for planning

## Overview

Blue Ball is a 2D physics-based platformer in the spirit of the Flash game *Red Ball*. The player controls a blue ball that rolls through hand-built levels, jumping over hazards and onto enemies to reach a goal. The project has three pillars:

1. **A smooth, hand-built platforming experience** built on real 2D physics.
2. **An AI opponent**, trained with a Genetic Algorithm (GA), that the player can race.
3. **An infinite mode** that procedurally stitches premade obstacle chunks into endless levels.

This spec defines the architecture for all three pillars. The first deliverable (v1) is a polished single-level vertical slice covering pillar 1. The AI and infinite mode are later milestones, but the architecture commits today to the interfaces they require.

## Goals

- The single-level v1 must feel smooth: stable framerate, responsive controls, no perceptible input lag, and ball motion that feels weighted and momentum-y in the Red Ball tradition.
- The engine is deterministic with a fixed seed so that the same inputs produce the same world state — required for reproducible GA training and replayable race ghosts.
- The codebase is composed of small, single-purpose modules with clear interfaces. Each pillar plugs into those interfaces without restructuring the others.
- Level content (v1, infinite mode, and GA training environments) all share one chunk-based data model and one chunk library.

## Non-goals (v1)

- Audio. The architecture exposes a hook in the scene update path; implementation is deferred.
- Save files and progression. Deferred until unlockable abilities (double jump, wall jump, ground pound) arrive.
- Sprite/asset pipeline. v1 ships with a flat geometric visual style drawn entirely from PyGame primitives.
- User-authored levels. Chunks are Python classes in v1; a declarative format may follow.

## Player abilities (v1 and future)

- **v1:** move left, move right, variable-height jump.
- **Future unlockables** (not v1, but the ability system is structured as composable verbs from the start): double jump, wall jump, ground pound. Each is a new entry in the player's ability set.

## Tech stack

- **PyGame** — required by the user. Handles rendering (filled primitives in v1), input, and the display loop.
- **Pymunk** — Python bindings for the Chipmunk2D rigid-body physics engine. Provides rolling motion, friction, restitution, multi-body interaction (relevant for pushable objects in the future), and built-in raycast queries (relevant for AI vision).
- **NumPy** — vectorized math for the neural network forward pass and raycast batching.
- **A custom GA implementation** — ~150 lines, no external library needed for the fixed-topology approach.
- **`neat-python`** — an escape hatch behind the same `Agent` interface, used only if the fixed-topology approach plateaus.
- **pytest** — automated tests for physics determinism, chunk loading, collision rules, input feel, and the trainer smoke path.

## Architecture

```
                ┌──────────────────────────────┐
                │       main.py / app          │  entry; routes between scenes
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐          ┌─────▼─────┐          ┌─────▼──────┐
   │  scenes  │          │  config   │          │  assets    │
   │  Play /  │          │ constants │          │ sprites,   │
   │  Menu /  │          │ tunables  │          │ sounds     │
   │  Train   │          └───────────┘          └────────────┘
   └────┬─────┘
        │ owns
   ┌────▼────────────────────────────────────────────────────┐
   │                       World                             │
   │  ┌─────────────┐   ┌───────────┐   ┌─────────────────┐  │
   │  │  Physics    │   │ Entities  │   │ Level (chunks   │  │
   │  │  (Pymunk    │   │ Player /  │   │  loaded into    │  │
   │  │   Space,    │   │ Enemy /   │   │  bodies +       │  │
   │  │   fixed     │   │ Pickup    │   │  entities)      │  │
   │  │   step)     │   │           │   │                 │  │
   │  └─────────────┘   └───────────┘   └─────────────────┘  │
   └─────────────────┬───────────────────────────────────────┘
                     │ observed by
            ┌────────▼────────┐         ┌─────────────────┐
            │   Renderer      │         │   Agent (in     │
            │ (PyGame surface,│         │   Player)       │
            │  camera, draw)  │         └────────┬────────┘
            └─────────────────┘                  │ uses
                                          ┌──────▼──────────┐
                                          │   ai/           │
                                          │   agent.py      │
                                          │   ftnn.py       │
                                          │   ga_trainer.py │
                                          └─────────────────┘
```

**Module boundaries:**

- **`World`** is the source of truth. It owns the Pymunk space, the level definition, and every entity. It is headless-friendly — it knows nothing about rendering, input, or windows. The same `World` runs in the live game and in GA training.
- **`Renderer`** reads `World`; it never mutates it. Camera modes live here.
- **`Scene`** glues input, world, and renderer for a specific mode. v1 implements `PlayScene` and `MenuScene`. Later milestones add `TrainScene`, `RaceScene`, and `InfiniteScene`.
- **`Agent`** is an interface (`act(observation) -> action`). The `Player` entity holds an `Agent` instance and asks it for actions each tick. A `HumanAgent` reads the keyboard; an `FTNNAgent` reads raycast observations and runs them through a fixed-topology neural network (FTNN); a `ReplayAgent` plays back a recorded action sequence (used for race-mode ghosts).
- **`ai/`** is isolated. The GA trainer constructs many `World`s, runs them headless, scores them, breeds them. The whole package could be swapped for `neat-python` later without touching the rest of the game.

## Game loop, physics, and input

### Fixed-timestep physics with render interpolation

The biggest single lever for "feels smooth" on PyGame. Physics and rendering run on separate clocks; the renderer interpolates between the two most recent physics states.

```
render at vsync (~60–144 Hz)
┌─────────────────────────────────────────────────────────────────┐
│  accumulator += frame_dt                                        │
│  while accumulator >= PHYS_DT (1/120s):                         │
│      world.physics_step(PHYS_DT)        ← deterministic         │
│      save previous_state, current_state                         │
│      accumulator -= PHYS_DT                                     │
│  alpha = accumulator / PHYS_DT          ← 0.0 to 1.0            │
│  renderer.draw(world, alpha)            ← lerp prev → current   │
└─────────────────────────────────────────────────────────────────┘
```

- **Physics tick rate: 120 Hz** (`PHYS_DT = 1/120 s`).
- **Determinism:** with fixed `PHYS_DT` and a fixed Pymunk seed, two runs with identical inputs produce identical states. Required for replayable ghosts and reproducible GA fitness scores.
- **Headless training** runs physics as fast as the CPU allows; rendering and the accumulator are skipped.

### Player physics tuning

The ball is a single Pymunk circle body. Tunables live in `config.py` for fast iteration:

| Knob | Initial value | Controls |
|---|---|---|
| `BALL_RADIUS` | 16 px | size |
| `BALL_MASS` | 1.0 | inertia |
| `BALL_FRICTION` | 0.9 | grip on the ground (affects rolling speed) |
| `MOVE_TORQUE` | 800 | how hard left/right press spins the ball |
| `MAX_ANGULAR_VEL` | 25 rad/s | speed cap so the ball doesn't spin infinitely |
| `JUMP_IMPULSE` | 400 | initial jump kick |
| `JUMP_CUT_FACTOR` | 0.4 | upward velocity multiplier when jump is released early |
| `GRAVITY` | (0, 1200) | downward force |
| `AIR_CONTROL` | 0.3 | torque multiplier when the ball is not grounded |

These are starting points; the first work after the engine is up is a tuning loop adjusting them by feel.

### Input feel layer

Three small mandatory tricks that separate "good" platformers from frustrating ones. All three operate on the `Player` regardless of whether the controlling `Agent` is a human or a GA-trained network — race ghosts therefore exhibit the same forgiveness window.

- **Jump buffer (~100 ms):** a jump press just before landing fires the moment the ball touches ground. Avoids the "I pressed jump but nothing happened" feeling.
- **Coyote time (~80 ms):** jumping is allowed for a short window after walking off a ledge. Forgives near-misses without changing intentionally-possible jumps.
- **Jump cut:** releasing the jump button mid-rise multiplies upward velocity by `JUMP_CUT_FACTOR`. Lets the player do small hops vs. tall jumps without a second input.

### Grounded detection

Pymunk does not provide "is grounded" natively. We track it through collision callbacks: a contact whose normal points up (within ~30° of vertical) sets `grounded = True`. Cleared at the start of each tick before contacts are re-evaluated. Coyote time uses the last-grounded timestamp.

## Entities

A flat, composition-based system. No deep inheritance hierarchy; every entity is a small class that owns its physics shape(s) and knows how to render itself.

```
Entity (abstract base)
├── Player              circle body; applies torque/jump; holds an Agent
├── Spike               static triangle shape; instant-kill on contact
├── Patroller           rectangle body; walks back and forth; dies if stomped
├── FallingHazard       triggered hazard (boulder / swinging spike / crumbling platform)
├── Collectible         sensor circle; removed on contact; counts toward fitness/score
└── Goal                sensor rectangle; ends the level on contact
```

Each entity exposes:

- `bodies` / `shapes` — physics objects added to the Pymunk space at level load.
- `update(dt)` — per-physics-tick logic (patroller reverses at edges; falling hazard triggers when the player crosses a line).
- `draw(renderer, alpha)` — render with interpolation between previous and current physics states.
- **Collision category bits** — used so race-mode ghosts don't collide with the human player and so collectibles act as sensors rather than solid blockers.

All entity collisions are routed through a single dispatcher. One place to look when behavior is wrong:

- `(Player, Spike)` → `player.die()`
- `(Player, Patroller)` with upward contact normal → `patroller.die()`; otherwise → `player.die()`
- `(Player, Collectible)` → `collectible.collect()`
- `(Player, Goal)` → `world.complete_level()`

## Level data model

A level is a **JSON file** with an ordered list of chunk references plus a few level-level fields. The same data model serves v1 (one hand-authored level) and infinite mode (a sampler emits chunks indefinitely).

```json
{
  "name": "Tutorial Hill",
  "background": "#7ec7ff",
  "ground": "#3b8a4a",
  "spawn": [50, 300],
  "chunks": [
    {"type": "flat", "width": 8},
    {"type": "spike_pit", "width": 3, "spikes": 4},
    {"type": "flat", "width": 4, "collectibles": [[2, 1], [5, 2]]},
    {"type": "patrol_platform", "length": 6, "patroller_speed": 60},
    {"type": "gap", "width": 4},
    {"type": "stairs_up", "steps": 3, "step_height": 32},
    {"type": "goal"}
  ]
}
```

Each chunk type is a Python class in `levels/chunks/` that materializes itself into bodies and entities at a given x-offset. Each chunk reports its own width so the next chunk is placed at the correct position. v1 ships with roughly eight chunk types (`flat`, `gap`, `spike_pit`, `patrol_platform`, `stairs_up`, `stairs_down`, `bump`, `goal`) and one hand-authored level. The chunk library is expected to grow toward "a few dozen" obstacles by the time infinite mode ships, per the user's stated goal for that pillar. Adding a new chunk type means writing one class and adding it to the library; both v1 levels and future infinite mode immediately benefit.

## Rendering (v1: flat / geometric style)

PyGame draws everything as filled primitives — circles, rectangles, triangles, polygons. No sprite assets needed for v1.

- Clear with the level's background color.
- Draw a parallax-able sky/horizon strip for cheap visual depth.
- Iterate entities and call each one's `draw(renderer, alpha)`.
- **Ball:** filled circle plus a darker arc rotated by the body's angle, so the ball visibly spins as it rolls. Sells the rolling momentum without sprite art.
- **Spikes:** filled triangles.
- **Ground/platforms:** filled rectangles with a darker top edge.
- **Collectibles:** filled circles with a small pulse animation (sine-of-time scale).

**Performance cushion:** primitives are cheap, but if frame time creeps the lever is **dirty-rect rendering** — redraw only the regions of the surface that changed each frame instead of the full surface. PyGame supports this natively. Not needed for v1 (~200 primitives at 60 Hz is well within budget), but the renderer is structured so it can be added later without rewriting entities.

## Camera

A `Camera` object owns the world-to-screen transform. Two modes:

- **`FollowCamera`** — tracks the player with a smoothed lerp plus a small dead-zone rectangle. The camera only moves when the player leaves the dead-zone box, preventing jitter on tiny movements. Used in `PlayScene` and (later) `RaceScene` and `InfiniteScene`.
- **`FreeCamera`** — arrow keys pan; `+` / `-` optionally zoom. Used in `TrainScene` so the developer can fly around watching the GA population train.

Switching scenes swaps the camera. Both implement the same `world_to_screen(pos)` interface; entities don't care which is active.

## Forward-looking: Agent interface

```python
class Agent:
    def reset(self, world): ...
    def act(self, observation: Observation) -> Action: ...

Observation = NamedTuple of:
    rays: np.ndarray              # shape (8,), raycast distances normalized 0..1
    vel: np.ndarray               # shape (2,), ball linear velocity
    ang_vel: float                # ball angular velocity
    grounded: bool
    nearest_collectible: tuple    # (dx, dy) relative offset, or None

Action = one of: IDLE, LEFT, RIGHT, JUMP, LEFT_JUMP, RIGHT_JUMP
```

Implementations:

- `HumanAgent` — reads the keyboard, ignores `observation`.
- `FTNNAgent` — passes `observation` through the fixed-topology neural network (FTNN); the output layer's argmax picks the action.
- `ReplayAgent` — replays a recorded action sequence tick-for-tick. How race-mode ghosts work after training.

The `Player` entity takes an `Agent` in its constructor. `PlayScene` constructs `Player(HumanAgent())`; race mode constructs two players with different agents.

## Forward-looking: Genetic Algorithm (GA) training

### Network shape (FTNN)

```
~14 input neurons (raycasts + scalars from Observation)
         ↓ fully connected
12 hidden neurons (tanh activation)
         ↓ fully connected
6 output neurons (one per Action; argmax wins)
```

Each agent's **genome** is the flat float32 array of weights and biases — roughly 246 numbers for the sizes above. Every agent in the population has the same shape, so crossover and mutation are array operations.

- **Mutation:** pick a random subset of weights, perturb each by a small Gaussian (`w += randn() * 0.1`). Occasionally re-roll a weight entirely.
- **Crossover:** per-weight uniform crossover between two parents (each weight independently inherits from parent A or parent B with 50/50 probability).
- **Selection:** tournament-style — for each child, sample N random parents and breed the top 2 by fitness.

### Training loop

```
def train(generations=200, pop_size=80):
    population = [random_ftnn() for _ in range(pop_size)]
    for gen in range(generations):
        fitnesses = [evaluate(genome) for genome in population]   # parallel
        log_stats(gen, fitnesses)
        population = breed(population, fitnesses)
        snapshot_best(population[0], gen)

def evaluate(genome) -> float:
    world = build_world(seed=TRAINING_LEVEL_SEED)
    agent = FTNNAgent(genome)
    player = world.spawn_player(agent)
    for step in range(MAX_STEPS):           # hard timeout, starting value ~3000 ticks (~25s at 120 Hz); tunable per level
        world.physics_step(PHYS_DT)
        if player.dead or player.reached_goal:
            break
    return fitness(player, step)
```

### Fitness function (starting point)

```
fitness = (
    progress_x
    + 50  * collectibles_collected
    + 200 * (1 if reached_goal else 0)
    - 0.01 * steps_taken
    - 100 * (1 if died else 0)
)
```

Tunable; the fitness function and the network topology are the two things most likely to need iteration during training.

### Parallelism and the training scene

- **Headless training** uses `multiprocessing.Pool` to run `evaluate()` across CPU cores. Each worker builds its own `World` with no renderer. Determinism guarantees the same genome scores the same fitness on every run.
- **`TrainScene`** is a separate path: it runs the same trainer in-process and renders the full population live on the same level so the developer can watch behavior emerge. Collision filtering makes the balls non-interacting. The free-fly camera (arrow keys) lets the developer pan around the population while training runs. The user explicitly requested this debugging affordance.

### Escape hatch: NeuroEvolution of Augmenting Topologies (NEAT)

If the FTNN approach plateaus — most likely once infinite mode arrives and agents must generalize across procedural chunk combinations — we drop in `neat-python` behind the same `Agent` interface. The trainer is swapped wholesale; nothing else changes.

## Forward-looking: Race mode

Two `Player` entities on the same level instance, sharing one camera (the existing `FollowCamera` tracking the human player). The AI player is constructed with a `ReplayAgent` loaded from the best recorded run on this level seed. Rendered at ~50% alpha; collision filtering prevents the ghost from blocking the player. First to reach the goal wins.

Race mode therefore reuses the existing `World`, `Player`, `Renderer`, and `FollowCamera` without modification. The only new components are the `ReplayAgent`, a recording mechanism on `FTNNAgent`, and a `RaceScene`.

## Forward-looking: Infinite mode

A `ChunkSampler` takes a seeded random number generator (RNG) and emits a stream of chunks indefinitely. The `World` maintains a window of chunks around the player — roughly two screens forward and one screen back. Chunks ahead are materialized as the player approaches; chunks behind are torn down once off-screen so the Pymunk space doesn't grow without bound.

**Difficulty curve via weighted chunk sampling:** each chunk in the library is tagged with a difficulty score. The sampler's weights shift toward harder chunks the further the player has traveled. The weighting function lives in `config.py`.

**Scoring** in infinite mode: distance traveled, collectibles, time survived, with optional combo multipliers for chained obstacles.

## Testing strategy

- **Physics determinism test** — run the same level with the same scripted input twice; assert that the resulting world states are byte-equal. Guards the foundational guarantee that the entire AI and replay system depends on.
- **Chunk loading tests** — each chunk class round-trips through the loader and produces the expected bodies and entities in the expected positions.
- **Collision dispatcher tests** — spike-on-player kills the player; top-of-patroller-on-player kills the patroller; side-of-patroller-on-player kills the player; collectible removes itself on contact. Pymunk runs headless in these tests.
- **Input feel tests** — jump buffer fires on landing; coyote time allows jump after walk-off; jump cut reduces upward velocity on release. Scripted input plus assertions on player state.
- **Trainer smoke test** — a 5-generation run on a tiny population on a trivial level finishes without crashing, and best-of-last-5 fitness ≥ best-of-first-5 (loose monotonic-improvement check tolerant to small-population variance).
- **Renderer tests are intentionally omitted.** Visual output is hard to assert. Smoothness is verified by running the game and watching it.

## Deferred decisions

- **Audio** — out of scope for v1. A hook in `Scene.update()` will exist; the SFX system is added later.
- **Save / progression** — out of scope until unlockable abilities arrive.
- **Mod-friendly chunk loading from disk** — chunks are Python classes in v1. If users want to ship custom levels, we add a declarative chunk spec format later.
- **NEAT** — escape hatch only, behind the same `Agent` interface.
- **Pushable physics objects (e.g., boxes the player pushes to reach a ledge)** — explicitly noted by the user as a future feature. Pymunk handles multi-body dynamics natively, so no architectural commitment is needed beyond ensuring the level data model can mark chunks/entities as dynamic.

## Acronyms used in this document

- **GA** — Genetic Algorithm
- **FTNN** — Fixed-topology Neural Network
- **NEAT** — NeuroEvolution of Augmenting Topologies
- **RNG** — Random Number Generator
- **SFX** — Sound Effects
