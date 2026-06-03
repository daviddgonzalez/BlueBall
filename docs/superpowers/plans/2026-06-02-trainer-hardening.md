# Trainer Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GA trainer trustworthy before a real run — reproducible across machines, verified under multiprocessing, and able to visually train on the real Infinite Run course with the full population evaluated.

**Architecture:** Three independent workstreams. WS1 adds a drift-free fixed substep to `World` and points the headless evaluators at it. WS2 adds tests pinning the `multiprocessing.Pool` path against serial. WS3 rewrites `TrainScene` to split a cosmetic live Infinite Run display from an asynchronous `Pool.map_async` truth-eval of the whole population.

**Tech Stack:** Python, numpy, pymunk, pygame (headless via `SDL_VIDEODRIVER=dummy`), pytest, multiprocessing.

**Spec:** `docs/superpowers/specs/2026-06-02-trainer-hardening-design.md`

**Conventions:**
- Run tests with the repo venv: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest …` (run from the worktree root so `conftest.py` puts the worktree `src/` on `sys.path`).
- Work happens in the `feature/ai-scaffolding` worktree at `/home/ddgg0/projects/BlueBall/.worktrees/ai-scaffolding`.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/blueball/world.py` | physics world + stepping | Add `substep()`; factor the per-substep body shared with `step()` |
| `src/blueball/ai/trainer.py` | headless eval + GA loop | `evaluate`/`evaluate_infinite` use `world.substep()`; make `INFINITE_SPAWN` public |
| `src/blueball/scenes/train.py` | visual trainer scene | Rewrite: Infinite Run display + async full-pop Pool eval |
| `train_main.py` | visual trainer entry | Default to `infinite_seed=config.INFINITE_RUN_SEED` |
| `tests/test_world_determinism.py` | world stepping tests | Add `substep()` exactness test |
| `tests/test_ai_smoke.py` | AI tests | Add Pool tests + long-run determinism; rewrite the two TrainScene tests |

---

### Task 1: Drift-free fixed substep (WS1)

**Goal:** Add `World.substep()` that advances exactly one `PHYS_DT` substep with no accumulator, and point the headless evaluators at it so long runs are bit-identical across machines.

**Files:**
- Modify: `src/blueball/world.py` (the `step` method region, lines 42-63)
- Modify: `src/blueball/ai/trainer.py` (loops in `evaluate` ~line 65-72 and `evaluate_infinite` ~line 110-117)
- Test: `tests/test_world_determinism.py`

**Acceptance Criteria:**
- [ ] `World.substep()` runs exactly one `space.step(PHYS_DT)` plus one entity-update pass, with no change to `_accumulator`.
- [ ] `step()` and `substep()` share one per-substep code path (cannot drift).
- [ ] `evaluate` and `evaluate_infinite` call `world.substep()` once per iteration instead of `world.step(config.PHYS_DT)`.
- [ ] `evaluate_infinite` is bit-identical across two calls at `max_steps=2000`.
- [ ] Full suite stays green.

**Verify:** `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_world_determinism.py tests/test_ai_smoke.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_world_determinism.py`:

```python
def test_substep_runs_exactly_one_fixed_substep():
    """substep() advances one PHYS_DT substep with no accumulator change."""
    import pymunk
    from blueball import config
    from blueball.world import World

    w = World()
    body = pymunk.Body(mass=1.0, moment=10.0)
    body.position = (0.0, 0.0)
    shape = pymunk.Circle(body, 5.0)
    w.space.add(body, shape)

    accum_before = w._accumulator
    w.substep()
    # One substep of gravity: velocity gains gravity_y * PHYS_DT.
    assert abs(body.velocity.y - config.GRAVITY[1] * config.PHYS_DT) < 1e-9
    # The accumulator must be untouched (substep bypasses it).
    assert w._accumulator == accum_before


def test_substep_calls_entity_update_once():
    """Each substep runs exactly one entity update pass."""
    from blueball.world import World

    class Counter:
        bodies = ()
        shapes = ()
        constraints = ()
        def __init__(self):
            self.n = 0
        def update(self, dt):
            self.n += 1

    w = World()
    c = Counter()
    w.add_entity(c)
    w.substep()
    w.substep()
    assert c.n == 2
```

Add to `tests/test_ai_smoke.py` (near the other infinite tests):

```python
def test_evaluate_infinite_deterministic_over_long_run():
    """Bit-identical fitness at a large max_steps, where accumulator float
    drift would previously have surfaced."""
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    g = random_genome(np.random.default_rng(11))
    _, f1 = evaluate_infinite((0, g, 1234, 1, 2000))
    _, f2 = evaluate_infinite((0, g, 1234, 1, 2000))
    assert f1 == f2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_world_determinism.py::test_substep_runs_exactly_one_fixed_substep -q`
Expected: FAIL with `AttributeError: 'World' object has no attribute 'substep'`

(The long-run determinism test may already pass since one `step(PHYS_DT)` fires one substep today; it guards against regressions once `substep()` is wired.)

- [ ] **Step 3: Add `substep()` and factor the shared body in `world.py`**

Replace the `step` method (lines 42-63) with a factored version plus `substep`:

```python
    def _run_one_substep(self) -> None:
        """Advance physics + entities by exactly one PHYS_DT substep."""
        self.space.step(config.PHYS_DT)
        for entity in self.entities:
            update = getattr(entity, "update", None)
            if update is not None:
                update(config.PHYS_DT)

    def substep(self) -> None:
        """Advance by exactly one fixed PHYS_DT substep, bypassing the
        real-time accumulator. Deterministic across hosts — N calls == N
        substeps with no float residual. Used by the headless trainer; the
        live game uses step(frame_dt) for real-time pacing.
        """
        self._run_one_substep()

    def step(self, frame_dt: float) -> int:
        """Advance the simulation by `frame_dt` real seconds.

        Internally runs zero or more fixed substeps of `config.PHYS_DT`.
        Returns the number of substeps actually executed (useful for tests
        and for debug overlays).
        """
        self._accumulator += frame_dt
        substeps = 0
        while self._accumulator >= config.PHYS_DT and substeps < config.MAX_ACCUMULATED_STEPS:
            self._run_one_substep()
            self._accumulator -= config.PHYS_DT
            substeps += 1

        if self._accumulator >= config.PHYS_DT:
            # Spiral-of-death guard: drop leftover time we couldn't run
            self._accumulator = 0.0
        return substeps
```

- [ ] **Step 4: Point the evaluators at `substep()`**

In `src/blueball/ai/trainer.py`, `evaluate` loop — replace:

```python
        world.step(config.PHYS_DT)
        steps += 1
        if player.dead or player.reached_goal:
            break
```

with:

```python
        world.substep()
        steps += 1
        if player.dead or player.reached_goal:
            break
```

In `evaluate_infinite` loop — replace:

```python
        terrain.maintain(player.body.position.x)
        world.step(config.PHYS_DT)
        steps += 1
        if player.dead:
            break
```

with:

```python
        terrain.maintain(player.body.position.x)
        world.substep()
        steps += 1
        if player.dead:
            break
```

Also update the comment block in `evaluate` that references `World.step` to say the trainer now uses the drift-free `substep()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_world_determinism.py tests/test_ai_smoke.py -q`
Expected: PASS (all)

- [ ] **Step 6: Run full suite**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all green)

- [ ] **Step 7: Commit**

```bash
git add src/blueball/world.py src/blueball/ai/trainer.py tests/test_world_determinism.py tests/test_ai_smoke.py
git commit -m "feat(world): drift-free substep() for deterministic headless eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Multiprocessing.Pool integration test (WS2)

**Goal:** Pin that the parallel evaluation path matches serial and survives pickling/reordering, so long `Pool.imap` runs are trustworthy.

**Files:**
- Test: `tests/test_ai_smoke.py`

**Acceptance Criteria:**
- [ ] `train(infinite_seed=…, map_fn=Pool(2).imap)` yields a `best_genome` `array_equal` to the serial `map` run with the same seeds.
- [ ] Evaluating several genomes via `Pool.imap` returns one `(idx, fitness)` per genome and reorders correctly to match serial.
- [ ] The test closes the Pool and skips cleanly if the platform cannot start workers.
- [ ] Full suite stays green.

**Verify:** `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k pool -q` → pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ai_smoke.py`:

```python
def _make_pool(n):
    """Create a Pool, or skip the test if the platform can't start workers."""
    import multiprocessing
    try:
        return multiprocessing.Pool(n)
    except (OSError, ValueError) as e:  # pragma: no cover - platform guard
        pytest.skip(f"multiprocessing unavailable: {e}")


def test_pool_eval_matches_serial_determinism():
    """train() under Pool.imap produces the same best genome as serial map."""
    from blueball.ai.trainer import train
    serial = train(pop_size=6, generations=2, infinite_seed=7,
                   max_steps=200, ga_seed=0, world_seed=1)
    pool = _make_pool(2)
    try:
        parallel = train(pop_size=6, generations=2, infinite_seed=7,
                         max_steps=200, ga_seed=0, world_seed=1,
                         map_fn=pool.imap)
    finally:
        pool.close()
        pool.join()
    assert np.array_equal(serial.best_genome, parallel.best_genome)


def test_pool_evaluate_infinite_reorders_results():
    """evaluate_infinite is picklable; Pool results, once sorted by idx,
    match the serial mapping element-for-element."""
    from blueball.ai.trainer import evaluate_infinite
    from blueball.ai.genome import random_genome
    rng = np.random.default_rng(0)
    args = [(i, random_genome(rng), 1234, 1, 150) for i in range(5)]
    serial = sorted(map(evaluate_infinite, args), key=lambda r: r[0])
    pool = _make_pool(2)
    try:
        parallel = sorted(pool.imap(evaluate_infinite, args), key=lambda r: r[0])
    finally:
        pool.close()
        pool.join()
    assert [r[0] for r in parallel] == [0, 1, 2, 3, 4]
    for s, p in zip(serial, parallel):
        assert s[0] == p[0]
        assert s[1] == p[1]
```

- [ ] **Step 2: Run tests to verify they pass (this is coverage of existing code)**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k pool -q`
Expected: PASS. If `test_pool_eval_matches_serial_determinism` FAILS on `array_equal`, that is a real nondeterminism bug — Task 1 (`substep()`) must be complete first; investigate before forcing the test green.

- [ ] **Step 3: Run full suite**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_ai_smoke.py
git commit -m "test(ai): pin multiprocessing.Pool eval matches serial

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: TrainScene Infinite Run parity + async viz-only eval (WS3)

**Goal:** Rewrite `TrainScene` so the visual trainer streams the real Infinite Run course for display while an asynchronous `Pool.map_async` evaluates the full population for authoritative fitness — removing the `-inf` invisible-agent dilution and matching the headless trainer.

**Files:**
- Modify: `src/blueball/ai/trainer.py` (rename `_INFINITE_SPAWN` → `INFINITE_SPAWN`, public)
- Rewrite: `src/blueball/scenes/train.py`
- Modify: `train_main.py`
- Test: `tests/test_ai_smoke.py` (rewrite the two `test_train_scene_*` tests + add full-pop eval test)

**Acceptance Criteria:**
- [ ] `TrainScene` accepts exactly one of `level_path` / `infinite_seed`; raises `ValueError` otherwise.
- [ ] With `infinite_seed`, the display World streams chunks via `TerrainStream`; `n_visible` players are stepped live (cosmetic).
- [ ] Authoritative fitness is computed for the **full** `pop_size` via an injectable pool's `map_async`; generation advances when the result is ready; breeding uses those fitnesses.
- [ ] An injected synchronous pool stub lets a test advance a generation and confirm all `pop_size` genomes were evaluated (not just `n_visible`).
- [ ] `train_main.py` launches Infinite Run training with `config.INFINITE_RUN_SEED`.
- [ ] Full suite stays green.

**Verify:** `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k train_scene -q` → pass

**Steps:**

- [ ] **Step 1: Make the Infinite Run spawn public in `trainer.py`**

Rename the constant and its one use in `evaluate_infinite`:

```python
# Spawn for streamed Infinite Run evaluation. Matches PlayScene's default
# Infinite Run spawn so the headless trainer and the visual TrainScene drop
# the ball where a human would start. The guaranteed Flat at x=0 gives it
# ground to land on.
INFINITE_SPAWN = (80.0, 540.0)
```

In `evaluate_infinite`, change `spawn_x, spawn_y = _INFINITE_SPAWN` to `spawn_x, spawn_y = INFINITE_SPAWN`.

- [ ] **Step 2: Write the failing/updated tests**

Replace the two existing `test_train_scene_constructs_and_steps` and `test_train_scene_advances_generation_after_max_steps` tests in `tests/test_ai_smoke.py` with the following (the constructor signature and eval model changed, so the old bodies no longer apply):

```python
class _SyncResult:
    """A map_async result that has already computed eagerly."""
    def __init__(self, values):
        self._values = values
    def ready(self):
        return True
    def get(self, timeout=None):
        return self._values


class _SyncPool:
    """Synchronous stand-in for multiprocessing.Pool — runs map_async eagerly
    so TrainScene tests are deterministic and process-free."""
    def __init__(self):
        self.calls = 0
    def map_async(self, fn, iterable):
        self.calls += 1
        return _SyncResult([fn(x) for x in iterable])
    def close(self):
        pass
    def terminate(self):
        pass
    def join(self):
        pass


def test_train_scene_constructs_and_steps_infinite(headless_pygame):
    """TrainScene builds on an Infinite Run seed, owns n_visible FTNN players
    on a streamed terrain, and update() ticks do not crash."""
    from blueball.scenes.train import TrainScene
    from blueball.agent import FTNNAgent
    from blueball import collision
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=6,
        n_visible=4,
        generations=2,
        max_steps=60,
        pool=_SyncPool(),
    )
    assert len(scene._players) == 4
    for p in scene._players:
        assert isinstance(p.agent, FTNNAgent)
        assert p.shape.filter.group == collision.PLAYER_GROUP
    for _ in range(10):
        scene.update(1 / 60)


def test_train_scene_evaluates_full_population(headless_pygame):
    """The async truth-eval scores all pop_size genomes, not just n_visible,
    and a generation advances when the result is ready."""
    from blueball.scenes.train import TrainScene
    pool = _SyncPool()
    scene = TrainScene(
        headless_pygame,
        infinite_seed=1234,
        pop_size=8,
        n_visible=3,
        generations=3,
        max_steps=40,
        pool=pool,
    )
    start_gen = scene.current_gen
    # The sync pool is ready immediately, so the first update() that polls it
    # advances a generation.
    scene.update(1 / 60)
    assert scene.current_gen == start_gen + 1
    # The eval covered the full population (8), not just the 3 visible.
    assert scene._last_fitnesses is not None
    assert len(scene._last_fitnesses) == 8


def test_train_scene_rejects_neither_or_both_sources(headless_pygame):
    from blueball.scenes.train import TrainScene
    with pytest.raises(ValueError):
        TrainScene(headless_pygame, pool=_SyncPool())
    with pytest.raises(ValueError):
        TrainScene(headless_pygame, level_path=_level_path(),
                   infinite_seed=1, pool=_SyncPool())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k train_scene -q`
Expected: FAIL (old signature `TrainScene(screen, level_path, …)` no longer matches; `pool`/`infinite_seed` kwargs and `_last_fitnesses` don't exist yet).

- [ ] **Step 4: Rewrite `src/blueball/scenes/train.py`**

Replace the entire file with:

```python
"""TrainScene — visual GA trainer.

Splits two independent paths:

* Display (cosmetic): one World streaming the real Infinite Run course (or a
  static level), with n_visible FTNN players stepped live so the user can
  watch agents run. Determines nothing about selection.
* Truth (authoritative): the full population is evaluated headlessly off the
  render loop via an injectable pool's map_async. Its fitnesses drive elitism
  and breeding. Because players share PLAYER_GROUP and terrain is identical for
  a fixed seed, a visible player's live run reproduces its headless fitness —
  the display is a faithful window, not an approximation.

The generation advances when the async eval completes (the on-screen run may be
cut mid-animation; the HUD shows the authoritative numbers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .. import collision, config
from ..agent import FTNNAgent
from ..ai.ga import breed
from ..ai.genome import random_genome
from ..ai.trainer import INFINITE_SPAWN, evaluate, evaluate_infinite
from ..camera import FreeCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..levels.streaming import TerrainStream
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


class TrainScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        *,
        level_path: Path | None = None,
        infinite_seed: int | None = None,
        pop_size: int = config.TRAIN_POP_SIZE,
        n_visible: int = 16,
        generations: int = config.TRAIN_GENERATIONS,
        ga_seed: int = config.GA_SEED,
        world_seed: int = config.DEFAULT_SEED,
        max_steps: int = config.MAX_STEPS,
        pool=None,
    ) -> None:
        if (level_path is None) == (infinite_seed is None):
            raise ValueError(
                "TrainScene requires exactly one of level_path or infinite_seed"
            )
        self.screen = screen
        self.level_path = level_path
        self.infinite_seed = infinite_seed
        self.pop_size = pop_size
        self.n_visible = min(n_visible, pop_size)
        self.generations = generations
        self.ga_seed = ga_seed
        self.world_seed = world_seed
        self.max_steps = max_steps

        # Injectable for tests; default to a real Pool sized to the machine.
        if pool is None:
            import multiprocessing
            pool = multiprocessing.Pool()
        self._pool = pool
        self._eval_fn = evaluate if infinite_seed is None else evaluate_infinite

        pygame.display.set_caption("Blue Ball — Train")
        self.camera = FreeCamera(screen.get_width(), screen.get_height())
        self.renderer = Renderer(screen, self.camera)
        self._font = pygame.font.Font(None, 20)

        self._ga_rng = np.random.default_rng(ga_seed)
        self.population: list[np.ndarray] = [
            random_genome(self._ga_rng) for _ in range(pop_size)
        ]
        self.current_gen = 0
        self.best_fitness = float("-inf")
        self.best_mean = 0.0
        self._last_fitnesses: np.ndarray | None = None
        self._done = False

        self._start_generation()

    # ---- Generation lifecycle ----

    def _make_args(self, i: int) -> tuple:
        if self.infinite_seed is None:
            return (i, self.population[i], self.world_seed, self.level_path, self.max_steps)
        return (i, self.population[i], int(self.infinite_seed), self.world_seed, self.max_steps)

    def _start_generation(self) -> None:
        """Build the cosmetic display World + n_visible players, and launch the
        async authoritative eval for the whole population."""
        self.world = World(seed=self.world_seed)
        register_collisions(self.world.space, world_ref=self.world)

        if self.infinite_seed is None:
            self.level_meta = load_level(self.level_path, self.world)
            self._spawn_xy = (float(self.level_meta.spawn[0]),
                              float(self.level_meta.spawn[1]))
            self._terrain = None
        else:
            from ..levels.loader import LevelMeta, _hex_to_rgb
            self.level_meta = LevelMeta(
                name=f"Infinite Run (seed={self.infinite_seed})",
                spawn=INFINITE_SPAWN,
                background=_hex_to_rgb("#202028"),
                ground=_hex_to_rgb("#666c70"),
                total_width=0.0,
            )
            self._spawn_xy = INFINITE_SPAWN
            self._terrain = TerrainStream(self.world, int(self.infinite_seed))

        self._players: list[Player] = []
        for i in range(self.n_visible):
            p = Player(agent=FTNNAgent(self.population[i]), spawn_xy=self._spawn_xy)
            self.world.add_entity(p)
            self._players.append(p)
        self.camera.position = self._spawn_xy
        self.renderer.reset_interpolation()

        # Launch authoritative eval of the FULL population off the render loop.
        self._eval_result = self._pool.map_async(
            self._eval_fn, [self._make_args(i) for i in range(self.pop_size)]
        )

    def _leading_visible_x(self) -> float:
        return max((p.body.position.x for p in self._players), default=self._spawn_xy[0])

    def _complete_generation(self) -> None:
        results = sorted(self._eval_result.get(), key=lambda r: r[0])
        fits = np.array([r[1] for r in results], dtype=np.float64)
        self._last_fitnesses = fits
        self.best_fitness = max(self.best_fitness, float(fits.max()))
        self.best_mean = float(fits.mean())
        self.current_gen += 1
        if self.current_gen < self.generations:
            self.population = breed(
                self.population, fits, self._ga_rng,
                elitism=config.GA_ELITISM,
                tournament_k=config.GA_TOURNAMENT_K,
                mutation_rate=config.GA_MUTATION_RATE,
                mutation_sigma=config.GA_MUTATION_SIGMA,
            )
            self._start_generation()
        else:
            self._done = True
            self._close_pool()

    def _close_pool(self) -> None:
        try:
            self._pool.close()
            self._pool.join()
        except Exception:
            pass

    # ---- Scene API ----

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                try:
                    self._pool.terminate()
                except Exception:
                    pass
                return None
        self.camera.handle_events(events)
        return self

    def update(self, frame_dt: float) -> None:
        if self._done:
            return
        self.renderer.begin_frame(self.world)
        keys = pygame.key.get_pressed()
        self.camera.update(keys_pressed=keys, dt=frame_dt)
        # Cosmetic: stream terrain ahead of the leading visible player, step
        # the display World in real time.
        if self._terrain is not None:
            self._terrain.maintain(self._leading_visible_x())
        self.world.step(frame_dt)
        # Generation flips when the authoritative async eval is ready.
        if self._eval_result.ready():
            self._complete_generation()

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        if self._done:
            label = f"DONE  best {self.best_fitness:.1f}  mean {self.best_mean:.1f}"
        else:
            evaluating = "" if self._eval_result.ready() else "  evaluating…"
            label = (
                f"gen {self.current_gen + 1}/{self.generations}  "
                f"pop {self.pop_size} (showing {self.n_visible})  "
                f"best {self.best_fitness:.1f}  mean {self.best_mean:.1f}{evaluating}"
            )
        surf = self._font.render(label, True, (255, 255, 255))
        self.screen.blit(surf, (12, 12))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/test_ai_smoke.py -k train_scene -q`
Expected: PASS

- [ ] **Step 6: Update `train_main.py` to default to Infinite Run**

Replace the level-path construction in `train_main.py`:

```python
    scene = TrainScene(screen, infinite_seed=config.INFINITE_RUN_SEED)
```

Remove the now-unused `level_path = Path(...)` line and the `from pathlib import Path` import if nothing else uses it.

- [ ] **Step 7: Smoke-run the entry headlessly (no real training, just boot)**

Run:
```bash
SDL_VIDEODRIVER=dummy timeout 5 /home/ddgg0/projects/BlueBall/.venv/bin/python -c "
import os; os.environ['SDL_VIDEODRIVER']='dummy'
import pygame; pygame.init(); pygame.font.init()
s=pygame.display.set_mode((320,240))
from blueball import config
from blueball.scenes.train import TrainScene
sc=TrainScene(s, infinite_seed=config.INFINITE_RUN_SEED, pop_size=4, n_visible=2, generations=1, max_steps=30)
for _ in range(30): sc.update(1/60)
print('boot ok')
"
```
(Run from the worktree root with `PYTHONPATH=src` if the installed package shadows the worktree.) Expected: prints `boot ok` (a real Pool spins up and a generation completes).

- [ ] **Step 8: Run full suite**

Run: `SDL_VIDEODRIVER=dummy /home/ddgg0/projects/BlueBall/.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all green)

- [ ] **Step 9: Commit**

```bash
git add src/blueball/scenes/train.py src/blueball/ai/trainer.py train_main.py tests/test_ai_smoke.py
git commit -m "feat(train): visual trainer on Infinite Run with async full-pop eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Worktree boundary:** stay in `/home/ddgg0/projects/BlueBall/.worktrees/ai-scaffolding`; do not switch branches.
- **Test invocation:** always `SDL_VIDEODRIVER=dummy` with the repo venv python; run from the worktree root so `conftest.py` injects the worktree `src/`.
- **Do not** touch the deferred items (watch-best mode, fitness shaping, observation encoding, ReplayAgent, NEAT) — they are explicitly out of scope.
- **Order matters:** Task 1 before Task 2 (Pool determinism equality depends on the drift-free substep). Task 3 is independent of Task 2 but depends on Task 1's public `INFINITE_SPAWN`/`substep`.
