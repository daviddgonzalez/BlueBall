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
            evaluating = "" if self._eval_result.ready() else "  evaluating..."
            label = (
                f"gen {self.current_gen + 1}/{self.generations}  "
                f"pop {self.pop_size} (showing {self.n_visible})  "
                f"best {self.best_fitness:.1f}  mean {self.best_mean:.1f}{evaluating}"
            )
        surf = self._font.render(label, True, (255, 255, 255))
        self.screen.blit(surf, (12, 12))
