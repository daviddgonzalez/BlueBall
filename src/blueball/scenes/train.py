"""TrainScene — runs the GA trainer in-process and renders the live population.

Owns one shared World with N visible Player entities, each driven by an
FTNNAgent. All players share PLAYER_GROUP so they don't collide with each
other. At each generation boundary the World is rebuilt fresh.

The trainer's evaluation happens INSIDE this scene's update loop — we don't
call ai.trainer.train() because we want to render mid-generation. Instead
the scene runs its own per-tick generation accumulator using the same
fitness/breed primitives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .. import collision, config
from ..agent import FTNNAgent
from ..ai.fitness import FitnessInputs, fitness
from ..ai.ftnn import GENOME_SIZE
from ..ai.ga import breed
from ..ai.genome import random_genome
from ..camera import FreeCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


class TrainScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        level_path: Path,
        *,
        pop_size: int = config.TRAIN_POP_SIZE,
        n_visible: int = 16,
        generations: int = config.TRAIN_GENERATIONS,
        ga_seed: int = 0,
        world_seed: int = config.DEFAULT_SEED,
        max_steps: int = config.MAX_STEPS,
    ) -> None:
        self.screen = screen
        self.level_path = level_path
        self.pop_size = pop_size
        self.n_visible = min(n_visible, pop_size)
        self.generations = generations
        self.ga_seed = ga_seed
        self.world_seed = world_seed
        self.max_steps = max_steps

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

        self._build_world_for_current_gen()

    # ---- Generation lifecycle ----

    def _build_world_for_current_gen(self) -> None:
        """Construct a fresh World, load the level, spawn n_visible Players
        driven by the first n_visible genomes of the current population."""
        self.world = World(seed=self.world_seed)
        register_collisions(self.world.space, world_ref=self.world)
        self.level_meta = load_level(self.level_path, self.world)
        self._spawn_xy = (float(self.level_meta.spawn[0]),
                          float(self.level_meta.spawn[1]))
        self._players: list[Player] = []
        for i in range(self.n_visible):
            agent = FTNNAgent(self.population[i])
            p = Player(agent=agent, spawn_xy=self._spawn_xy)
            self.world.add_entity(p)
            self._players.append(p)
        # Snap camera to the spawn so the developer sees something on gen start.
        self.camera.position = self._spawn_xy
        self._gen_steps = 0

    def _all_visible_done(self) -> bool:
        return all(p.dead or p.reached_goal for p in self._players)

    def _score_visible_players(self) -> np.ndarray:
        """Compute fitness for the n_visible players. The remaining
        (pop_size - n_visible) population members get -inf so they are never
        chosen by tournament_select; once headless evaluation is added in a
        follow-up task they'll get real fitness values."""
        fits = np.full(self.pop_size, -np.inf, dtype=np.float64)
        for i, p in enumerate(self._players):
            fits[i] = fitness(FitnessInputs(
                progress_x=float(p.body.position.x - self._spawn_xy[0]),
                collectibles=int(p.collectibles_collected),
                reached_goal=bool(p.reached_goal),
                died=bool(p.dead),
                steps_taken=self._gen_steps,
            ))
        return fits

    def _advance_generation(self) -> None:
        fits = self._score_visible_players()
        self.best_fitness = max(self.best_fitness, float(fits.max()))
        self.best_mean = float(fits[:self.n_visible].mean())
        self.population = breed(
            self.population, fits, self._ga_rng,
            elitism=config.GA_ELITISM,
            tournament_k=config.GA_TOURNAMENT_K,
            mutation_rate=config.GA_MUTATION_RATE,
            mutation_sigma=config.GA_MUTATION_SIGMA,
        )
        self.current_gen += 1
        if self.current_gen < self.generations:
            self._build_world_for_current_gen()

    # ---- Scene API ----

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
        self.camera.handle_events(events)
        return self

    def update(self, frame_dt: float) -> None:
        if self.current_gen >= self.generations:
            return
        self.renderer.begin_frame(self.world)
        keys = pygame.key.get_pressed()
        self.camera.update(keys_pressed=keys, dt=frame_dt)
        # Drive physics via the world's accumulator (real-time pace) so the
        # visualization plays back at the same rate the game does.
        substeps = self.world.step(frame_dt)
        self._gen_steps += substeps
        if self._gen_steps >= self.max_steps or self._all_visible_done():
            self._advance_generation()

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        live = sum(1 for p in self._players if not (p.dead or p.reached_goal))
        if self.current_gen >= self.generations:
            label = f"DONE  best {self.best_fitness:.1f}  mean {self.best_mean:.1f}"
        else:
            label = (
                f"gen {self.current_gen + 1}/{self.generations}  "
                f"best {self.best_fitness:.1f}  mean {self.best_mean:.1f}  "
                f"live {live}/{self.n_visible}"
            )
        surf = self._font.render(label, True, (255, 255, 255))
        self.screen.blit(surf, (12, 12))
