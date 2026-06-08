"""PlayScene — gameplay loop. Accepts either a level path or in-memory data,
and (for Infinite Run) lazily streams chunks from a ChunkSampler instead of
materializing all of them at level load.
"""

from __future__ import annotations

import random
from pathlib import Path

import pygame

from .. import config, save
from ..abilities import Ability
from ..agent import HumanAgent
from ..camera import FollowCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..levels.streaming import (
    TerrainStream,
    LOAD_AHEAD as _LOAD_AHEAD,
    LOAD_BEHIND as _LOAD_BEHIND,
    INITIAL_BUILD_CHUNKS as _INITIAL_BUILD_CHUNKS,
    MAX_GROUND_ELEV as _MAX_GROUND_ELEV,
)
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


class PlayScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        level_path: Path | None = None,
        level_data: dict | None = None,
        sampler_seed: int | None = None,
    ) -> None:
        if (level_path is None) == (level_data is None):
            raise ValueError("PlayScene requires exactly one of level_path or level_data")
        self.screen = screen
        self.level_path = level_path
        self.level_data = level_data
        self.sampler_seed = sampler_seed
        # Streaming is opted into by providing sampler_seed. Regular hand-built
        # levels still load eagerly via load_level().
        self._streaming: bool = sampler_seed is not None
        from ..render.core import RenderCore
        self.core = RenderCore(screen)
        self.camera = FollowCamera(self.core.vw, self.core.vh)
        # Preserve the pre-overhaul visible-world span on the smaller surface.
        self.camera.scale = 1.0 / self.core.scale
        self.renderer = Renderer(self.core, self.camera)
        # Visual-only FX. Persist across _reset() so a death-burst keeps
        # animating after respawn.
        from ..render.particles import ParticleSystem
        from ..render.theme import get_active_theme
        self.particles = ParticleSystem(
            int(get_active_theme().params.get("particle_cap", 300))
        )
        self._was_grounded = False
        self._prev_collected = 0
        self._last_respawn_xy: tuple[float, float] | None = None
        self._exit_to_menu: bool = False
        # Infinite Run score = 10 * furthest x reached this run; best persists
        # per the local user's save across runs.
        self._best_score: int = save.get_best_score() if self._streaming else 0
        self._run_max_x: float = 0.0
        self._score: int = 0
        self._reset()

    def _reset(self) -> None:
        self.world = World()
        register_collisions(self.world.space, world_ref=self.world)

        if self._streaming:
            self._init_streaming_level()
        else:
            source = self.level_path if self.level_path is not None else self.level_data
            self.level_meta = load_level(source, self.world)

        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=unlocked,
        )
        if self._last_respawn_xy is not None:
            self.player.body.position = self._last_respawn_xy
        self.world.add_entity(self.player)
        self.camera.position = (self.player.body.position.x, self.player.body.position.y)
        # Reset transition trackers so a fresh run's first frame doesn't
        # spuriously emit landing/collect FX. The ParticleSystem and shake
        # state intentionally persist across the reset.
        self._was_grounded = False
        self._prev_collected = 0

    def _init_streaming_level(self) -> None:
        """Set up the per-tick chunk pipeline for Infinite Run.

        Computes the cosmetic level_meta, then hands terrain streaming to a
        shared `TerrainStream` (the same machine the headless trainer uses).
        """
        from ..levels.loader import LevelMeta, _hex_to_rgb

        meta_dict = self.level_data or {}
        background = _hex_to_rgb(meta_dict.get("background", "#202028"))
        ground = _hex_to_rgb(meta_dict.get("ground", "#666c70"))
        spawn = tuple(meta_dict.get("spawn", [80, 540]))
        # total_width is unknowable for a streaming level; report 0.0.
        self.level_meta = LevelMeta(
            name=meta_dict.get("name", f"Infinite Run (seed={self.sampler_seed})"),
            spawn=spawn,
            background=background,
            ground=ground,
            total_width=0.0,
        )
        self._terrain = TerrainStream(self.world, self.sampler_seed)

    # --- Thin delegations to the shared TerrainStream. PlayScene's tests and
    # the renderer reach for these names, so they stay on the scene. ---

    @property
    def _built_chunks(self) -> list[dict]:
        return self._terrain.built_chunks

    @property
    def _build_x(self) -> float:
        return self._terrain.build_x

    @property
    def _base_y(self) -> float:
        return self._terrain.base_y

    @property
    def _sampler(self):
        return self._terrain.sampler

    def _materialize_chunk(self, chunk) -> float:
        return self._terrain.materialize_chunk(chunk)

    def _maintain_streaming(self, player_x: float) -> None:
        self._terrain.maintain(player_x)

    def handle_events(self, events):
        if self._exit_to_menu:
            from .menu import MenuScene
            return MenuScene(self.screen)
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                from .menu import MenuScene
                return MenuScene(self.screen)
        return self

    def update(self, frame_dt: float) -> None:
        self.renderer.begin_frame(self.world)
        if self._streaming:
            self._maintain_streaming(self.player.body.position.x)
        # Sampled BEFORE the step purely to scale a landing's screen shake to the
        # impact speed — read-only, no effect on physics.
        pre_vy = self.player.body.velocity.y
        self.world.step(frame_dt)
        # Age existing visual FX every frame (before emitting new ones).
        self.core.update(frame_dt)
        self.particles.update(frame_dt)
        if self._streaming:
            self._run_max_x = max(self._run_max_x, self.player.body.position.x)
            self._score = int(10 * self._run_max_x)
        if self.player.dead:
            # Visual-only death burst + shake, emitted before the existing
            # score/reset logic so it animates through the respawn.
            self.particles.emit(
                "burst",
                (self.player.body.position.x, self.player.body.position.y),
                n=16,
            )
            self.core.add_shake(6.0)
            if self._streaming:
                # Bank the run's score, then re-randomize the run on death
                # instead of replaying the same deterministic layout; no
                # checkpoint respawn in Infinite Run.
                if self._score > self._best_score:
                    self._best_score = self._score
                    save.set_best_score(self._score)
                self._run_max_x = 0.0
                self._score = 0
                self.sampler_seed = random.randrange(1 << 32)
                self._last_respawn_xy = None
            else:
                self._last_respawn_xy = self.player.respawn_xy
            self._reset()
            return
        if self.world.level_complete:
            for ability in self.player.abilities:
                save.add_ability(ability.value)
            self._last_respawn_xy = None
            self._exit_to_menu = True
            return
        # Player is alive and the level isn't complete: emit visual-only FX by
        # observing existing player state (landing / collect / boost). None of
        # this feeds back into physics.
        pos = (self.player.body.position.x, self.player.body.position.y)
        grounded = self.player.grounded
        if grounded and not self._was_grounded:
            self.particles.emit("dust", pos, n=8)
            if abs(pre_vy) > 200:
                self.core.add_shake(min(8.0, abs(pre_vy) / 60.0))
        self._was_grounded = grounded
        if self.player.collectibles_collected > self._prev_collected:
            self.particles.emit("sparkle", pos, n=10)
            self._prev_collected = self.player.collectibles_collected
        if self.player._boost_multiplier > 1.0:
            self.particles.emit("trail", pos, n=3)
        self.camera.update(
            target=(self.player.body.position.x, self.player.body.position.y),
            dt=frame_dt,
        )

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        self.particles.draw(self.renderer)
        if self._streaming:
            self.renderer.draw_score(self._score, max(self._best_score, self._score))
        self.core.present()
