"""PlayScene — gameplay loop. Accepts either a level path or in-memory data."""

from __future__ import annotations

from pathlib import Path

import pygame

from .. import config, save
from ..abilities import Ability
from ..agent import HumanAgent
from ..camera import FollowCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
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
        self.camera = FollowCamera(screen.get_width(), screen.get_height())
        self.renderer = Renderer(screen, self.camera)
        self._last_respawn_xy: tuple[float, float] | None = None
        self._exit_to_menu: bool = False
        self._reset()

    def _reset(self) -> None:
        self.world = World()
        register_collisions(self.world.space, world_ref=self.world)
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
        # Snap the camera so the first frame doesn't lerp from origin
        self.camera.position = (self.player.body.position.x, self.player.body.position.y)

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
        self.world.step(frame_dt)
        if self.player.dead:
            self._last_respawn_xy = self.player.respawn_xy
            self._reset()
            return
        if self.world.level_complete:
            for ability in self.player.abilities:
                save.add_ability(ability.value)
            print(f"Level complete! Collectibles: {self.player.collectibles_collected}")
            self._last_respawn_xy = None
            self._exit_to_menu = True
            return
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
        pygame.display.flip()
