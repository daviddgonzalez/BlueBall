"""MenuScene — level select with 5 entries."""

from __future__ import annotations

import time
from pathlib import Path

import pygame

from .base import Scene
from .play import PlayScene


class MenuScene(Scene):
    INFINITE_RUN = "__infinite__"

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        levels_dir = Path(__file__).parent.parent / "levels"
        self.entries: list[tuple[str, object]] = [
            ("Tutorial Hill", levels_dir / "tutorial_hill.json"),
            ("Vertical Climb", levels_dir / "vertical_climb.json"),
            ("Speed Run", levels_dir / "speed_run.json"),
            ("Maze", levels_dir / "maze.json"),
            ("Infinite Run", self.INFINITE_RUN),
        ]
        self.cursor: int = 0
        self._font = None
        self._title_font = None

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.cursor = max(0, self.cursor - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.cursor = min(len(self.entries) - 1, self.cursor + 1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    _, target = self.entries[self.cursor]
                    if target == self.INFINITE_RUN:
                        from ..levels.sampler import ChunkSampler
                        seed = int(time.time() * 1000) & 0xFFFFFFFF
                        sampler = ChunkSampler(seed=seed)
                        level_data = {
                            "name": f"Infinite Run (seed={seed})",
                            "background": "#202028",
                            "ground": "#666c70",
                            "spawn": [80, 540],
                            "chunks": list(sampler),
                        }
                        return PlayScene(self.screen, level_data=level_data, sampler_seed=seed)
                    return PlayScene(self.screen, level_path=target)
        return self

    def update(self, frame_dt: float) -> None:
        pass

    def _ensure_fonts(self) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont(None, 36)
        if self._title_font is None:
            self._title_font = pygame.font.SysFont(None, 64)

    def draw(self) -> None:
        self._ensure_fonts()
        self.screen.fill((20, 30, 50))
        title_surf = self._title_font.render("Blue Ball", True, (255, 255, 255))
        self.screen.blit(title_surf, ((self.screen.get_width() - title_surf.get_width()) // 2, 80))
        for i, (label, _) in enumerate(self.entries):
            color = (255, 220, 80) if i == self.cursor else (200, 200, 200)
            prefix = "> " if i == self.cursor else "  "
            surf = self._font.render(prefix + label, True, color)
            self.screen.blit(surf, (self.screen.get_width() // 2 - 100, 220 + i * 50))
        hint = self._font.render("Up/Down: select   Enter: play   Esc: quit", True, (140, 140, 160))
        self.screen.blit(hint, ((self.screen.get_width() - hint.get_width()) // 2, self.screen.get_height() - 60))
        pygame.display.flip()
