"""ModeSelectScene — landing screen: Single Player or Race."""
from __future__ import annotations

import pygame

from .base import Scene


class ModeSelectScene(Scene):
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.entries = [("Single Player", "single"), ("Race", "race")]
        self.cursor = 0
        self._font = None
        self._title_font = None

    def handle_events(self, events):
        from .menu import MenuScene
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
                    return MenuScene(self.screen, mode=self.entries[self.cursor][1])
        return self

    def update(self, frame_dt: float) -> None:
        pass

    def draw(self) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont(None, 40)
            self._title_font = pygame.font.SysFont(None, 64)
        self.screen.fill((20, 30, 50))
        title = self._title_font.render("Blue Ball", True, (255, 255, 255))
        self.screen.blit(title, ((self.screen.get_width() - title.get_width()) // 2, 100))
        for i, (label, _) in enumerate(self.entries):
            color = (255, 220, 80) if i == self.cursor else (200, 200, 200)
            prefix = "> " if i == self.cursor else "  "
            surf = self._font.render(prefix + label, True, color)
            self.screen.blit(surf, (self.screen.get_width() // 2 - 90, 260 + i * 60))
        pygame.display.flip()
