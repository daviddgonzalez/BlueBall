"""MenuScene stub — minimal placeholder; Task 26 replaces this with the real implementation."""

from __future__ import annotations

import pygame

from .base import Scene


class MenuScene(Scene):
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen

    def handle_events(self, events):
        return self

    def update(self, frame_dt: float) -> None:
        pass

    def draw(self) -> None:
        pass
