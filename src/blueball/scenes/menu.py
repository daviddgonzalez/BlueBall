"""MenuScene — level select with 5 entries."""

from __future__ import annotations

import time
from pathlib import Path

import pygame

from .base import Scene
from .play import PlayScene


class MenuScene(Scene):
    INFINITE_RUN = "__infinite__"

    def __init__(self, screen: pygame.Surface, mode: str = "single") -> None:
        self.screen = screen
        self.mode = mode
        levels_dir = Path(__file__).parent.parent / "levels"
        all_entries: list[tuple[str, object]] = [
            ("Tutorial Hill", levels_dir / "tutorial_hill.json"),
            ("Vertical Climb", levels_dir / "vertical_climb.json"),
            ("Speed Run", levels_dir / "speed_run.json"),
            ("Maze", levels_dir / "maze.json"),
            ("Lava Rising", levels_dir / "lava_rising.json"),
            ("Infinite Run", self.INFINITE_RUN),
        ]
        if mode == "race":
            self.entries = [(lbl, tgt) for lbl, tgt in all_entries if tgt != self.INFINITE_RUN]
        else:
            self.entries = all_entries
        self.cursor: int = 0
        self._font = None
        self._title_font = None

    def _start_race(self, level_path):
        """Record the level's ghost (if a genome is bundled) and launch the race.
        Falls back to a ghostless PlayScene when no genome is available."""
        import numpy as np
        from .. import config
        from .ghost import GhostRunner, record_ghost_track
        from ..abilities import Ability
        ghost = None
        genome_path = config.resolve_race_ghost_genome(Path(level_path).stem)
        if genome_path is not None:
            track = record_ghost_track(np.load(genome_path), level_path,
                                       abilities=config.RACE_GHOST_ABILITIES)
            ghost = GhostRunner(track)
        return PlayScene(self.screen, level_path=level_path, ghost=ghost,
                         mode="race",
                         extra_abilities={Ability(a) for a in config.RACE_GHOST_ABILITIES})

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    from .mode_select import ModeSelectScene
                    return ModeSelectScene(self.screen)
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.cursor = max(0, self.cursor - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.cursor = min(len(self.entries) - 1, self.cursor + 1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    _, target = self.entries[self.cursor]
                    if target == self.INFINITE_RUN:
                        # Streaming infinite: PlayScene builds chunks as the
                        # player approaches them. We only pass metadata
                        # (palette, spawn, name); the chunk list is empty.
                        # sampler_seed signals streaming mode.
                        seed = int(time.time() * 1000) & 0xFFFFFFFF
                        level_data = {
                            "name": f"Infinite Run (seed={seed})",
                            "background": "#202028",
                            "ground": "#666c70",
                            "spawn": [80, 540],
                            "chunks": [],
                        }
                        return PlayScene(self.screen, level_data=level_data, sampler_seed=seed,
                                         mode=self.mode)
                    if self.mode == "race":
                        return self._start_race(target)
                    return PlayScene(self.screen, level_path=target, mode="single")
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
