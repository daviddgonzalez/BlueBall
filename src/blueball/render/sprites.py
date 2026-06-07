"""Data-defined pixel sprites. A grid of palette-key chars baked once into a
cached pygame.Surface. Lowercase char = palette[key]; uppercase = palette[key+'_hi'];
'.' = transparent."""

from __future__ import annotations

import pygame

Color = tuple[int, int, int]


class SpriteDef:
    def __init__(self, grid, palette_key: str, frames: int = 1) -> None:
        # Single-frame: grid is list[str]. Multi-frame: grid is list[list[str]].
        self.frames_n = frames
        self._grids = grid if frames > 1 else [grid]
        self.palette_key = palette_key
        self._cache: list[pygame.Surface] | None = None

    def _resolve(self, ch: str, palette):
        if ch == ".":
            return None
        if ch.isupper():
            return palette.get(f"{self.palette_key}_hi", palette[self.palette_key])
        return palette[self.palette_key]

    def _bake_grid(self, rows, palette) -> pygame.Surface:
        h = len(rows)
        w = max(len(r) for r in rows)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                col = self._resolve(ch, palette)
                if col is not None:
                    surf.set_at((x, y), col)
        return surf

    def bake(self, palette) -> pygame.Surface:
        return self.frame(0, palette)

    def frame(self, i: int, palette=None) -> pygame.Surface:
        if self._cache is None:
            if palette is None:
                raise ValueError("first bake needs a palette")
            self._cache = [self._bake_grid(g, palette) for g in self._grids]
        return self._cache[i % self.frames_n]
