"""Data-defined pixel sprites. A grid of palette-key chars baked once into a
cached pygame.Surface. Lowercase char = palette[key]; uppercase = palette[key+'_hi'];
'.' = transparent."""

from __future__ import annotations

import pygame

Color = tuple[int, int, int]


class SpriteDef:
    def __init__(self, grid, palette_key: str, frames: int = 1) -> None:
        # Single-frame: grid is list[str]. Multi-frame: grid is list[list[str]].
        # Validate the shape up front so a frames/grid mismatch fails loudly here
        # instead of as an out-of-range frame() lookup later.
        if frames > 1:
            if not (grid and isinstance(grid[0], list)):
                raise TypeError(
                    "multi-frame SpriteDef (frames>1) needs grid=list[list[str]]"
                )
            if len(grid) != frames:
                raise ValueError(
                    f"frames={frames} but grid has {len(grid)} frame(s)"
                )
            self._grids = grid
        else:
            self._grids = [grid]
        # Derive the real frame count from the grids so frame()'s modulo can
        # never index past the cache.
        self.frames_n = len(self._grids)
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
        # The sprite is baked ONCE against the palette of the first call and the
        # result cached; `palette` is ignored on later calls (sprites are static
        # per theme — a theme switch builds fresh SpriteDefs).
        if self._cache is None:
            if palette is None:
                raise ValueError("first bake needs a palette")
            self._cache = [self._bake_grid(g, palette) for g in self._grids]
        return self._cache[i % self.frames_n]
