"""RenderCore — theme-agnostic engine: a low-res virtual surface that is
nearest-neighbor-upscaled to the window. Owns the screen-shake offset that the
renderer adds into its world->surface transform (populated by a later task;
(0, 0) until then)."""

from __future__ import annotations

import pygame

from .. import config


class RenderCore:
    def __init__(self, window: pygame.Surface, pixel_scale: int | None = None) -> None:
        self.window = window
        self.scale = pixel_scale if pixel_scale is not None else config.PIXEL_SCALE
        ww, wh = window.get_size()
        if ww % self.scale or wh % self.scale:
            raise ValueError(
                f"window {ww}x{wh} not divisible by pixel_scale {self.scale}"
            )
        self.vw, self.vh = ww // self.scale, wh // self.scale
        surf = pygame.Surface((self.vw, self.vh))
        # .convert() requires a display; guard so headless construction works.
        self.surface = surf.convert() if pygame.display.get_surface() else surf
        self.shake_offset: tuple[float, float] = (0.0, 0.0)

    def present(self, flip: bool = True) -> None:
        """Upscale the virtual surface onto the window (nearest-neighbor)."""
        pygame.transform.scale(self.surface, self.window.get_size(), self.window)
        if flip:
            pygame.display.flip()
