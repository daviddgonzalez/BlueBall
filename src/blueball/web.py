"""Browser (pygbag) entry: the play loop from cli.cmd_play, made async.

Kept out of cli.py on purpose: cli.py imports multiprocessing and the training
stack at module level, and the browser's Python can't load all of that. This
module imports only what playing the game needs.
"""

from __future__ import annotations

import asyncio

import pygame

from . import config


async def play_web() -> None:
    from .scenes.mode_select import ModeSelectScene

    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    scene = ModeSelectScene(screen)
    while scene is not None:
        scene = scene.handle_events(pygame.event.get())
        if scene is None:
            break
        scene.update(clock.tick(config.TARGET_FPS) / 1000.0)
        scene.draw()
        await asyncio.sleep(0)  # REQUIRED for pygbag: yields the frame to the browser
    pygame.quit()
