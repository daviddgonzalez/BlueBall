"""Entry script for the GA training scene.

Kept separate from main.py so the play loop entry point stays minimal.
Run with:  python train_main.py
"""

import sys
from pathlib import Path

import pygame

from blueball import config
from blueball.scenes.train import TrainScene


def main() -> int:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    level_path = Path(__file__).parent / "src" / "blueball" / "levels" / "tutorial_hill.json"
    scene = TrainScene(screen, level_path)

    while scene is not None:
        events = pygame.event.get()
        scene = scene.handle_events(events)
        if scene is None:
            break
        frame_dt = clock.tick(config.TARGET_FPS) / 1000.0
        scene.update(frame_dt)
        scene.draw()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
