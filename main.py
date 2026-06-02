import sys

import pygame

from blueball import config
from blueball.scenes.menu import MenuScene


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball")
    clock = pygame.time.Clock()

    scene = MenuScene(screen)

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
