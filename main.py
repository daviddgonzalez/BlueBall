import os
import sys

import pygame

from blueball import config
from blueball.scenes.menu import MenuScene


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Blue Ball")
    clock = pygame.time.Clock()
    # Optional live FPS readout in the title bar (BLUEBALL_FPS=1). Off by default
    # so normal play is unaffected; lets you measure real window-present FPS,
    # which a headless profiler can't (display.flip is a no-op under dummy SDL).
    show_fps = bool(os.environ.get("BLUEBALL_FPS"))

    scene = MenuScene(screen)

    while scene is not None:
        events = pygame.event.get()
        scene = scene.handle_events(events)
        if scene is None:
            break
        frame_dt = clock.tick(config.TARGET_FPS) / 1000.0
        scene.update(frame_dt)
        scene.draw()
        if show_fps:
            pygame.display.set_caption(f"Blue Ball — {clock.get_fps():4.1f} fps")

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
