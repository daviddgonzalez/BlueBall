import os

import pygame
import pytest


@pytest.fixture
def screen():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.display.init()
    pygame.font.init()
    s = pygame.display.set_mode((1280, 720))
    yield s
    pygame.display.quit()


def test_playscene_drains_sound_queue(screen):
    from blueball.ai.episodes import resolve_level_paths
    from blueball.scenes.play import PlayScene

    play = PlayScene(screen, level_path=resolve_level_paths(["tutorial_hill"])[0])

    played = []

    class _Spy:
        def play(self, name):
            played.append(name)

    play._sound = _Spy()
    play.world.sound_events.append("whoosh")
    play.update(1 / 60.0)

    assert "whoosh" in played
    assert play.world.sound_events == []
