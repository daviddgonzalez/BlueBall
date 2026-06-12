import os

import pygame
import pytest


@pytest.fixture
def screen():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.font.init()
    s = pygame.display.set_mode((1280, 720))
    yield s
    pygame.display.quit()


def _key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def test_mode_select_routes_to_menu_mode(screen):
    from blueball.scenes.mode_select import ModeSelectScene
    from blueball.scenes.menu import MenuScene
    race = ModeSelectScene(screen)
    race.cursor = 1
    nxt = race.handle_events([_key(pygame.K_RETURN)])
    assert isinstance(nxt, MenuScene) and nxt.mode == "race"

    single = ModeSelectScene(screen)
    single.cursor = 0
    nxt = single.handle_events([_key(pygame.K_RETURN)])
    assert isinstance(nxt, MenuScene) and nxt.mode == "single"

    assert ModeSelectScene(screen).handle_events([_key(pygame.K_ESCAPE)]) is None


def test_race_menu_hides_infinite_run(screen):
    from blueball.scenes.menu import MenuScene
    race_labels = [lbl for lbl, _ in MenuScene(screen, mode="race").entries]
    single_labels = [lbl for lbl, _ in MenuScene(screen, mode="single").entries]
    assert "Infinite Run" in single_labels
    assert "Infinite Run" not in race_labels


def test_race_level_select_builds_ghost(screen):
    from blueball.scenes.menu import MenuScene
    from blueball.scenes.play import PlayScene
    from blueball.scenes.ghost import GhostRunner
    menu = MenuScene(screen, mode="race")
    menu.cursor = 0
    play = menu.handle_events([_key(pygame.K_RETURN)])
    assert isinstance(play, PlayScene)
    assert isinstance(play._ghost, GhostRunner)


def test_playscene_ghost_updates_and_draws(screen):
    import numpy as np
    from blueball.ai.episodes import resolve_level_paths
    from blueball.scenes.ghost import GhostRunner
    from blueball.scenes.play import PlayScene
    track = np.array([[80, 540, 0], [120, 540, 0.5], [160, 540, 1.0]], dtype=np.float32)
    play = PlayScene(screen, level_path=resolve_level_paths(["tutorial_hill"])[0],
                     ghost=GhostRunner(track), mode="race")
    for _ in range(5):
        play.update(1 / 60.0)
        play.draw()
