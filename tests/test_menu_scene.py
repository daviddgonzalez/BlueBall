import pygame
import pytest

from blueball.scenes.menu import MenuScene
from blueball.scenes.play import PlayScene


@pytest.fixture(autouse=True)
def _init_pygame():
    pygame.init()
    pygame.display.set_mode((800, 600))
    yield
    pygame.quit()


def _key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, {"key": key})


def test_menu_cursor_moves_down():
    m = MenuScene(pygame.display.get_surface())
    assert m.cursor == 0
    m.handle_events([_key_event(pygame.K_DOWN)])
    assert m.cursor == 1
    # Wrap-clamp at end
    for _ in range(20):
        m.handle_events([_key_event(pygame.K_DOWN)])
    assert m.cursor == len(m.entries) - 1


def test_menu_enter_on_normal_level_returns_playscene():
    m = MenuScene(pygame.display.get_surface())
    m.cursor = 0  # Tutorial Hill
    result = m.handle_events([_key_event(pygame.K_RETURN)])
    assert isinstance(result, PlayScene)
    assert result.level_path is not None


def test_menu_enter_on_infinite_run_returns_streaming_playscene():
    """Infinite Run uses streaming — PlayScene materializes chunks lazily.
    The MenuScene hands over only the level metadata (no chunks list) and
    PlayScene's sampler iterator emits them as the player advances.
    """
    m = MenuScene(pygame.display.get_surface())
    # Infinite Run is the last entry
    m.cursor = len(m.entries) - 1
    result = m.handle_events([_key_event(pygame.K_RETURN)])
    assert isinstance(result, PlayScene)
    assert result.level_data is not None
    assert result.sampler_seed is not None
    assert result._streaming is True
    # Empty chunks list — PlayScene will stream them in
    assert result.level_data["chunks"] == []
    # Initial buffer of chunks has been built into the world
    assert len(result._built_chunks) > 0


def test_menu_esc_returns_none():
    m = MenuScene(pygame.display.get_surface())
    result = m.handle_events([_key_event(pygame.K_ESCAPE)])
    assert result is None
