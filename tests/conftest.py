"""Pytest configuration.

The project's shared .venv editable-installs blueball at the MAIN repo's
src/blueball directory (see `direct_url.json` in site-packages). When this
conftest runs from inside a worktree on a feature branch, the worktree's
own modules — including any new package like `blueball.ai` — are invisible
to pytest unless we prepend the worktree's `src/` to `sys.path` so it
shadows the editable install.

After the branch merges back to main, this prepend becomes a harmless
no-op: the worktree's `src/` and the main repo's `src/` are the same
directory. Until then, be aware that editing files in BOTH the main repo
checkout and this worktree without re-running tests in each will produce
divergent results — pytest will only see whichever `src/` happens to be
at sys.path[0] in the current invocation.

Implementation detail: `resolve()` runs once at conftest load time, and
the `not in sys.path` guard prevents duplicate entries on re-import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_WORKTREE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))

# Ensure pygame can operate in headless CI / WSL environments.  Setting
# SDL_VIDEODRIVER=dummy before pygame is imported (which happens lazily inside
# HumanAgent.act) lets pygame.key.get_pressed() work without a real display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="session", autouse=True)
def _pygame_headless_init():
    """Initialize pygame once per test session using the dummy video driver.

    Any test that instantiates HumanAgent and drives physics (substep / step)
    will invoke pygame.key.get_pressed(); without a prior pygame.init() that
    call raises ``pygame.error: video system not initialized``.  This fixture
    runs automatically for every session so individual test modules don't need
    to repeat the boilerplate.
    """
    import pygame  # noqa: PLC0415
    pygame.init()
    yield
    pygame.quit()
