"""Pytest configuration for the ai-scaffolding worktree.

The shared venv's editable install points at the main repo src/.
Insert this worktree's src/ at the front of sys.path so modules added
here (e.g. blueball.ai) take precedence when running tests in the worktree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKTREE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))
