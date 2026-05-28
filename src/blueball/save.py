"""Simple JSON-backed save file for the local user: unlocked abilities and the
best Infinite Run score. All readers/writers preserve unrelated fields so, e.g.,
unlocking an ability never clobbers the high score."""

from __future__ import annotations

import json
import os
from pathlib import Path

SAVE_PATH = Path(os.environ.get(
    "BLUEBALL_SAVE_PATH",
    str(Path.home() / ".blueball" / "save.json"),
))


def _read() -> dict:
    if not SAVE_PATH.exists():
        return {}
    try:
        data = json.loads(SAVE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict) -> None:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps(data, indent=2))


def load() -> set[str]:
    return set(_read().get("unlocked_abilities", []))


def add_ability(name: str) -> None:
    data = _read()
    abilities = set(data.get("unlocked_abilities", []))
    if name in abilities:
        return
    abilities.add(name)
    data["unlocked_abilities"] = sorted(abilities)
    _write(data)


def get_best_score() -> int:
    return int(_read().get("best_score", 0))


def set_best_score(score: int) -> None:
    """Persist `score` as the best only if it beats the stored best."""
    data = _read()
    if int(score) > int(data.get("best_score", 0)):
        data["best_score"] = int(score)
        _write(data)
