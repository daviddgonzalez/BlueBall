"""Simple JSON-backed save file storing the set of unlocked ability names."""

from __future__ import annotations

import json
import os
from pathlib import Path

SAVE_PATH = Path(os.environ.get(
    "BLUEBALL_SAVE_PATH",
    str(Path.home() / ".blueball" / "save.json"),
))


def load() -> set[str]:
    if not SAVE_PATH.exists():
        return set()
    data = json.loads(SAVE_PATH.read_text())
    return set(data.get("unlocked_abilities", []))


def add_ability(name: str) -> None:
    abilities = load()
    if name in abilities:
        return
    abilities.add(name)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps(
        {"unlocked_abilities": sorted(abilities)}, indent=2,
    ))
