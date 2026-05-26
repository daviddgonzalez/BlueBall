import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_save(monkeypatch, tmp_path):
    """Redirect BLUEBALL_SAVE_PATH at a tmp file and force the save module to re-read it."""
    save_path = tmp_path / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(save_path))
    # Force a fresh import so the SAVE_PATH module-level constant picks up the env var
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    return save_path, save_mod


def test_load_returns_empty_set_when_file_missing(tmp_save):
    _path, save_mod = tmp_save
    assert save_mod.load() == set()


def test_add_ability_creates_file_and_persists(tmp_save):
    path, save_mod = tmp_save
    save_mod.add_ability("double_jump")
    assert path.exists()
    assert save_mod.load() == {"double_jump"}
    data = json.loads(path.read_text())
    assert data == {"unlocked_abilities": ["double_jump"]}


def test_add_ability_is_idempotent(tmp_save):
    _path, save_mod = tmp_save
    save_mod.add_ability("double_jump")
    save_mod.add_ability("double_jump")
    assert save_mod.load() == {"double_jump"}


def test_add_ability_preserves_existing_unlocks(tmp_save):
    path, save_mod = tmp_save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unlocked_abilities": ["wall_jump"]}))
    save_mod.add_ability("double_jump")
    assert save_mod.load() == {"double_jump", "wall_jump"}
    # Stored sorted
    assert json.loads(path.read_text())["unlocked_abilities"] == ["double_jump", "wall_jump"]


def test_add_ability_creates_parent_directory(tmp_save, tmp_path, monkeypatch):
    nested = tmp_path / "nested" / "dir" / "save.json"
    monkeypatch.setenv("BLUEBALL_SAVE_PATH", str(nested))
    import importlib
    import blueball.save as save_mod
    importlib.reload(save_mod)
    save_mod.add_ability("double_jump")
    assert nested.exists()
