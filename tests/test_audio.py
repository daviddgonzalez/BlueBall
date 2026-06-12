def test_sound_manager_constructs_and_plays(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("BLUEBALL_NO_AUDIO", raising=False)
    from blueball.audio import SoundManager
    sm = SoundManager()
    sm.play("whoosh")        # must not raise
    sm.play("nonexistent")   # must not raise


def test_sound_manager_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BLUEBALL_NO_AUDIO", "1")
    from blueball.audio import SoundManager
    sm = SoundManager()
    assert sm.enabled is False
    sm.play("whoosh")        # no-op, must not raise
