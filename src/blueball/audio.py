"""SoundManager — loads + plays the game's SFX via pygame.mixer, degrading to a
silent no-op when audio is unavailable (CI / headless / no device / disabled)."""
from __future__ import annotations

import os
from pathlib import Path

from . import config

_SFX_DIR = Path(__file__).resolve().parent / "assets" / "sfx"
_SOUNDS = ("whoosh", "spring", "key", "fanfare")


class SoundManager:
    def __init__(self) -> None:
        self.enabled = False
        self._sounds = {}
        if not config.AUDIO_ENABLED or os.environ.get("BLUEBALL_NO_AUDIO"):
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            for name in _SOUNDS:
                path = _SFX_DIR / f"{name}.wav"
                if path.exists():
                    self._sounds[name] = pygame.mixer.Sound(str(path))
            self.enabled = True
        except Exception:
            # No audio device, mixer failure, etc. -> stay silent, never crash.
            self.enabled = False
            self._sounds = {}

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            try:
                snd.play()
            except Exception:
                pass
