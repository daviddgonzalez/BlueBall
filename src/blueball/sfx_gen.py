"""Procedurally synthesize the game's SFX as original (non-copyrighted) WAVs.

Run `python -m blueball.sfx_gen` to (re)generate the committed assets under
assets/sfx/. Deterministic (seeded), so regeneration is reproducible."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
SOUND_NAMES = ("whoosh", "spring", "key", "fanfare")


def _env(n, attack=0.005, release=0.1):
    """Linear attack/release amplitude envelope of length n samples."""
    a = max(1, int(SR * attack))
    r = max(1, int(SR * release))
    e = np.ones(n)
    e[:a] = np.linspace(0.0, 1.0, a)
    e[n - r:] = np.linspace(1.0, 0.0, r)
    return e


def _tone(freq, dur, attack=0.005, release=0.05):
    n = int(SR * dur)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t) * _env(n, attack, release)


def _triangle(freq, dur, attack=0.005, release=0.08):
    n = int(SR * dur)
    t = np.arange(n) / SR
    tri = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    return tri * _env(n, attack, release)


def _whoosh():
    n = int(SR * 0.35)
    rng = np.random.default_rng(1)
    brown = np.cumsum(rng.standard_normal(n))  # brown-ish noise
    brown /= np.max(np.abs(brown)) + 1e-9
    return brown * _env(n, 0.005, 0.25) * 0.7


def _spring():
    n = int(SR * 0.30)
    f = np.linspace(300.0, 900.0, n)              # upward pitch glide -> "boing"
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase) * _env(n, 0.005, 0.2) * 0.6


def _key():
    return np.concatenate([_tone(1200, 0.06) * 0.5,   # tick
                           _tone(500, 0.12) * 0.6])    # thunk


def _fanfare():
    notes = [523.25, 659.25, 783.99, 1046.50]         # C5 E5 G5 C6
    return np.concatenate([_triangle(f, 0.14) * 0.5 for f in notes])


_SYNTH = {"whoosh": _whoosh, "spring": _spring, "key": _key, "fanfare": _fanfare}


def _write_wav(path, samples):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate_sfx(out_dir) -> None:
    """Write the four SFX WAVs into out_dir (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SOUND_NAMES:
        _write_wav(out / f"{name}.wav", _SYNTH[name]())


if __name__ == "__main__":
    generate_sfx(Path(__file__).resolve().parent / "assets" / "sfx")
    print("wrote SFX to assets/sfx/")
