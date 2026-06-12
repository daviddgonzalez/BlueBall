import wave

from blueball.sfx_gen import generate_sfx, SOUND_NAMES


def test_generate_sfx_writes_valid_wavs(tmp_path):
    generate_sfx(tmp_path)
    for name in SOUND_NAMES:
        p = tmp_path / f"{name}.wav"
        assert p.exists(), f"{name}.wav not written"
        with wave.open(str(p), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() > 0


def test_committed_assets_present():
    from pathlib import Path
    import blueball
    sfx = Path(blueball.__file__).parent / "assets" / "sfx"
    for name in SOUND_NAMES:
        assert (sfx / f"{name}.wav").exists(), f"committed {name}.wav missing"
