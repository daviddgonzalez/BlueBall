from dataclasses import replace
import blueball.config as config
from blueball.render.theme import (
    Theme, register_theme, get_theme, get_active_theme, _REGISTRY,
)


def test_pixel_theme_is_default():
    t = get_active_theme()
    assert isinstance(t, Theme)
    assert t.pixel_scale == config.PIXEL_SCALE
    assert "ball" in t.palette


def test_register_and_switch(monkeypatch):
    dummy = replace(get_theme("pixel"), palette={"ball": (1, 2, 3)})
    register_theme("dummy", dummy)
    monkeypatch.setattr(config, "ACTIVE_THEME", "dummy")
    try:
        assert get_active_theme().palette["ball"] == (1, 2, 3)
    finally:
        _REGISTRY.pop("dummy", None)


def test_unknown_theme_raises():
    import pytest
    with pytest.raises(KeyError):
        get_theme("does-not-exist")
