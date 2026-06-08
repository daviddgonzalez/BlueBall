from blueball.render.parallax import layer_offset


def test_offset_wraps():
    assert layer_offset(0.0, 0.5, 100) == 0.0
    off = layer_offset(250.0, 0.5, 100)   # 250*0.5=125 -> wrap into (-100, 0]
    assert -100 < off <= 0


def test_factor_zero_is_static():
    assert layer_offset(9999.0, 0.0, 64) == 0.0


def test_factor_one_tracks_camera():
    # factor 1: offset is -(camera_x % tile_w)
    assert layer_offset(64.0, 1.0, 64) == 0.0
    assert layer_offset(96.0, 1.0, 64) == -32.0
