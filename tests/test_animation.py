from blueball.render.animation import squash_stretch, Anim, palette_cycle


def test_rest_is_identity():
    sx, sy = squash_stretch(0.0, max_amount=0.3)
    assert abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6


def test_fast_rise_stretches_vertically():
    sx, sy = squash_stretch(-400.0, max_amount=0.3)
    assert sy > 1.0 and sx < 1.0           # taller, thinner
    assert abs(sx * sy - 1.0) < 0.05       # ~area-preserving


def test_fast_fall_squashes_vertically():
    # pymunk y-down: vy>0 is falling. A fast fall should squash the ball
    # short+wide (the inverse of the rising stretch), still ~area-preserving.
    sx, sy = squash_stretch(400.0, max_amount=0.3)
    assert sy < 1.0 and sx > 1.0           # shorter, wider
    assert abs(sx * sy - 1.0) < 0.05       # ~area-preserving


def test_anim_cycles():
    a = Anim(n_frames=3, fps=10)
    assert a.index(0.0) == 0
    assert a.index(0.15) == 1
    assert a.index(0.25) == 2
    assert a.index(0.35) == 0


def test_palette_cycle_wraps():
    assert palette_cycle([0, 1, 2], 0.0, hz=10) == 0
    assert palette_cycle([0, 1, 2], 0.15, hz=10) == 1
    assert palette_cycle([0, 1, 2], 0.35, hz=10) == 0
