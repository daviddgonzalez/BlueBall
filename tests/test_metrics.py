from blueball.ai.metrics import behavior_metrics


def test_basic_rates():
    m = behavior_metrics(jumps_fired=10, airborne_steps=300, steps=600,
                         progress_x=1000.0)
    assert abs(m["jumps_per_100px"] - 1.0) < 1e-6      # 10 jumps / 1000px * 100
    assert abs(m["airtime_pct"] - 50.0) < 1e-6         # 300/600


def test_zero_denominators_safe():
    m = behavior_metrics(jumps_fired=0, airborne_steps=0, steps=0,
                         progress_x=0.0)
    assert m["jumps_per_100px"] == 0.0
    assert m["airtime_pct"] == 0.0
