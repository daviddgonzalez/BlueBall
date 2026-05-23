def test_package_imports():
    import blueball
    from blueball import config
    assert blueball.__version__ == "0.1.0"
    assert config.PHYS_HZ == 120
    assert abs(config.PHYS_DT - (1.0 / 120)) < 1e-12
