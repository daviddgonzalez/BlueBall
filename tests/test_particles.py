from blueball.render.particles import ParticleSystem


def test_cap_enforced():
    ps = ParticleSystem(cap=10)
    ps.emit("dust", (0, 0), n=50)
    assert len(ps) <= 10


def test_particles_age_out():
    ps = ParticleSystem(cap=100)
    ps.emit("burst", (0, 0), n=5)
    for _ in range(1000):
        ps.update(0.1)
    assert len(ps) == 0
