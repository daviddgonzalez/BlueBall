import pytest

from blueball.abilities import Ability
from blueball.levels.segments import SegmentSampler

ALL = frozenset({Ability.DOUBLE_JUMP})


def _names(sampler, n):
    return [type(sampler.emit_next()).__name__ for _ in range(n)]


def test_deterministic_for_a_fixed_seed():
    assert _names(SegmentSampler(123, ALL), 30) == _names(SegmentSampler(123, ALL), 30)


def test_difficulty_ramps_with_depth():
    s = SegmentSampler(7, ALL)
    early = [s.emit_next().tier for _ in range(5)]
    for _ in range(40):
        s.emit_next()
    late = [s.emit_next().tier for _ in range(10)]
    assert sum(early) / len(early) < sum(late) / len(late)


def test_ability_filter_excludes_doublejump_tiers_when_not_granted():
    s = SegmentSampler(1, frozenset())  # single jump only
    tiers = {s.emit_next().tier for _ in range(60)}
    assert 2 not in tiers and 3 not in tiers


def test_no_immediate_duplicate_template_type():
    s = SegmentSampler(99, ALL)
    seq = [type(s.emit_next()).__name__ for _ in range(50)]
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def test_empty_pool_raises():
    # Future-guard: if every template required an ability, an empty grant has no
    # eligible templates. Monkeypatch the registry to an all-DJ pool.
    import blueball.levels.segments as seg
    only_dj = [t for t in seg.SEGMENT_TEMPLATES if t.min_abilities]
    saved = seg.SEGMENT_TEMPLATES
    seg.SEGMENT_TEMPLATES = only_dj
    try:
        with pytest.raises(ValueError):
            SegmentSampler(0, frozenset())
    finally:
        seg.SEGMENT_TEMPLATES = saved
