import numpy as np
import pytest

from creativity3d.scoring import (
    breadth,
    build_fingerprint,
    outlier_flags,
    percentile,
    rarity_per_item,
    semantic_distances,
)


def unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_semantic_distances_basic_and_clip():
    stim = unit(1, 0, 0)
    vecs = np.array([unit(1, 0, 0), unit(0, 1, 0), unit(-1, 0, 0)])
    d = semantic_distances(stim, vecs)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [0.0, 1.0, 2.0], atol=1e-9)
    assert d.min() >= 0 and d.max() <= 2


def test_semantic_distances_empty():
    assert semantic_distances(unit(1, 0), np.zeros((0, 2))).shape == (0,)


def test_outlier_fallback_when_small_pool():
    flags = outlier_flags(np.array([0.5, 0.95]), np.array([0.1] * 19))
    assert flags.tolist() == [False, True]


def test_outlier_iqr():
    pool = np.linspace(0.3, 0.5, 21)  # Q1=0.35, Q3=0.45 → 門檻 0.6
    flags = outlier_flags(np.array([0.59, 0.61]), pool, fallback=0.1)
    assert flags.tolist() == [False, True]


def test_breadth():
    assert breadth(np.array([unit(1, 0, 0), unit(0, 1, 0)])) is None
    b = breadth(np.array([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]))
    assert b == pytest.approx(1.0)
    same = breadth(np.array([unit(1, 0, 0)] * 3))
    assert same == pytest.approx(0.0, abs=1e-9)


def test_rarity_per_item():
    fruit, newton = unit(1, 0, 0), unit(0, 1, 0)
    pool = {f"u{i}": np.array([fruit]) for i in range(8)}
    pool["u8"] = np.array([newton, newton])  # 同一人寫兩次只算一次
    pool["u9"] = np.array([unit(0, 0, 1)])
    r = rarity_per_item(np.array([fruit, newton]), pool, threshold=0.85, min_users=5)
    assert r == pytest.approx([0.2, 0.9])


def test_rarity_insufficient_users():
    pool = {"a": np.array([unit(1, 0)])}
    assert rarity_per_item(np.array([unit(1, 0), unit(0, 1)]), pool, 0.85, 5) == [None, None]


def test_percentile():
    assert percentile(0.6, [0.2, 0.4, 0.6, 0.8]) == pytest.approx(62.5)
    assert percentile(1.0, [0.2, 0.4]) == 100.0
    assert percentile(0.0, [0.2, 0.4]) == 0.0
    with pytest.raises(ValueError):
        percentile(1.0, [])


def test_build_fingerprint_raw_values():
    valid = np.array([unit(1, 0, 0), unit(0, 1, 0)])
    fp = build_fingerprint(
        effective_distances=[0.6, 0.1, 0.0],
        weights=[1.0, 1.0, 0.0],
        valid_vecs=valid,
        rarities=[0.9, None, 0.5],
        population=[],
        min_users_for_percentile=20,
    )
    assert fp.distance.raw == pytest.approx(0.7 / 3)
    assert fp.appropriateness.raw == pytest.approx(2 / 3)
    assert fp.fluency.raw == 2
    assert fp.rarity.raw == pytest.approx(0.7)
    assert fp.breadth.raw is None
    assert all(getattr(fp, k).percentile is None for k in
               ("distance", "breadth", "rarity", "appropriateness", "fluency"))


def test_build_fingerprint_all_rarity_none():
    fp = build_fingerprint(effective_distances=[0.5], weights=[1.0], valid_vecs=np.array([unit(1, 0)]),
                           rarities=[None], population=[], min_users_for_percentile=1)
    assert fp.rarity.raw is None and fp.rarity.percentile is None


def test_build_fingerprint_percentiles():
    population = [
        {"distance": 0.1 * i, "breadth": None, "rarity": 0.5, "appropriateness": 1.0, "fluency": float(i)}
        for i in range(4)
    ]
    fp = build_fingerprint(
        effective_distances=[0.25], weights=[1.0], valid_vecs=np.array([unit(1, 0)]),
        rarities=[0.5], population=population, min_users_for_percentile=4,
    )
    assert fp.distance.percentile == pytest.approx(75.0)  # 0,0.1,0.2 < 0.25
    assert fp.breadth.percentile is None  # 自己是 None
    assert fp.rarity.percentile == pytest.approx(50.0)  # 全部相等 → 一半
    assert fp.fluency.percentile == pytest.approx(37.5)  # 0 小於、1 等於 → 1.5/4

    fp2 = build_fingerprint(
        effective_distances=[0.25], weights=[1.0], valid_vecs=np.array([unit(1, 0)]),
        rarities=[0.5], population=population, min_users_for_percentile=5,
    )
    assert fp2.distance.percentile is None


def test_build_fingerprint_skips_none_in_population():
    population = [{"distance": None, "breadth": 0.5, "rarity": None, "appropriateness": 1.0, "fluency": 1.0},
                  {"distance": 0.1, "breadth": 0.5, "rarity": None, "appropriateness": 1.0, "fluency": 1.0}]
    valid = np.array([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    fp = build_fingerprint(effective_distances=[0.5], weights=[1.0], valid_vecs=valid,
                           rarities=[0.3], population=population, min_users_for_percentile=2)
    assert fp.distance.percentile == 100.0  # 只比較 0.1
    assert fp.rarity.percentile is None  # population 全 None
    assert fp.breadth.percentile == 100.0
