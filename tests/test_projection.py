import numpy as np
import pytest

from creativity3d.projection import Projector


def norm3(p):
    return float(np.sqrt(p.x**2 + p.y**2 + p.z**2))


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    dim = 16
    stim = rng.standard_normal(dim)
    stim /= np.linalg.norm(stim)
    refs = rng.standard_normal((30, dim))
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    return stim, refs


def test_fit_components_orthonormal(data):
    stim, refs = data
    proj = Projector.fit(stim, refs)
    assert proj.components.shape == (3, stim.size)
    np.testing.assert_allclose(proj.components @ proj.components.T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("distance", [0.0, 0.1, 0.6, 1.37, 2.0])
def test_project_radius_equals_distance(data, distance):
    stim, refs = data
    proj = Projector.fit(stim, refs)
    for v in refs[:10]:
        assert norm3(proj.project(v, distance)) == pytest.approx(distance, abs=1e-9)


def test_few_references_uses_fixed_seed_basis(data):
    stim, refs = data
    a = Projector.fit(stim, refs[:2])
    b = Projector.fit(stim, refs[:2])
    np.testing.assert_allclose(a.components, b.components)
    np.testing.assert_allclose(a.components @ a.components.T, np.eye(3), atol=1e-9)
    assert norm3(a.project(refs[5], 0.7)) == pytest.approx(0.7)
    empty = Projector.fit(stim, np.zeros((0, stim.size)))
    np.testing.assert_allclose(empty.components @ empty.components.T, np.eye(3), atol=1e-9)


def test_rank_deficient_references_completed(data):
    stim, refs = data
    dup = np.vstack([refs[0]] * 5)  # 秩 1
    proj = Projector.fit(stim, dup)
    np.testing.assert_allclose(proj.components @ proj.components.T, np.eye(3), atol=1e-9)


def test_zero_residual_gets_fixed_direction(data):
    stim, refs = data
    proj = Projector.fit(stim, refs)
    p = proj.project(stim, 0.5)
    assert (p.x, p.y, p.z) == pytest.approx((0.0, 0.0, 0.5))


def test_direction_follows_pca(data):
    stim, _ = data
    dim = stim.size
    axis = np.zeros(dim)
    axis[0] = 1.0
    refs = np.array([stim + t * axis for t in (-1, -0.5, 0.5, 1, 2)]) + 1e-3 * np.random.default_rng(1).standard_normal((5, dim))
    proj = Projector.fit(stim, refs)
    p = proj.project(stim + 0.3 * axis, 1.0)
    assert abs(p.x) == pytest.approx(1.0, abs=1e-2)  # 主要變異方向 → 第一軸


def test_dict_roundtrip(data):
    stim, refs = data
    proj = Projector.fit(stim, refs)
    again = Projector.from_dict(proj.to_dict())
    p1, p2 = proj.project(refs[3], 0.8), again.project(refs[3], 0.8)
    assert (p1.x, p1.y, p1.z) == pytest.approx((p2.x, p2.y, p2.z))
