from datetime import datetime, timezone

import numpy as np
import pytest

from creativity3d.projection import Projector
from creativity3d.schemas import (
    AnalysisResponse,
    DimensionScore,
    Fingerprint,
    Point3D,
    PoolStats,
    RuleFlag,
    ScoredAssociation,
    Stimulus,
    StimulusSource,
    Verdict,
)
from creativity3d.store import Store

DIM = 8
S1, S2, S3 = "session-aaaa", "session-bbbb", "session-cccc"


def vec(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    v[(i + 1) % DIM] = 0.3
    return v / np.linalg.norm(v)


STIM_VEC = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sub" / "test.db")
    s.ensure_presets([
        Stimulus(id="s09", word="蘋果", source=StimulusSource.preset, abstraction_level=9, source_note="MRC"),
        Stimulus(id="s10", word="牛奶", source=StimulusSource.preset, abstraction_level=10),
    ])
    return s


def make_response(items, distance_raw=0.4):
    """items: list of (text, weight, distance)"""
    assocs = [
        ScoredAssociation(
            index=i, text=t, rule_flag=RuleFlag.ok,
            verdict=Verdict.reasonable if w == 1 else (Verdict.borderline if w else Verdict.unreasonable),
            weight=w, distance=d, effective_distance=d * w, outlier=False,
            position=Point3D(x=0, y=0, z=d),
        )
        for i, (t, w, d) in enumerate(items)
    ]
    fp = Fingerprint(
        distance=DimensionScore(raw=distance_raw), breadth=DimensionScore(raw=None),
        rarity=DimensionScore(raw=None), appropriateness=DimensionScore(raw=1.0),
        fluency=DimensionScore(raw=float(sum(1 for _, w, _ in items if w > 0))),
    )
    return AnalysisResponse(
        submission_id="pending",
        stimulus=Stimulus(id="s09", word="蘋果", source=StimulusSource.preset),
        associations=assocs, fingerprint=fp, interpretation=None,
        pool=PoolStats(stimulus_id="s09", human_users=0, seed_points=0, enough_for_percentile=False),
        cloud=[], created_at=datetime.now(timezone.utc),
    )


def submit(store, session, items, vecs, share=True, distance_raw=0.4):
    sid = store.save_submission(stimulus_id="s09", session_id=session,
                                response=make_response(items, distance_raw), vecs=np.array(vecs))
    if share:
        assert store.share_submission(sid, session)
    return sid


# --- 刺激詞 ---------------------------------------------------------------

def test_presets_idempotent(store):
    store.ensure_presets([Stimulus(id="s09", word="蘋果", source=StimulusSource.preset, abstraction_level=9)])
    stims = store.list_stimuli(None)
    assert [s.id for s in stims] == ["s09", "s10"]
    assert all(s.source == StimulusSource.preset and s.ready for s in stims)


def test_create_and_visibility(store):
    st = store.create_stimulus("海浪", S1)
    assert st.id.startswith("u") and len(st.id) == 9
    int(st.id[1:], 16)
    assert st.source == StimulusSource.mine and not st.ready and st.abstraction_level is None

    assert st.id in [s.id for s in store.list_stimuli(S1)]
    assert st.id not in [s.id for s in store.list_stimuli(S2)]
    assert st.id not in [s.id for s in store.list_stimuli(None)]
    assert store.get_stimulus(st.id, S2) is None
    assert store.get_stimulus(st.id, S1).source == StimulusSource.mine

    store.mark_ready(st.id)
    assert store.get_stimulus(st.id, S1).ready

    assert store.publish_stimulus(st.id, S2) is None  # 非建立者
    assert store.publish_stimulus("s09", S1) is None  # 預設詞
    pub = store.publish_stimulus(st.id, S1)
    assert pub.source == StimulusSource.community
    assert store.get_stimulus(st.id, S2).source == StimulusSource.community
    ids = [s.id for s in store.list_stimuli(S2)]
    assert ids == ["s09", "s10", st.id]


def test_find_by_word(store):
    assert store.find_stimulus_by_word("蘋果").id == "s09"
    assert store.find_stimulus_by_word("海浪") is None
    st = store.create_stimulus("海浪", S1)
    assert store.find_stimulus_by_word("海浪").id == st.id


def test_count_created_today(store):
    assert store.count_created_today(S1) == 0
    store.create_stimulus("海浪", S1)
    store.create_stimulus("星星", S1)
    store.create_stimulus("月亮", S2)
    assert store.count_created_today(S1) == 2


def test_community_sorted_by_response_count(store):
    a = store.create_stimulus("海浪", S1)
    b = store.create_stimulus("星星", S1)
    store.publish_stimulus(a.id, S1)
    store.publish_stimulus(b.id, S1)
    for sess in (S2, S3):
        sid = store.save_submission(stimulus_id=b.id, session_id=sess,
                                    response=make_response([("沙灘", 1.0, 0.4)]), vecs=np.array([vec(1)]))
        store.share_submission(sid, sess)
    stims = store.list_stimuli(None)
    assert [s.id for s in stims][2:] == [b.id, a.id]
    assert stims[2].response_count == 2


# --- 投影器 ---------------------------------------------------------------

def test_projector_roundtrip(store):
    assert store.load_projector("s09") is None
    proj = Projector.fit(STIM_VEC, np.array([vec(i) for i in range(5)]))
    store.save_projector("s09", proj.to_dict())
    store.save_projector("s09", proj.to_dict())  # 覆寫
    again = Projector.from_dict(store.load_projector("s09"))
    np.testing.assert_allclose(again.components, proj.components)


# --- 作答與分享 -------------------------------------------------------------

def test_save_and_share_permissions(store):
    sid = submit(store, S1, [("牛頓", 1.0, 0.6)], [vec(0)], share=False)
    assert len(sid) == 32
    int(sid, 16)
    assert store.pool_vectors_by_user("s09") == {}
    assert store.pool_stats("s09", 1).human_users == 0  # 沒同意 → 不在分布中

    assert not store.share_submission(sid, S2)
    assert not store.share_submission("nope", S1)
    assert store.share_submission(sid, S1)
    assert store.share_submission(sid, S1)  # 重複分享仍為 True
    assert list(store.pool_vectors_by_user("s09")) == [S1]


def test_save_submission_rejects_mismatched_vecs(store):
    with pytest.raises(ValueError):
        store.save_submission(stimulus_id="s09", session_id=S1,
                              response=make_response([("牛頓", 1.0, 0.6), ("水果", 1.0, 0.1)]),
                              vecs=np.array([vec(0)]))


def test_pool_excludes_zero_weight(store):
    submit(store, S1, [("牛頓", 1.0, 0.6), ("化油器", 0.0, 0.9), ("水果", 0.5, 0.1)],
           [vec(0), vec(1), vec(2)])
    submit(store, S2, [("牛頓", 1.0, 0.6)], [vec(0)])
    pool = store.pool_vectors_by_user("s09")
    assert pool[S1].shape == (2, DIM) and pool[S1].dtype == np.float32
    np.testing.assert_allclose(pool[S1][0], vec(0), rtol=1e-6)
    assert pool[S2].shape == (1, DIM)
    assert sorted(store.pool_distances("s09").round(4).tolist()) == [0.1, 0.6, 0.6]


def test_pool_raw_fingerprints(store):
    submit(store, S1, [("牛頓", 1.0, 0.6)], [vec(0)], distance_raw=0.3)
    submit(store, S1, [("水果", 1.0, 0.1)], [vec(1)], distance_raw=0.5)  # 同一人：取最新
    submit(store, S2, [("牛頓", 1.0, 0.6)], [vec(0)], distance_raw=0.7)
    submit(store, S3, [("牛頓", 1.0, 0.6)], [vec(0)], share=False)
    fps = store.pool_raw_fingerprints("s09", exclude_session=None)
    assert sorted(f["distance"] for f in fps) == [0.5, 0.7]
    assert set(fps[0]) == {"distance", "breadth", "rarity", "appropriateness", "fluency"}
    fps2 = store.pool_raw_fingerprints("s09", exclude_session=S2)
    assert [f["distance"] for f in fps2] == [0.5]


def test_pool_stats(store):
    store.add_seed_points("s09", ["水果", "紅色"], np.array([vec(3), vec(4)]))
    submit(store, S1, [("牛頓", 1.0, 0.6)], [vec(0)])
    submit(store, S1, [("水果", 1.0, 0.1)], [vec(1)])
    submit(store, S2, [("牛頓", 1.0, 0.6)], [vec(0)])
    st = store.pool_stats("s09", min_users=2)
    assert (st.human_users, st.seed_points, st.enough_for_percentile) == (2, 2, True)
    assert not store.pool_stats("s09", min_users=3).enough_for_percentile
    assert store.get_stimulus("s09", None).response_count == 2


# --- 星雲 ----------------------------------------------------------------

def test_cloud_points(store):
    assert store.cloud_points("s09") == []  # 沒有投影器
    seeds = [vec(3), vec(4), vec(3)]
    store.add_seed_points("s09", ["水果", "紅色", "水果"], np.array(seeds))
    store.add_seed_points("s09", ["派"], np.array([vec(5)]), distances=np.array([0.42]))
    proj = Projector.fit(STIM_VEC, np.array(seeds + [vec(0), vec(5)]))
    store.save_projector("s09", proj.to_dict())

    submit(store, S1, [("牛頓", 1.0, 0.6), ("化油器", 0.0, 0.9)], [vec(0), vec(1)])
    submit(store, S2, [("牛頓", 1.0, 0.6)], [vec(0)])
    submit(store, S3, [("牛頓", 1.0, 0.6)], [vec(0)], share=False)

    cloud = store.cloud_points("s09")
    by_key = {(c.source, c.text): c for c in cloud}
    assert ("human", "化油器") not in by_key
    newton = by_key[("human", "牛頓")]
    assert newton.count == 2
    assert cloud[0].text == "牛頓"  # count 最多的排前面
    assert by_key[("seed", "水果")].count == 2
    assert by_key[("seed", "紅色")].count == 1

    def r(p):
        return float(np.sqrt(p.x**2 + p.y**2 + p.z**2))

    assert r(newton.position) == pytest.approx(0.6, abs=1e-6)
    assert r(by_key[("seed", "派")].position) == pytest.approx(0.42, abs=1e-6)
    expected = 1 - float(vec(4) @ STIM_VEC)
    assert r(by_key[("seed", "紅色")].position) == pytest.approx(expected, abs=1e-5)

    assert len(store.cloud_points("s09", limit=2)) == 2


def test_add_seed_points_validates(store):
    with pytest.raises(ValueError):
        store.add_seed_points("s09", ["a", "b"], np.array([vec(0), vec(1)]), distances=np.array([0.1]))
