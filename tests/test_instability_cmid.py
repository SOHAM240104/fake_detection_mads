import numpy as np

from explainability.cross_modal import compute_cross_modal_sync
from explainability.instability import confidence_instability


def test_cmid_basic_behaviour():
    # Simple case: perfectly aligned embeddings -> high similarity, low CMID.
    T, D = 5, 3
    a = np.ones((T, D))
    v = np.ones((T, D))
    out = compute_cross_modal_sync(a, v)
    sim = np.array(out["similarity"])
    cmid = np.array(out["cmid"])
    assert sim.shape == (T,)
    assert cmid.shape == (T,)
    assert np.allclose(sim, sim[0])
    assert np.allclose(cmid, 0.0)


def test_confidence_instability_variance():
    # Two sequences: one stable, one highly oscillatory.
    p_stable = np.array([0.5] * 20)
    p_osc = np.array([0.0, 1.0] * 10)

    out_stable = confidence_instability(p_stable, window=2)
    out_osc = confidence_instability(p_osc, window=2)

    assert out_stable["CII"] >= 0.0
    assert out_osc["CII"] > out_stable["CII"]

