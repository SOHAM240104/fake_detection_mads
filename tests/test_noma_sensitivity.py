import numpy as np

from explainability.noma_feature_sensitivity import compute_noma_permutation_feature_sensitivity


class DummyPipeline:
    def __init__(self) -> None:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Simple fake classifier: p_fake = sigmoid(sum(X)), p_real = 1 - p_fake
        s = X.sum(axis=1)
        z = np.exp(-s)
        p_fake = 1.0 / (1.0 + z)
        p_real = 1.0 - p_fake
        return np.vstack([p_fake, p_real]).T


def test_noma_sensitivity_schema():
    X = np.random.rand(4, 3)
    feature_names = ("f1", "f2", "f3")
    times = np.array([0.0, 1.0, 2.0, 3.0])

    out = compute_noma_permutation_feature_sensitivity(
        feature_matrix=X,
        pipeline=DummyPipeline(),
        feature_names=feature_names,
        block_times_seconds=times,
        max_blocks=None,
        top_k=2,
    )

    assert out["schema_version"].startswith("noma_permutation_feature_sensitivity")
    sens = np.asarray(out["sensitivity_abs"])
    assert sens.shape == (4, 3)
    assert len(out["topk_per_block"]) == 4

