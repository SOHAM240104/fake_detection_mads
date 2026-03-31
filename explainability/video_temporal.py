import numpy as np


def compute_temporal_inconsistency(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute per-frame temporal inconsistency from a sequence of embeddings.

    Args:
        embeddings: array of shape (T, D), e.g. per-frame visual embeddings.

    Returns:
        delta_t: array of shape (T,), where delta_t[0] = 0 and
                 delta_t[t] = ||E_t - E_{t-1}||_2 for t > 0.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (T, D)")
    T = embeddings.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=float)
    if T == 1:
        return np.array([0.0], dtype=float)

    diffs = embeddings[1:] - embeddings[:-1]
    delta = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], delta])

