import numpy as np


def compute_cross_modal_sync(audio_emb: np.ndarray, visual_emb: np.ndarray) -> dict:
    """
    Compute cross-modal similarity and CMID scores from aligned audio and visual embeddings.

    Args:
        audio_emb: (T, D) audio embeddings.
        visual_emb: (T, D) visual embeddings.

    Returns:
        dict with:
          - similarity: (T,) cosine similarity over time.
          - cmid: (T,) Cross-Modal Inconsistency Detection scores (higher = more inconsistent).
    """
    a = np.asarray(audio_emb, dtype=float)
    v = np.asarray(visual_emb, dtype=float)
    if a.shape != v.shape or a.ndim != 2:
        raise ValueError("audio_emb and visual_emb must both be 2D with the same shape (T, D)")
    if a.shape[0] == 0:
        return {"similarity": [], "cmid": []}

    # Normalize to unit vectors.
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    sim = np.sum(a_norm * v_norm, axis=1)
    mu = np.median(sim)
    cmid = np.maximum(0.0, mu - sim)

    return {"similarity": sim.tolist(), "cmid": cmid.tolist()}

