import numpy as np


def confidence_instability(p: np.ndarray, window: int = 3) -> dict:
    """
    Compute per-time confidence variance and a global Confidence Instability Index (CII).

    Args:
        p: (T,) probability sequence (e.g., p(fake) per block).
        window: half-window size for local variance (number of steps before/after).

    Returns:
        dict with:
          - variance_per_time: (T,) local variance over sliding windows.
          - CII: scalar mean variance over time.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be 1D")
    T = p.shape[0]
    if T == 0:
        return {"variance_per_time": [], "CII": 0.0}

    var = np.zeros(T, dtype=float)
    for t in range(T):
        s = max(0, t - window)
        e = min(T, t + window + 1)
        var[t] = np.var(p[s:e])
    cii = float(var.mean())
    return {"variance_per_time": var.tolist(), "CII": cii}

