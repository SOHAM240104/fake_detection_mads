import numpy as np


def compute_noma_permutation_feature_sensitivity(
    *,
    feature_matrix: np.ndarray,
    pipeline,
    feature_names: tuple[str, ...] | list[str],
    block_times_seconds: np.ndarray | None = None,
    seed: int = 42,
    max_blocks: int | None = None,
    top_k: int = 5,
    use_calibrated_p_fake: bool = True,
) -> dict:
    """
    Permutation sensitivity heatmap per 1-second block for the NOMA SVM.

    Perturbation:
      For each feature i, shuffle the values of that single feature across blocks
      (keeping all other features intact), then recompute P(fake).

    Output:
      - `sensitivity_abs`: abs(delta calibrated p(fake)) matrix, shape (B, F)
      - `delta_p_fake`: signed delta calibrated p(fake), shape (B, F)
      - `topk_per_block`: list of top-k feature contributions per block
    """

    from calibration_runtime import noma_p_fake_to_calibrated

    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2D: (num_blocks, num_features).")

    X = np.asarray(feature_matrix, dtype=np.float64)
    B, F = X.shape
    if len(feature_names) != F:
        raise ValueError(f"feature_names length {len(feature_names)} != feature_matrix width {F}")

    if max_blocks is not None and B > max_blocks:
        idx = np.linspace(0, B - 1, num=max_blocks, dtype=int)
        X = X[idx]
        B = X.shape[0]
        if block_times_seconds is not None:
            block_times_seconds = np.asarray(block_times_seconds, dtype=float)[idx]

    baseline_proba = pipeline.predict_proba(X)
    p_fake_raw = baseline_proba[:, 0]
    if use_calibrated_p_fake:
        p_fake_base = noma_p_fake_to_calibrated(p_fake_raw)
    else:
        p_fake_base = np.asarray(p_fake_raw, dtype=float)

    rng = np.random.default_rng(seed)

    sensitivity_abs = np.zeros((B, F), dtype=np.float64)
    delta_p_fake = np.zeros((B, F), dtype=np.float64)

    # For speed, avoid copying the whole matrix for every feature if possible.
    # We do one copy per feature to keep semantics simple and deterministic.
    for fi in range(F):
        Xp = X.copy()
        perm = rng.permutation(B)
        Xp[:, fi] = Xp[perm, fi]

        proba_perm = pipeline.predict_proba(Xp)
        p_fake_raw_perm = proba_perm[:, 0]
        if use_calibrated_p_fake:
            p_fake_perm = noma_p_fake_to_calibrated(p_fake_raw_perm)
        else:
            p_fake_perm = np.asarray(p_fake_raw_perm, dtype=float)

        delta = p_fake_perm - p_fake_base
        delta_p_fake[:, fi] = delta
        sensitivity_abs[:, fi] = np.abs(delta)

    # Top-k per block for interpretability.
    topk_per_block = []
    for bi in range(B):
        ranked = np.argsort(sensitivity_abs[bi])[::-1][:top_k]
        topk_per_block.append(
            {
                "block_index": int(bi),
                "time_seconds": float(block_times_seconds[bi]) if block_times_seconds is not None else None,
                "topk": [
                    {
                        "feature": str(feature_names[int(idx)]),
                        "feature_index": int(idx),
                        "sensitivity_abs": float(sensitivity_abs[bi, int(idx)]),
                        "delta_p_fake": float(delta_p_fake[bi, int(idx)]),
                    }
                    for idx in ranked
                ],
            }
        )

    return {
        "schema_version": "noma_permutation_feature_sensitivity_v1",
        "perturbation": "permute_feature_value_across_blocks",
        "seed": int(seed),
        "use_calibrated_p_fake": bool(use_calibrated_p_fake),
        "block_times_seconds": (
            np.asarray(block_times_seconds, dtype=float).tolist() if block_times_seconds is not None else None
        ),
        "feature_names": [str(n) for n in feature_names],
        "baseline_p_fake": np.asarray(p_fake_base, dtype=float).tolist(),
        "sensitivity_abs": sensitivity_abs.tolist(),
        "delta_p_fake": delta_p_fake.tolist(),
        "topk_per_block": topk_per_block,
    }

