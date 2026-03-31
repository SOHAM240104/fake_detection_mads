import json
import math
import os
from functools import lru_cache

from config import PROJECT_ROOT


DEFAULTS = {
    # AVH score -> p(fake) mapping is logistic(sigmoid) by default.
    # p_fake = sigmoid((score - bias) / temperature)
    "avh_temperature": 1.0,
    "avh_bias": 0.0,
    # Uncertainty is computed as closeness to p=0.5.
    "avh_uncertainty_margin": 0.12,
    "noma_uncertainty_margin": 0.12,
    # NOMA recalibration is applied in logit-space to p(fake).
    # p_cal = sigmoid((logit(p_raw) + noma_bias) / noma_temperature)
    "noma_temperature": 1.0,
    "noma_bias": 0.0,
}


@lru_cache(maxsize=1)
def _load_calibration_artifacts() -> dict:
    # This file is expected to be created by the evaluation/calibration harness (offline).
    # If it doesn't exist, we fall back to DEFAULTS.
    path_candidates = [
        os.path.join(PROJECT_ROOT, "calibration_artifacts.json"),
        os.path.join(PROJECT_ROOT, "artifacts", "calibration_artifacts.json"),
    ]
    for p in path_candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def get_uncertainty_margins() -> tuple[float, float]:
    art = _load_calibration_artifacts()
    return (
        float(art.get("avh_uncertainty_margin", DEFAULTS["avh_uncertainty_margin"])),
        float(art.get("noma_uncertainty_margin", DEFAULTS["noma_uncertainty_margin"])),
    )


def avh_score_to_p_fake(score: float) -> float:
    art = _load_calibration_artifacts()
    T = float(art.get("avh_temperature", DEFAULTS["avh_temperature"]))
    b = float(art.get("avh_bias", DEFAULTS["avh_bias"]))
    if T == 0:
        T = 1.0
    x = (float(score) - b) / T
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def noma_p_fake_to_calibrated(p_fake):
    """
    Recalibrate NOMA p(fake) values using a logit-space temperature scaling model.

    If calibration artifacts are missing, this becomes an identity transform.
    """
    import numpy as np

    art = _load_calibration_artifacts()
    T = float(art.get("noma_temperature", DEFAULTS["noma_temperature"]))
    b = float(art.get("noma_bias", DEFAULTS["noma_bias"]))
    if T == 0:
        T = 1.0

    p = np.asarray(p_fake, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    x = (logit + b) / T
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    if np.isscalar(p_fake):
        return float(out.item())
    return out

