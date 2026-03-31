import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import PROJECT_ROOT


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) for binary probabilities.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        ece += (hi - lo) * abs(acc - conf)
    return float(ece)


def _fit_temperature_logistic_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit p = sigmoid(a*x + c) and return (temperature, bias) matching:
      runtime: p = sigmoid((x - bias) / temperature)

    From (x - bias)/T = a*x + c:
      a = 1/T
      c = -bias/T  => bias = -c/a
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1)

    # Large C to reduce regularization; no feature scaling needed for 1D.
    model = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=10_000)
    model.fit(x, y)

    a = float(model.coef_.reshape(-1)[0])
    c = float(model.intercept_.reshape(-1)[0])
    if a == 0:
        # Degenerate fallback: identity mapping.
        return 1.0, 0.0

    temperature = 1.0 / a
    bias = -c / a
    return float(temperature), float(bias)


def _fit_noma_logit_temperature(p_fake_raw: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Fit p_fake_raw -> p_fake via logit-space logistic:
      p = sigmoid(a*logit(p_raw) + c)

    Runtime mapping:
      p_cal = sigmoid((logit(p_raw) + noma_bias) / noma_temperature)

    => a = 1/noma_temperature and c = noma_bias/noma_temperature => noma_bias = c/a
    """
    p = np.asarray(p_fake_raw, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))

    logit = logit.reshape(-1, 1)
    y = np.asarray(labels, dtype=float).reshape(-1)

    model = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=10_000)
    model.fit(logit, y)

    a = float(model.coef_.reshape(-1)[0])
    c = float(model.intercept_.reshape(-1)[0])
    if a == 0:
        return 1.0, 0.0

    noma_temperature = 1.0 / a
    noma_bias = c / a
    return float(noma_temperature), float(noma_bias)


def _quantile_margin(p: np.ndarray, uncertain_rate: float) -> float:
    p = np.asarray(p, dtype=float)
    dist = np.abs(p - 0.5)
    return float(np.quantile(dist, uncertain_rate))


def main():
    ap = argparse.ArgumentParser(description="Fit probability calibration for AVH and NOMA.")
    ap.add_argument("--avh_csv", type=str, default=None, help="CSV with columns: score, label (1=fake,0=real)")
    ap.add_argument("--noma_csv", type=str, default=None, help="CSV with columns: p_fake, label (1=fake,0=real)")
    ap.add_argument("--out_path", type=str, default=os.path.join(PROJECT_ROOT, "calibration_artifacts.json"))
    ap.add_argument("--uncertain_rate", type=float, default=0.2, help="Fraction of most-uncertain samples in validation.")
    args = ap.parse_args()

    if not args.avh_csv and not args.noma_csv:
        raise SystemExit("Provide at least one of --avh_csv or --noma_csv.")

    artifacts: dict[str, float] = {}

    if args.avh_csv:
        df = pd.read_csv(args.avh_csv)
        score = df["score"].to_numpy(dtype=float)
        label = df["label"].to_numpy(dtype=float)

        T, b = _fit_temperature_logistic_1d(score, label)
        logits = (score - b) / T
        p_fake = 1.0 / (1.0 + np.exp(-logits))

        artifacts["avh_temperature"] = T
        artifacts["avh_bias"] = b
        artifacts["avh_uncertainty_margin"] = _quantile_margin(p_fake, args.uncertain_rate)

        auc = roc_auc_score(label, p_fake)
        ap = average_precision_score(label, p_fake)
        brier = brier_score_loss(label, p_fake)
        ece = _ece(p_fake, label)
        print(f"[AVH] ROC-AUC={auc:.4f} AP={ap:.4f} Brier={brier:.4f} ECE={ece:.4f}")

    if args.noma_csv:
        df = pd.read_csv(args.noma_csv)
        p_fake_raw = df["p_fake"].to_numpy(dtype=float)
        label = df["label"].to_numpy(dtype=float)

        noma_T, noma_b = _fit_noma_logit_temperature(p_fake_raw, label)
        p = np.clip(p_fake_raw, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        p_fake = 1.0 / (1.0 + np.exp(-(logit + noma_b) / noma_T))

        artifacts["noma_temperature"] = noma_T
        artifacts["noma_bias"] = noma_b
        artifacts["noma_uncertainty_margin"] = _quantile_margin(p_fake, args.uncertain_rate)

        auc = roc_auc_score(label, p_fake)
        ap = average_precision_score(label, p_fake)
        brier = brier_score_loss(label, p_fake)
        ece = _ece(p_fake, label)
        print(f"[NOMA] ROC-AUC={auc:.4f} AP={ap:.4f} Brier={brier:.4f} ECE={ece:.4f}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, sort_keys=True)
    print(f"Wrote calibration artifacts to: {args.out_path}")


if __name__ == "__main__":
    main()

