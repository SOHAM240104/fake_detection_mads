"""
Minimal smoke test:
- Runs NOMA on a synthetic tone using the vendored model.
This can be called from CI or locally via:

    python smoke_test.py
"""

import os
import tempfile

import numpy as np
import soundfile as sf

from detectors.noma import get_noma_model_path, run_noma_prediction_with_features


def _make_tone(sec: float = 2.0, sr: int = 22050) -> str:
    n = int(sr * sec)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    y = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t) + 0.01 * rng.standard_normal(n)
    y = np.clip(y, -1, 1).astype("float32")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr)
    return path


def main() -> None:
    model_path = get_noma_model_path()
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit("NOMA model path not found.")

    p = _make_tone()
    try:
        times, probas, X, feature_names = run_noma_prediction_with_features(model_path, audio_path=p)
    finally:
        os.remove(p)

    print(f"NOMA smoke: blocks={len(times)}, features={X.shape[1]}, first_p_fake={probas[0,0]:.4f}")


if __name__ == "__main__":
    main()

