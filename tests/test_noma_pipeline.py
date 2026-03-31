import os
import tempfile

import numpy as np
import soundfile as sf

from detectors.noma import get_noma_model_path, run_noma_prediction_with_features


def _make_tone(sec: float = 2.5, sr: int = 22050) -> str:
    n = int(sr * sec)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    y = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t) + 0.01 * rng.standard_normal(n)
    y = np.clip(y, -1, 1).astype("float32")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr)
    return path


def test_noma_pipeline_shapes():
    model_path = get_noma_model_path()
    assert model_path is not None and os.path.isfile(model_path)

    p = _make_tone()
    try:
        times, probas, X, feature_names = run_noma_prediction_with_features(model_path, audio_path=p)
    finally:
        os.remove(p)

    assert times.ndim == 1
    assert probas.shape[0] == times.shape[0]
    assert probas.shape[1] == 2
    assert X.shape == (times.shape[0], 41)
    assert len(feature_names) == 41

