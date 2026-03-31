import numpy as np


def compute_frame_spectrum_stats(frames_gray: np.ndarray) -> dict:
    """
    Compute simple spectral stats for each frame.

    Args:
        frames_gray: (T, H, W) grayscale frames.

    Returns:
        dict with:
          - high_freq_energy: (T,) mean high-frequency energy per frame.
    """
    frames = np.asarray(frames_gray, dtype=float)
    if frames.ndim != 3:
        raise ValueError("frames_gray must be 3D (T, H, W)")
    T, H, W = frames.shape
    high = np.zeros(T, dtype=float)

    yy, xx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    r_max = r.max() or 1.0
    r2 = 0.5 * r_max
    mask_high = r >= r2

    for t in range(T):
        F = np.fft.fft2(frames[t])
        P = np.abs(F) ** 2
        high[t] = float(P[mask_high].mean())

    return {"high_freq_energy": high.tolist()}

