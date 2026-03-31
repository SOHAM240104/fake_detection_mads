from typing import Tuple

import cv2
import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        sigma = 1.0
    return (x - mu) / sigma


def generate_fused_heatmap(
    gradcam_map: np.ndarray,
    optical_flow_error: np.ndarray,
    frequency_noise_map: np.ndarray,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
) -> np.ndarray:
    """
    Fuse Grad-CAM, optical flow error, and frequency noise into a single heatmap.

    All inputs must have shape (T, H, W).
    """
    if not (
        gradcam_map.shape == optical_flow_error.shape == frequency_noise_map.shape
    ):
        raise ValueError("All maps must have the same shape (T, H, W).")
    gc = _zscore(gradcam_map)
    flow = _zscore(optical_flow_error)
    freq = _zscore(frequency_noise_map)
    return w1 * gc + w2 * flow + w3 * freq


def compute_optical_flow_error(frames_gray: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """
    Compute motion inconsistency map from grayscale frames.

    Args:
        frames_gray: (T, H, W) uint8 or float32 grayscale frames.

    Returns:
        flow_err: (T, H, W) motion anomaly magnitude per frame (T[0] is zeros).
    """
    T, H, W = frames_gray.shape
    if T < 2:
        return np.zeros_like(frames_gray, dtype=float)

    # Ensure uint8 for OpenCV.
    if frames_gray.dtype != np.uint8:
        fg = np.clip(frames_gray, 0, 255).astype("uint8")
    else:
        fg = frames_gray

    mags = np.zeros((T - 1, H, W), dtype=float)
    for t in range(T - 1):
        prev = fg[t]
        nxt = fg[t + 1]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.linalg.norm(flow, axis=2)
        mags[t] = mag

    # Smooth over time to estimate baseline motion, then subtract.
    mags_flat = mags.reshape(T - 1, -1)
    kernel = np.ones(smooth_window) / smooth_window
    baseline = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="same"), axis=0, arr=mags_flat
    ).reshape(T - 1, H, W)
    err = np.maximum(0.0, mags - baseline)
    # Pad first frame with zeros.
    flow_err = np.zeros((T, H, W), dtype=float)
    flow_err[1:] = err
    return flow_err


def compute_frequency_noise_map(frames_gray: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Compute a per-pixel high-frequency noise map from grayscale frames.

    Args:
        frames_gray: (T, H, W) grayscale frames in [0, 255] or [0, 1].
        patch_size: size of square patches for FFT aggregation.

    Returns:
        freq_map: (T, H, W) high-frequency energy per pixel.
    """
    T, H, W = frames_gray.shape
    if frames_gray.dtype != np.float32:
        f = frames_gray.astype("float32")
    else:
        f = frames_gray

    freq_map = np.zeros((T, H, W), dtype=float)
    step = patch_size
    for t in range(T):
        frame = f[t]
        for y in range(0, H, step):
            for x in range(0, W, step):
                patch = frame[y : y + step, x : x + step]
                if patch.size == 0:
                    continue
                F = np.fft.fft2(patch)
                P = np.abs(F) ** 2
                # Treat outer 50% of frequencies as "high".
                h, w = patch.shape
                cy, cx = h // 2, w // 2
                yy, xx = np.ogrid[:h, :w]
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                r_max = dist.max() or 1.0
                mask_high = dist >= 0.5 * r_max
                high_energy = float(P[mask_high].mean())
                freq_map[t, y : y + step, x : x + step] = high_energy
    return freq_map


def prepare_gray_frames_from_video(video_path: str, max_frames: int | None = None) -> Tuple[np.ndarray, float]:
    """
    Utility to load video and produce grayscale frames for fusion computations.

    Returns:
        frames_gray: (T, H, W) uint8 grayscale.
        fps: frames per second.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    stack = np.stack(frames, axis=0)
    return stack, float(fps)

