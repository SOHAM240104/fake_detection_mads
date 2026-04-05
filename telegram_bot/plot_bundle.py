"""
Build PNG sequences for Telegram from Combined `res` / `cam_idx` (matplotlib; no Streamlit).
"""

from __future__ import annotations

import glob
import io
import os
from typing import Any

import numpy as np
import pandas as pd


def _mpl_line_png(x: np.ndarray, y: np.ndarray, title: str, xlab: str, ylab: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), color="#2563eb", lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _mpl_imshow_png(z: np.ndarray, title: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(z, dtype=float), aspect="auto", cmap="inferno")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_plot_items(res: dict[str, Any]) -> list[tuple[bytes | str, str]]:
    """
    Returns ordered (png bytes or file path, caption). Paths are sent as-is for large overlays.
    """
    items: list[tuple[bytes | str, str]] = []
    cam_idx = res.get("cam_idx") if isinstance(res.get("cam_idx"), dict) else None

    # ── NOMA p(fake) vs time ─────────────────────────────────────────────
    noma_df = res.get("noma_df")
    if isinstance(noma_df, pd.DataFrame) and len(noma_df) > 0 and "p_fake" in noma_df.columns:
        sec = noma_df["Seconds"].values if "Seconds" in noma_df.columns else np.arange(len(noma_df))
        pf = noma_df["p_fake"].astype(float).values
        items.append(
            (
                _mpl_line_png(sec, pf, "NOMA p(fake) per block (calibrated)", "time (s)", "p(fake)"),
                "Chart: NOMA p(fake) vs time",
            )
        )

    # ── CII variance series ────────────────────────────────────────────
    inst = res.get("noma_confidence_instability")
    if isinstance(inst, dict) and isinstance(inst.get("variance_per_time"), list):
        v = inst["variance_per_time"]
        x = np.arange(len(v), dtype=float)
        items.append(
            (
                _mpl_line_png(x, np.asarray(v, dtype=float), "CII — local variance of p(fake)", "block index", "variance"),
                "Chart: NOMA confidence instability (local variance)",
            )
        )

    # ── CMID from cmid dict (matplotlib) ──────────────────────────────
    cmid_d = res.get("cmid")
    if isinstance(cmid_d, dict):
        sim = cmid_d.get("similarity") or []
        cmid_vals = cmid_d.get("cmid") or []
        if sim:
            x = np.arange(len(sim), dtype=float)
            items.append(
                (
                    _mpl_line_png(x, np.asarray(sim, dtype=float), "Cross-modal cosine similarity", "frame index", "similarity"),
                    "Chart: audio–visual embedding similarity",
                )
            )
        if cmid_vals and not sim:
            x = np.arange(len(cmid_vals), dtype=float)
            items.append(
                (
                    _mpl_line_png(x, np.asarray(cmid_vals, dtype=float), "CMID (deviation from median sim)", "frame index", "CMID"),
                    "Chart: CMID",
                )
            )

    if not cam_idx:
        items.extend(_reviewer_and_calibration_only(res, cam_idx))
        items.extend(_overlay_files(res, limit=12))
        return items

    roi_fps = float(cam_idx.get("roi_fps") or 0.0) or 1.0

    # ── Grad-CAM intensity vs time ─────────────────────────────────────
    cp = cam_idx.get("cam_per_frame")
    if isinstance(cp, list) and len(cp) > 0:
        t = np.arange(len(cp), dtype=float) / roi_fps
        items.append(
            (
                _mpl_line_png(t, np.asarray(cp, dtype=float), "Grad-CAM mean intensity vs time", "time (s)", "intensity"),
                "Chart: Grad-CAM intensity vs time",
            )
        )

    # ── Temporal inconsistency Δ_t ─────────────────────────────────────
    ti = cam_idx.get("temporal_inconsistency")
    if isinstance(ti, list) and len(ti) > 0:
        t = np.arange(len(ti), dtype=float) / roi_fps
        items.append(
            (
                _mpl_line_png(t, np.asarray(ti, dtype=float), "Temporal inconsistency (Δ_t)", "time (s)", "Δ_t"),
                "Chart: temporal inconsistency",
            )
        )

    # ── High-frequency energy ─────────────────────────────────────────
    vfs = cam_idx.get("video_frequency_stats")
    if isinstance(vfs, dict) and isinstance(vfs.get("high_freq_energy"), list):
        hfe = vfs["high_freq_energy"]
        x = np.arange(len(hfe), dtype=float)
        items.append(
            (
                _mpl_line_png(x, np.asarray(hfe, dtype=float), "High-frequency energy proxy", "frame index", "energy"),
                "Chart: high-frequency energy",
            )
        )

    # ── Fused heatmap mean intensity ───────────────────────────────────
    fused_path = cam_idx.get("fused_heatmap_path")
    if isinstance(fused_path, str) and os.path.isfile(fused_path):
        try:
            fused = np.load(fused_path)
            if fused.ndim == 3 and fused.shape[0] > 0:
                intensity = fused.reshape(fused.shape[0], -1).mean(axis=1)
                x = np.arange(len(intensity), dtype=float)
                items.append(
                    (
                        _mpl_line_png(x, intensity, "Fused anomaly intensity (mean over space)", "frame index", "intensity"),
                        "Chart: fused heatmap mean intensity",
                    )
                )
                mid = int(fused.shape[0] // 2)
                items.append(
                    (
                        _mpl_imshow_png(fused[mid], f"Fused heatmap slice (t={mid})"),
                        f"Fused heatmap slice (frame {mid})",
                    )
                )
        except OSError:
            pass

    items.extend(_reviewer_and_calibration_only(res, cam_idx))
    items.extend(_overlay_files(res, limit=12))
    return items


def _reviewer_and_calibration_only(res: dict[str, Any], cam_idx: dict[str, Any] | None) -> list[tuple[bytes | str, str]]:
    out: list[tuple[bytes | str, str]] = []
    cam_idx = cam_idx if isinstance(cam_idx, dict) else None

    try:
        from explainability.reviewer_figures import (
            figure_calibration_png_bytes,
            figure_mel_noma_png_bytes,
            figure_triptych_png_bytes,
        )
    except ImportError:
        return out

    roi_p = res.get("roi_path")
    fused_p = cam_idx.get("fused_heatmap_path") if cam_idx else None
    cam_vol = cam_idx.get("cam_volume_path") if cam_idx else None
    odir = None
    if cam_idx:
        odir = cam_idx.get("overlay_dir")
    if not odir or not (isinstance(odir, str) and os.path.isdir(odir)):
        odir = res.get("cam_overlays_dir")

    if (
        isinstance(roi_p, str)
        and os.path.isfile(roi_p)
        and cam_idx
        and isinstance(fused_p, str)
        and os.path.isfile(fused_p)
    ):
        try:
            tmax = int(cam_idx.get("T_use") or cam_idx.get("T_roi") or 1) - 1
            tmax = max(0, tmax)
            tri_t = min(tmax, max(0, tmax // 2))
            b = figure_triptych_png_bytes(
                roi_path=roi_p,
                frame_idx=int(tri_t),
                fused_npy_path=fused_p,
                cam_npy_path=cam_vol if isinstance(cam_vol, str) else None,
                overlay_dir=odir if isinstance(odir, str) else None,
            )
            out.append((b, "Reviewer: triptych (ROI | Grad-CAM | fused)"))
        except Exception:
            pass

    ap = res.get("audio_path")
    ndf = res.get("noma_df")
    if isinstance(ap, str) and os.path.isfile(ap) and isinstance(ndf, pd.DataFrame) and len(ndf) > 0 and "Seconds" in ndf.columns:
        try:
            sec = ndf["Seconds"].astype(float).values
            pf = ndf["p_fake"].astype(float).values
            b = figure_mel_noma_png_bytes(audio_path=ap, seconds=sec, p_fake=pf)
            out.append((b, "Reviewer: mel spectrogram + NOMA p(fake)"))
        except Exception:
            pass

    if isinstance(res.get("cmid"), dict):
        try:
            from explainability.reviewer_figures import figure_cmid_png_bytes

            b = figure_cmid_png_bytes(res["cmid"])
            out.append((b, "Reviewer: CMID / cosine similarity"))
        except Exception:
            pass

    if cam_idx and isinstance(cam_idx.get("attention_per_frame"), list) and isinstance(cam_idx.get("cam_per_frame"), list):
        try:
            from explainability.reviewer_figures import figure_attention_cam_png_bytes

            b = figure_attention_cam_png_bytes(cam_idx)
            out.append((b, "Reviewer: attention vs Grad-CAM"))
        except Exception:
            pass

    try:
        b = figure_calibration_png_bytes()
        out.append((b, "Calibration: AVH / NOMA raw → calibrated p(fake)"))
    except Exception:
        pass

    return out


def _overlay_files(res: dict[str, Any], *, limit: int) -> list[tuple[bytes | str, str]]:
    cam_idx = res.get("cam_idx") if isinstance(res.get("cam_idx"), dict) else None
    d = None
    if cam_idx:
        d = cam_idx.get("overlay_dir")
    if not d or not os.path.isdir(d):
        d = res.get("cam_overlays_dir")
    if not d or not os.path.isdir(d):
        return []
    paths = sorted(glob.glob(os.path.join(d, "cam_frame_*.png")))[:limit]
    out: list[tuple[bytes | str, str]] = []
    for i, p in enumerate(paths):
        out.append((p, f"Grad-CAM overlay {i + 1}/{len(paths)}: {os.path.basename(p)}"))
    return out
