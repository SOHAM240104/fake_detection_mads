import json
import hashlib
import os
import tempfile
from functools import lru_cache
from typing import Any

from config import PROJECT_ROOT, AVH_DIR, AVH_GRADCAM_SCRIPT, AVH_FUSION_CKPT, AVH_AVHUBERT_CKPT
from logging_utils import get_logger, log_timed
from metrics import inc_counter


def _get_readable_ckpt_path(path: str, tmp_name: str = "AVH-Align_AV1M.pt", force_tmp: bool = False):
    """Local copy of the checkpoint readability helper for Grad-CAM subprocess use."""
    import shutil

    tmp = os.path.join(tempfile.gettempdir(), tmp_name)
    try:
        with open(path, "rb") as f:
            f.read(1)
    except PermissionError:
        shutil.copy2(path, tmp)
        return tmp

    if force_tmp:
        if not os.path.isfile(tmp) or os.path.getsize(tmp) != os.path.getsize(path):
            shutil.copy2(path, tmp)
        return tmp

    return path


@lru_cache(maxsize=16)
def _sha256_file_cached(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gradcam_mouth_roi(
    video_path: str,
    python_exe: str,
    top_k: int = 2,
    adv_ckpt: str | None = None,
    roi_path: str | None = None,
    audio_path: str | None = None,
    timeout: int = 300,
    keep_temp: bool = False,
    capture_attention: bool = False,
) -> tuple[bool, Any, Any]:
    """
    Runs AVH/gradcam_mouth_roi.py and returns:
      - (ok, overlays_dir_or_error, index_json_or_none)
    """
    logger = get_logger("explainability.gradcam_avh")

    if not os.path.exists(AVH_GRADCAM_SCRIPT):
        return (False, "Grad-CAM script not found. Expected gradcam_mouth_roi.py under AVH/.", None)
    if not (os.path.exists(AVH_FUSION_CKPT) and os.path.exists(AVH_AVHUBERT_CKPT)):
        return (False, "AVH checkpoints missing for Grad-CAM.", None)

    if adv_ckpt:
        if not os.path.isfile(adv_ckpt):
            return (False, f"Adversary checkpoint not found: {adv_ckpt}", None)

    from subprocess_utils import run_subprocess_capture, validate_python_exe

    try:
        py = validate_python_exe(python_exe)
    except Exception as e:
        return (False, str(e), None)

    cache_root = os.path.join(PROJECT_ROOT, ".cache", "gradcam")
    os.makedirs(cache_root, exist_ok=True)

    # Cache key: input media + relevant checkpoint hashes + runtime params.
    # We hash the inputs to make the cache stable across temp path changes.
    video_hash = _sha256_file_cached(video_path)
    fusion_hash = _sha256_file_cached(AVH_FUSION_CKPT)
    avhubert_hash = _sha256_file_cached(AVH_AVHUBERT_CKPT)
    roi_hash = _sha256_file_cached(roi_path) if roi_path and os.path.isfile(roi_path) else "none"
    audio_hash = _sha256_file_cached(audio_path) if audio_path and os.path.isfile(audio_path) else "none"
    adv_hash = _sha256_file_cached(adv_ckpt) if adv_ckpt and os.path.isfile(adv_ckpt) else "none"

    cache_key = (
        f"video={video_hash}|fusion={fusion_hash}|avhubert={avhubert_hash}|roi={roi_hash}|audio={audio_hash}|"
        f"adv={adv_hash}|topk={int(top_k)}|cap_attn={bool(capture_attention)}"
    )
    cache_key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(cache_root, cache_key_hash)

    index_cache_path = os.path.join(cache_dir, "index.json")
    overlays_cache_dir = os.path.join(cache_dir, "overlays")

    def _restore_from_cache() -> tuple[bool, Any, Any]:
        if not (os.path.isfile(index_cache_path) and os.path.isdir(overlays_cache_dir)):
            return (False, None, None)

        out_dir = tempfile.mkdtemp(prefix="gradcam_out_", dir=tempfile.gettempdir())
        try:
            import shutil

            overlays_dir = os.path.join(out_dir, "overlays")
            shutil.copytree(overlays_cache_dir, overlays_dir)

            restored_index_path = os.path.join(out_dir, "index.json")
            shutil.copy2(index_cache_path, restored_index_path)
            with open(restored_index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)

            # Ensure overlay_dir points to the restored directory.
            idx["overlay_dir"] = overlays_dir
            return (True, overlays_dir, idx)
        except Exception:
            # If cache restore fails, fall back to recompute.
            try:
                import shutil

                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            return (False, None, None)

    restored_ok, restored_overlays_dir, restored_idx = _restore_from_cache()
    if restored_ok:
        inc_counter("gradcam_cache_hit", stage="gradcam")
        return (True, restored_overlays_dir, restored_idx)

    out_dir = tempfile.mkdtemp(prefix="gradcam_out_", dir=tempfile.gettempdir())
    try:
        import shutil

        fusion_path = _get_readable_ckpt_path(AVH_FUSION_CKPT, force_tmp=True)
        avhubert_path = _get_readable_ckpt_path(AVH_AVHUBERT_CKPT, "self_large_vox_433h.pt", force_tmp=True)

        cmd = [
            py,
            "gradcam_mouth_roi.py",
            "--video_path",
            os.path.abspath(video_path),
            "--out_dir",
            out_dir,
            "--top_k",
            str(int(top_k)),
            "--overwrite",
            "--device",
            "cpu",
            "--fusion_ckpt",
            fusion_path,
            "--avhubert_ckpt",
            avhubert_path,
        ]

        if roi_path and audio_path:
            cmd += ["--roi_path", os.path.abspath(roi_path), "--audio_path", os.path.abspath(audio_path)]
        if adv_ckpt:
            adv_path = _get_readable_ckpt_path(adv_ckpt, tmp_name="feature_adversary_latest.pt", force_tmp=True)
            cmd += ["--adv_ckpt", adv_path]
        if keep_temp:
            cmd.append("--keep_temp")
        if capture_attention:
            cmd.append("--capture_attention")

        with log_timed(logger, "gradcam_subprocess", cache_hit=False):
            run_res = run_subprocess_capture(cmd, cwd=AVH_DIR, timeout_s=timeout)
        out = (run_res.get("stdout") or "") + (run_res.get("stderr") or "")
        timed_out = bool(run_res.get("timed_out"))
        returncode = run_res.get("returncode")

        if timed_out or (returncode not in (0, None)):
            return (False, out if out else "Grad-CAM timed out or failed.", None)

        index_path = os.path.join(out_dir, "index.json")
        if not os.path.isfile(index_path):
            return (False, "Grad-CAM finished but index.json was not created.", None)

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        idx.setdefault("xai_status", {})

        # Optional: compute temporal inconsistency, region tracks, and fused heatmaps
        # if CAM volume and video are available.
        cam_volume_path = idx.get("cam_volume_path")
        video_src = idx.get("video_path") or idx.get("input_video")
        idx.setdefault("xai_errors", {})
        if cam_volume_path and os.path.isfile(cam_volume_path):
            import numpy as np
            from explainability.video_temporal import compute_temporal_inconsistency
            from explainability.video_regions import cam_to_binary_masks, track_regions_iou, summarize_region_anomalies
            from explainability.video_fusion import (
                prepare_gray_frames_from_video,
                compute_optical_flow_error,
                compute_frequency_noise_map,
                generate_fused_heatmap,
            )
            from explainability.video_frequency import compute_frame_spectrum_stats

            cam = np.load(cam_volume_path)  # expected shape (T, H, W)

            # Temporal inconsistency
            try:
                flat = cam.reshape(cam.shape[0], -1)
                delta_t = compute_temporal_inconsistency(flat)
                idx["temporal_inconsistency"] = delta_t.tolist()
                idx["xai_status"]["temporal_inconsistency"] = "computed"
            except Exception as e:
                idx["xai_status"]["temporal_inconsistency"] = "failed"
                idx["xai_errors"]["temporal_inconsistency"] = str(e)
                logger.exception("temporal inconsistency enrichment failed")

            # Region tracks
            try:
                masks = cam_to_binary_masks(cam)
                tracks = track_regions_iou(masks, cam)
                idx["region_tracks"] = summarize_region_anomalies(tracks)
                idx["xai_status"]["region_tracks"] = "computed"
            except Exception as e:
                idx["xai_status"]["region_tracks"] = "failed"
                idx["xai_errors"]["region_tracks"] = str(e)
                logger.exception("region tracking enrichment failed")

            # Multi-signal fusion and frequency stats if we have the original video.
            if video_src and os.path.isfile(video_src):
                try:
                    frames_gray, _ = prepare_gray_frames_from_video(video_src)
                    T_cam = cam.shape[0]
                    T_vid = frames_gray.shape[0]
                    T_use = min(T_cam, T_vid)
                    max_fusion_frames = int(idx.get("max_fusion_frames", 200)) if isinstance(idx.get("max_fusion_frames", None), int) else 200
                    if T_use > max_fusion_frames:
                        idx["xai_status"]["fusion"] = f"skipped:too_long:{T_use}>{max_fusion_frames}"
                        idx["xai_status"]["video_frequency_stats"] = f"skipped:too_long:{T_use}>{max_fusion_frames}"
                    else:
                        cam_use = cam[:T_use]
                        frames_use_full = frames_gray[:T_use]
                        T_use, Hc, Wc = cam_use.shape
                        frames_use = np.zeros((T_use, Hc, Wc), dtype=frames_use_full.dtype)
                        import cv2 as _cv2

                        for t in range(T_use):
                            frames_use[t] = _cv2.resize(
                                frames_use_full[t],
                                (Wc, Hc),
                                interpolation=_cv2.INTER_AREA,
                            )

                        flow_err = compute_optical_flow_error(frames_use)
                        freq_noise = compute_frequency_noise_map(frames_use)
                        fused = generate_fused_heatmap(
                            cam_use.astype(float), flow_err[:T_use], freq_noise[:T_use]
                        )
                        idx["fused_heatmap_path"] = os.path.join(os.path.dirname(cam_volume_path), "fused_heatmap.npy")
                        np.save(idx["fused_heatmap_path"], fused)
                        idx["xai_status"]["fusion"] = "computed"

                        freq_stats = compute_frame_spectrum_stats(frames_use)
                        idx["video_frequency_stats"] = freq_stats
                        idx["xai_status"]["video_frequency_stats"] = "computed"
                except Exception as e:
                    idx["xai_status"]["fusion"] = "failed"
                    idx["xai_status"]["video_frequency_stats"] = "failed"
                    idx["xai_errors"]["fusion"] = str(e)
                    idx["xai_errors"]["video_frequency_stats"] = str(e)
                    logger.exception("fusion/frequency enrichment failed")
            else:
                idx["xai_status"]["fusion"] = "skipped:no_video_src"
                idx["xai_status"]["video_frequency_stats"] = "skipped:no_video_src"
        else:
            idx["xai_status"]["temporal_inconsistency"] = "skipped:no_cam_volume"
            idx["xai_status"]["region_tracks"] = "skipped:no_cam_volume"
            idx["xai_status"]["fusion"] = "skipped:no_cam_volume"
            idx["xai_status"]["video_frequency_stats"] = "skipped:no_cam_volume"

        overlays_dir = idx.get("overlay_dir", os.path.join(out_dir, "overlays"))
        # Best-effort cache write.
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Write enriched index back before caching.
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(idx, f, indent=2)
            shutil.copy2(index_path, os.path.join(cache_dir, "index.json"))
            # Copy overlay images.
            if os.path.isdir(overlays_dir):
                if os.path.isdir(overlays_cache_dir):
                    shutil.rmtree(overlays_cache_dir, ignore_errors=True)
                shutil.copytree(overlays_dir, overlays_cache_dir)
        except Exception:
            pass

        return (True, overlays_dir, idx)
    finally:
        if not keep_temp:
            # overlays were rendered via st.image; safe to delete after returning.
            import shutil

            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass

