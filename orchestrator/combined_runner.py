import os
import tempfile
from typing import Any


def run_combined_avh_to_noma(
    *,
    video_path: str,
    video_name: str,
    use_unsup_avh: bool,
    python_exe: str | None,
    run_forensics_cam: bool,
    forensics_top_k: int,
    run_robustness_delta: bool,
    adv_ckpt_path: str,
    capture_attention: bool,
    export_bundle: bool,
    noma_model_path: str,
    timeout: int = 900,
) -> dict[str, Any]:
    """
    Multi-stage pipeline:
      AVH (preprocess -> feature extraction -> score) -> optional Grad-CAM evidence
      -> extract audio from AVH temp work dir -> NOMA per-block predictions
      -> optional evidence bundle export (.zip bytes)

    Returns a dict for Streamlit rendering (raw objects + errors).
    """
    import shutil
    import hashlib
    import json

    from detectors.avh_align import run_avh_on_video, run_avh_unsupervised_on_video
    from detectors.noma import run_noma_prediction
    from explainability.gradcam_avh import run_gradcam_mouth_roi
    from evidence.exporter import zip_evidence_bundle
    from config import PROJECT_ROOT, AVH_FUSION_CKPT, AVH_AVHUBERT_CKPT
    from logging_utils import get_logger, log_timed
    from metrics import inc_counter

    result: dict[str, Any] = {
        "avh_ok": False,
        "avh_score": None,
        "avh_error": None,
        "audio_path": None,
        "cam_ok": False,
        "cam_overlays_dir": None,
        "cam_idx": None,
        "roi_path": None,
        # Keep both keys during transition; `noma_df` is canonical.
        "noma_df": None,
        "nona_df": None,
        "bundle_bytes": None,
        "bundle_error": None,
        # For UI cleanup: Grad-CAM outputs are created under a temp dir.
        "cam_parent_dir": None,
        "cmid": None,
        "cmid_status": "not_computed",
    }

    logger = get_logger("orchestrator.combined_runner")
    work_dir = None
    try:
        # Step 1 cache: AVH stage artifacts (score + extracted audio + mouth ROI).
        def _sha256_file(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        cache_root = os.path.join(PROJECT_ROOT, ".cache", "avh_stage")
        os.makedirs(cache_root, exist_ok=True)

        avh_ret = None
        cache_dir: str | None = None
        meta_path: str | None = None
        if not result["avh_ok"]:
            try:
                video_hash = _sha256_file(video_path)
                fusion_hash = _sha256_file(AVH_FUSION_CKPT)
                avhubert_hash = _sha256_file(AVH_AVHUBERT_CKPT)
                py_str = (python_exe or "").strip()

                cache_key_str = (
                    f"video={video_hash}|unsup={bool(use_unsup_avh)}|py={py_str}|fusion={fusion_hash}|avhubert={avhubert_hash}"
                )
                cache_key_hash = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()
                cache_dir = os.path.join(cache_root, cache_key_hash)
                meta_path = os.path.join(cache_dir, "meta.json")

                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    cached_score = float(meta.get("score"))
                    cached_audio_basename = meta.get("audio_basename")
                    cached_audio_path = os.path.join(cache_dir, "audio", cached_audio_basename)
                    cached_roi_path = os.path.join(cache_dir, "mouth_roi.mp4")

                    if cached_audio_basename and os.path.isfile(cached_audio_path) and os.path.isfile(cached_roi_path):
                        work_dir = tempfile.mkdtemp(prefix="avh_restore_", dir=tempfile.gettempdir())
                        restored_audio_path = os.path.join(work_dir, cached_audio_basename)
                        restored_roi_path = os.path.join(work_dir, "mouth_roi.mp4")
                        shutil.copy2(cached_audio_path, restored_audio_path)
                        shutil.copy2(cached_roi_path, restored_roi_path)

                        result["avh_ok"] = True
                        result["avh_score"] = cached_score
                        result["audio_path"] = restored_audio_path
            except Exception:
                cache_dir = None
                meta_path = None
                work_dir = None

        # Step 1: run AVH (cache miss path)
        if not result["avh_ok"]:
        # Step 1: run AVH
            if use_unsup_avh:
                avh_ret = run_avh_unsupervised_on_video(
                    video_path,
                    timeout=timeout,
                    python_exe=python_exe,
                    keep_temp=True,
                )
            else:
                avh_ret = run_avh_on_video(
                    video_path,
                    timeout=timeout,
                    python_exe=python_exe,
                    keep_temp=True,
                )

            if not avh_ret or avh_ret[0] is False:
                result["avh_ok"] = False
                result["avh_error"] = avh_ret[1] if avh_ret else "AVH failed"
                return result

            # keep_temp=True => (True, score, audio_path)
            _, avh_score, audio_path = avh_ret
            result["avh_ok"] = True
            result["avh_score"] = float(avh_score)
            result["audio_path"] = audio_path

            if not audio_path:
                return result

            # Cache AVH stage artifacts (best-effort).
            try:
                cached_roi_path = os.path.join(os.path.dirname(audio_path), "mouth_roi.mp4")
                if os.path.isfile(cached_roi_path) and cache_dir and meta_path:
                    os.makedirs(os.path.join(cache_dir, "audio"), exist_ok=True)
                    cached_audio_basename = os.path.basename(audio_path)
                    shutil.copy2(audio_path, os.path.join(cache_dir, "audio", cached_audio_basename))
                    shutil.copy2(cached_roi_path, os.path.join(cache_dir, "mouth_roi.mp4"))
                    tmp_meta_path = meta_path + ".tmp"
                    with open(tmp_meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"score": float(avh_score), "audio_basename": cached_audio_basename},
                            f,
                            indent=2,
                        )
                    os.replace(tmp_meta_path, meta_path)
            except Exception:
                pass

        audio_path = result["audio_path"]
        if not audio_path:
            return result

        # Step 2: optional Grad-CAM evidence
        work_dir = os.path.dirname(audio_path)
        result["roi_path"] = os.path.join(work_dir, "mouth_roi.mp4")

        if run_forensics_cam:
            adv_for_cam = adv_ckpt_path.strip() or None if run_robustness_delta else None

            ok_cam, overlays_dir_or_err, cam_idx_or_none = run_gradcam_mouth_roi(
                video_path=video_path,
                python_exe=python_exe,
                top_k=forensics_top_k,
                adv_ckpt=adv_for_cam,
                roi_path=result["roi_path"],
                audio_path=audio_path,
                timeout=timeout,
                keep_temp=True,
                capture_attention=capture_attention,
            )

            result["cam_ok"] = bool(ok_cam)
            if ok_cam:
                result["cam_overlays_dir"] = overlays_dir_or_err
                result["cam_idx"] = cam_idx_or_none
                # overlays_dir is .../overlays; parent is the temp root for cleanup after UI.
                result["cam_parent_dir"] = os.path.dirname(result["cam_overlays_dir"])
            else:
                result["cam_overlays_dir"] = None
                result["cam_idx"] = None

        # Step 3: NOMA on extracted audio
        import pandas as pd

        times, probas = run_noma_prediction(noma_model_path, audio_path=audio_path)
        preds = probas.argmax(axis=1)
        confidences = probas.max(axis=1)
        preds_str = ["Fake" if i == 0 else "Real" for i in preds]
        from calibration_runtime import noma_p_fake_to_calibrated

        p_fake = noma_p_fake_to_calibrated(probas[:, 0])
        p_real = 1.0 - p_fake

        noma_df = pd.DataFrame(
            {
                "Seconds": times,
                "Prediction": preds_str,
                "Confidence": confidences,
                "p_fake": p_fake,
                "p_real": p_real,
            }
        )
        result["noma_df"] = noma_df
        result["nona_df"] = noma_df

        # Confidence Instability Index (CII) on NOMA p(fake).
        try:
            from explainability.instability import confidence_instability

            inst = confidence_instability(p_fake)
            result["noma_confidence_instability"] = inst
        except Exception:
            result["noma_confidence_instability"] = None

        # CMID hook: requires aligned AVH audio/visual embeddings.
        # This runner currently does not export embeddings by default, so report status.
        try:
            emb_audio = result.get("avh_audio_embeddings")
            emb_visual = result.get("avh_visual_embeddings")
            if emb_audio is not None and emb_visual is not None:
                from explainability.cross_modal import compute_cross_modal_sync

                result["cmid"] = compute_cross_modal_sync(emb_audio, emb_visual)
                result["cmid_status"] = "computed"
            else:
                result["cmid_status"] = "missing_embeddings"
        except Exception:
            result["cmid_status"] = "failed"

        # Step 4: optional evidence bundle export
        if export_bundle:
            bundle_bytes = zip_evidence_bundle(
                input_video_path=video_path,
                input_video_name=video_name,
                avh_score=result["avh_score"],
                audio_path=audio_path,
                roi_path=result["roi_path"],
                cam_idx=result["cam_idx"],
                overlays_dir=result["cam_overlays_dir"],
                noma_df=noma_df,
            )
            result["bundle_bytes"] = bundle_bytes
    finally:
        # Cleanup AVH temp directory to avoid disk bloat.
        # Grad-CAM temp outputs must remain until UI renders overlays.
        if work_dir:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    return result

