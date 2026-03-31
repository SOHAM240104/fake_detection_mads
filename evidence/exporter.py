import hashlib
import io
import json
import os
import zipfile

import pandas as pd


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_evidence_bundle(
    *,
    input_video_path: str,
    input_video_name: str,
    avh_score: float,
    audio_path: str | None,
    roi_path: str | None,
    cam_idx: dict | None,
    overlays_dir: str | None,
    noma_df: pd.DataFrame | None,
) -> bytes:
    """
    Create an evidence zip bundle (bytes) suitable for download:
    - hashes (sha256)
    - scores
    - extracted audio + mouth ROI
    - Grad-CAM overlays + index.json
    - NOMA per-second predictions (csv)
    """
    manifest = {
        "input_video": {
            "name": input_video_name,
            "path": input_video_path,
            "sha256": _sha256_file(input_video_path),
        },
        "avh": {"score": float(avh_score)},
        "artifacts": {
            "audio_wav_sha256": _sha256_file(audio_path) if audio_path and os.path.isfile(audio_path) else None,
            "mouth_roi_mp4_sha256": _sha256_file(roi_path) if roi_path and os.path.isfile(roi_path) else None,
        },
        "gradcam": cam_idx or None,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

        if audio_path and os.path.isfile(audio_path):
            z.write(audio_path, arcname="artifacts/audio.wav")
        if roi_path and os.path.isfile(roi_path):
            z.write(roi_path, arcname="artifacts/mouth_roi.mp4")

        if overlays_dir and os.path.isdir(overlays_dir):
            for fn in sorted(os.listdir(overlays_dir)):
                if fn.lower().endswith(".png"):
                    z.write(os.path.join(overlays_dir, fn), arcname=f"gradcam/overlays/{fn}")

        if cam_idx:
            z.writestr("gradcam/index.json", json.dumps(cam_idx, indent=2))

        if noma_df is not None and not noma_df.empty:
            z.writestr("noma/predictions.csv", noma_df.to_csv(index=False))

    return buf.getvalue()

