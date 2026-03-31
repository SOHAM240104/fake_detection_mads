import io
import os
import re
import sys
import tempfile

from config import (
    AVH_DIR,
    AVH_TEST_SCRIPT,
    AVH_TEST_UNSUP_SCRIPT,
    AVH_FUSION_CKPT,
    AVH_FACE_PREDICTOR,
    AVH_MEAN_FACE,
    AVH_AVHUBERT_CKPT,
    AVH_AVHUBERT_DIR,
)


def check_avh_setup():
    """Return list of (check_name, ok, detail) for AVH setup."""
    # Deterministic verification (optional SHA256 via lockfile).
    try:
        from artifact_manager import ensure_artifacts, default_artifacts

        art_res = ensure_artifacts(
            artifacts=default_artifacts(),
            download_missing=False,
            strict_hash=False,
            write_lock_if_missing=True,
        )
        art_by_name = art_res.get("artifacts", {})
    except Exception:
        art_by_name = {}

    def _status_for_path(path: str) -> tuple[bool, str]:
        if not os.path.isfile(path):
            return False, "missing"

        for _, entry in art_by_name.items():
            if entry.get("path") == path:
                status = entry.get("status", "unknown")
                sha = entry.get("sha256")
                if sha:
                    return status == "verified", f"{path} (sha256={sha[:12]}..)"
                return status == "verified", f"{path} (sha256 not computed)"

        # If we couldn't verify via lockfile, fall back to existence.
        return True, path

    checks = []
    checks.append(("AVH folder", os.path.isdir(AVH_DIR), AVH_DIR))
    checks.append(("test_video.py", os.path.isfile(AVH_TEST_SCRIPT), AVH_TEST_SCRIPT))

    ok, detail = _status_for_path(AVH_FUSION_CKPT)
    checks.append(("Fusion checkpoint (AVH-Align_AV1M.pt)", ok, detail))

    checks.append(("av_hubert/avhubert (clone AV-HuBERT)", os.path.isdir(AVH_AVHUBERT_DIR), AVH_AVHUBERT_DIR))

    ok, detail = _status_for_path(AVH_FACE_PREDICTOR)
    checks.append(("Face landmark predictor (dlib)", ok, detail))

    ok, detail = _status_for_path(AVH_MEAN_FACE)
    checks.append(("Mean face (20words_mean_face.npy)", ok, detail))

    ok, detail = _status_for_path(AVH_AVHUBERT_CKPT)
    checks.append(("AV-HuBERT checkpoint (self_large_vox_433h.pt)", ok, detail))

    return checks


def _get_readable_ckpt_path(path: str, tmp_name: str = "AVH-Align_AV1M.pt", force_tmp: bool = False):
    """
    Return a path we can read in subprocesses.

    If reading original checkpoint raises PermissionError, copy to /tmp.
    If force_tmp is True, always copy to /tmp (work around sandbox/subprocess permission issues).
    """
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


def run_avh_on_video(video_path: str, timeout: int = 300, python_exe: str | None = None, keep_temp: bool = False):
    """
    Run AVH/test_video.py on a video file.

    Returns:
      - (success, score_or_error_message) if keep_temp=False
      - (success, score, audio_path) if keep_temp=True
    """
    if not os.path.exists(AVH_TEST_SCRIPT):
        return (
            (False, "AVH test script not found. Ensure AVH repo is present.", None)
            if keep_temp
            else (False, "AVH test script not found. Ensure AVH repo is present.")
        )
    if not os.path.exists(AVH_FUSION_CKPT):
        return (
            (False, "AVH-Align checkpoint not found.", None)
            if keep_temp
            else (False, "AVH-Align checkpoint not found (checkpoints/AVH-Align_AV1M.pt).")
        )
    if not os.path.exists(AVH_AVHUBERT_CKPT):
        return (
            (False, "AV-HuBERT checkpoint not found.", None)
            if keep_temp
            else (False, "AV-HuBERT checkpoint not found (self_large_vox_433h.pt).")
        )

    from subprocess_utils import safe_read_json, run_subprocess_capture, validate_python_exe

    try:
        py = validate_python_exe(python_exe)
    except Exception as e:
        return (False, str(e), None) if keep_temp else (False, str(e))

    fusion_path = _get_readable_ckpt_path(AVH_FUSION_CKPT, force_tmp=True)
    avhubert_path = _get_readable_ckpt_path(AVH_AVHUBERT_CKPT, "self_large_vox_433h.pt", force_tmp=True)

    json_fd, json_out_path = tempfile.mkstemp(prefix="avh_score_", suffix=".json")
    os.close(json_fd)
    cmd = [
        py,
        "test_video.py",
        "--video",
        os.path.abspath(video_path),
        "--fusion_ckpt",
        fusion_path,
        "--avhubert_ckpt",
        avhubert_path,
        "--json_out",
        json_out_path,
    ]
    if keep_temp:
        cmd.append("--keep_temp")

    try:
        run_res = run_subprocess_capture(cmd, cwd=AVH_DIR, timeout_s=timeout)
        payload = safe_read_json(json_out_path)

        if payload is None:
            out = (run_res.get("stdout") or "") + (run_res.get("stderr") or "")
            err = out or ("Timed out" if run_res.get("timed_out") else "AVH failed without JSON output.")
            return (False, err, None) if keep_temp else (False, err)

        if payload.get("success"):
            score = float(payload.get("score"))
            if keep_temp:
                return True, score, payload.get("audio_path")
            return True, score

        err = payload.get("error") or "AVH pipeline failed."
        return (False, err, None) if keep_temp else (False, err)
    finally:
        try:
            os.unlink(json_out_path)
        except Exception:
            pass


def run_avh_unsupervised_on_video(
    video_path: str,
    timeout: int = 300,
    python_exe: str | None = None,
    keep_temp: bool = False,
):
    """
    Run AVH/test_video_unsupervised.py on a video file.

    Returns:
      - (success, score_or_error) if keep_temp=False
      - (success, score, audio_path) if keep_temp=True
    """
    if not os.path.exists(AVH_TEST_UNSUP_SCRIPT):
        return (
            (False, "Unsupervised AVH script not found.", None) if keep_temp else (False, "Unsupervised AVH script not found.")
        )
    if not os.path.exists(AVH_AVHUBERT_CKPT):
        return (
            (False, "AV-HuBERT checkpoint not found.", None) if keep_temp else (False, "AV-HuBERT checkpoint not found.")
        )

    from subprocess_utils import safe_read_json, run_subprocess_capture, validate_python_exe

    try:
        py = validate_python_exe(python_exe)
    except Exception as e:
        return (False, str(e), None) if keep_temp else (False, str(e))

    avhubert_path = _get_readable_ckpt_path(AVH_AVHUBERT_CKPT, "self_large_vox_433h.pt", force_tmp=True)

    json_fd, json_out_path = tempfile.mkstemp(prefix="avh_unsup_score_", suffix=".json")
    os.close(json_fd)
    cmd = [
        py,
        "test_video_unsupervised.py",
        "--video",
        os.path.abspath(video_path),
        "--avhubert_ckpt",
        avhubert_path,
        "--json_out",
        json_out_path,
    ]
    if keep_temp:
        cmd.append("--keep_temp")

    try:
        run_res = run_subprocess_capture(cmd, cwd=AVH_DIR, timeout_s=timeout)
        payload = safe_read_json(json_out_path)

        if payload is None:
            out = (run_res.get("stdout") or "") + (run_res.get("stderr") or "")
            err = out or ("Timed out" if run_res.get("timed_out") else "AVH failed without JSON output.")
            return (False, err, None) if keep_temp else (False, err)

        if payload.get("success"):
            score = float(payload.get("score"))
            if keep_temp:
                return True, score, payload.get("audio_path")
            return True, score

        err = payload.get("error") or "Unsupervised AVH pipeline failed."
        return (False, err, None) if keep_temp else (False, err)
    finally:
        try:
            os.unlink(json_out_path)
        except Exception:
            pass


def run_avh_from_npz(npz_bytes: bytes, fusion_ckpt_path: str):
    """
    Score using only the Fusion checkpoint and pre-extracted .npz (no av_hubert/dlib).

    .npz must have keys 'visual' and 'audio' (arrays shape (T, 1024)).
    Returns (success, score_or_error).
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        return False, "PyTorch is required: pip install torch"

    if not os.path.isfile(fusion_ckpt_path):
        return False, f"Fusion checkpoint not found: {fusion_ckpt_path}"

    ckpt_path = _get_readable_ckpt_path(fusion_ckpt_path)

    sys.path.insert(0, AVH_DIR)
    try:
        from model import FusionModel

        data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
        visual = data["visual"]
        audio = data["audio"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = FusionModel().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        visual_t = torch.from_numpy(visual).float().to(device)
        audio_t = torch.from_numpy(audio).float().to(device)

        visual_t = visual_t / (torch.linalg.norm(visual_t, ord=2, dim=-1, keepdim=True) + 1e-8)
        audio_t = audio_t / (torch.linalg.norm(audio_t, ord=2, dim=-1, keepdim=True) + 1e-8)

        with torch.no_grad():
            out = model(visual_t, audio_t)
            score = torch.logsumexp(-out, dim=0).squeeze().item()

        return True, score
    except Exception as e:
        return False, str(e)
    finally:
        if AVH_DIR in sys.path:
            sys.path.remove(AVH_DIR)

