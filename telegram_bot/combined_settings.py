"""
Defaults for Telegram Combined runs.

Path resolution matches Streamlit `unified_deepfake_app.py` where possible:
- NOMA: `detectors.noma.get_noma_model_path()` (same `NOMA_MODEL_CANDIDATES` / `model/noma-1`).
- AVH Python: `AVH_PYTHON` env if set, else first existing path from `config.AVH_PYTHON_ALLOWLIST`
  (same list Streamlit’s sidebar default + `AVH_PYTHON_ALLOWLIST_EXTRA` use).
- Unsupervised AVH default: same checkbox default as Streamlit — on when `AVH_FUSION_CKPT` is missing.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

from config import AVH_FUSION_CKPT, AVH_PYTHON_ALLOWLIST, PROJECT_ROOT
from config import STREAMLIT_GRADCAM_MAX_FUSION_FRAMES, STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP
from config import STREAMLIT_GRADCAM_REGION_TRACK_STRIDE, STREAMLIT_GRADCAM_SELECTION_MODE
from config import STREAMLIT_GRADCAM_TOP_K, get_late_fusion_mode


def find_default_avh_python() -> str | None:
    """Same candidate list as Streamlit `unified_deepfake_app._find_avh_python` (first hit wins)."""
    candidates = [
        "/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python",
        os.path.expanduser("~/miniforge3/envs/avh/bin/python"),
        os.path.expanduser("~/miniconda3/envs/avh/bin/python"),
        os.path.expanduser("~/anaconda3/envs/avh/bin/python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def resolve_avh_python() -> str | None:
    """
    Prefer explicit AVH_PYTHON, then first existing entry in config.AVH_PYTHON_ALLOWLIST,
    then Streamlit’s hardcoded default search.
    """
    explicit = (os.environ.get("AVH_PYTHON", "") or "").strip()
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)

    for p in AVH_PYTHON_ALLOWLIST:
        if p and os.path.isfile(p):
            return os.path.abspath(p)

    return find_default_avh_python()


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def resolve_use_unsup_avh() -> bool:
    """
    Streamlit sidebar: `use_unsup_avh = checkbox(..., value=not os.path.isfile(AVH_FUSION_CKPT))`.
    If TELEGRAM_USE_UNSUP_AVH is set, it overrides; otherwise use the same default rule.
    """
    raw = os.environ.get("TELEGRAM_USE_UNSUP_AVH", "").strip()
    if raw:
        return env_bool("TELEGRAM_USE_UNSUP_AVH", default=False)
    return not os.path.isfile(AVH_FUSION_CKPT)


def telegram_combined_persist_dir(video_name: str) -> str:
    """Stable folder under eval_runs/telegram_combined/ for artifacts."""
    safe = re.sub(r"[^\w.\-]+", "_", os.path.basename(video_name or "video").strip())[:80] or "video"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(PROJECT_ROOT, "eval_runs", "telegram_combined")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{ts}_{safe}")


def describe_resolved_paths() -> str:
    """Human-readable summary of paths used (for /start or logs)."""
    from detectors.noma import get_noma_model_path

    noma = get_noma_model_path()
    py = resolve_avh_python()
    lines = [
        "Resolved paths (same sources as Streamlit):",
        f"  NOMA model: {noma or '(not found)'}",
        f"  AVH fusion ckpt: {AVH_FUSION_CKPT}",
        f"  AVH Python: {py or '(not found — set AVH_PYTHON)'}",
        f"  Unsupervised AVH default: {resolve_use_unsup_avh()}",
        f"  Late fusion (env): {get_late_fusion_mode()}",
    ]
    return "\n".join(lines)


def combined_run_kwargs(*, video_path: str, video_name: str) -> dict:
    """
    Keyword arguments for orchestrator.combined_runner.run_combined_avh_to_noma.
    Override via env: TELEGRAM_USE_UNSUP_AVH, AVH_PYTHON, LATE_FUSION_MODE, TELEGRAM_SMART_CROP,
    TELEGRAM_DUMP_EMBEDDINGS, TELEGRAM_EXPORT_BUNDLE, TELEGRAM_NOMA_PERM_MAX_BLOCKS,
    TELEGRAM_PERSIST_DISK, TELEGRAM_CLEANUP_TEMP.

    NOMA and AVH Python use the same resolution as Streamlit (see module docstring).
    """
    from detectors.noma import get_noma_model_path

    noma_model_path = get_noma_model_path()
    if not noma_model_path:
        raise RuntimeError("NOMA model not found — set up model/noma-1 (same as Streamlit sidebar).")

    use_unsup = resolve_use_unsup_avh()
    py = resolve_avh_python()

    smart_crop = (os.environ.get("TELEGRAM_SMART_CROP", "face") or "face").strip().lower()
    if smart_crop not in ("auto", "off", "reel", "face"):
        smart_crop = "face"

    lf = (os.environ.get("LATE_FUSION_MODE", "") or "").strip().lower()
    late_fusion_mode = lf if lf else None  # None => combined_runner uses get_late_fusion_mode()

    persist = env_bool("TELEGRAM_PERSIST_DISK", default=True)
    cleanup = env_bool("TELEGRAM_CLEANUP_TEMP", default=True)
    persist_run_dir = telegram_combined_persist_dir(video_name) if persist else None

    perm_raw = os.environ.get("TELEGRAM_NOMA_PERM_MAX_BLOCKS", "0").strip()
    try:
        perm_n = int(perm_raw)
    except ValueError:
        perm_n = 0
    noma_perm = perm_n if perm_n > 0 else None

    return {
        "video_path": video_path,
        "video_name": video_name,
        "use_unsup_avh": use_unsup,
        "python_exe": py,
        "run_forensics_cam": env_bool("TELEGRAM_INCLUDE_GRADCAM", default=True),
        "forensics_top_k": int(STREAMLIT_GRADCAM_TOP_K),
        "forensics_selection_mode": str(STREAMLIT_GRADCAM_SELECTION_MODE),
        "forensics_min_temporal_gap": int(STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP),
        "forensics_max_fusion_frames": int(STREAMLIT_GRADCAM_MAX_FUSION_FRAMES),
        "region_track_stride": int(STREAMLIT_GRADCAM_REGION_TRACK_STRIDE),
        "run_robustness_delta": False,
        "adv_ckpt_path": "",
        "capture_attention": env_bool("TELEGRAM_CAPTURE_ATTENTION", default=False),
        "export_bundle": env_bool("TELEGRAM_EXPORT_BUNDLE", default=False),
        "noma_model_path": noma_model_path,
        "timeout": int(os.environ.get("TELEGRAM_COMBINED_TIMEOUT", "900")),
        "persist_run_dir": persist_run_dir,
        "cleanup_volatile_after_persist": cleanup if persist else False,
        "dump_embeddings_for_cmid": env_bool("TELEGRAM_DUMP_EMBEDDINGS", default=False),
        "noma_permutation_max_blocks": noma_perm,
        "smart_crop": smart_crop,
        "late_fusion_mode": late_fusion_mode or get_late_fusion_mode(),
    }
