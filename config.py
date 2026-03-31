import os

# Centralized path/config constants for the Streamlit app.
#
# NOTE: AVH runtime dependencies (fairseq/omegaconf/etc.) are handled by
# invoking AVH scripts via a user-selected `python_exe` (see UI sidebar).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── NOMA (audio-only) paths ────────────────────────────────────────────────────
# NOTE: the app currently supports the repo's bundled NOMA model directory
# `model/noma-1`. The original joblib artifact is not present in this repo.
NOMA_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model", "noma-1"),
]

# ─── AVH paths ────────────────────────────────────────────────────────────────
AVH_DIR = os.path.join(PROJECT_ROOT, "AVH")
AVH_TEST_SCRIPT = os.path.join(AVH_DIR, "test_video.py")
AVH_TEST_UNSUP_SCRIPT = os.path.join(AVH_DIR, "test_video_unsupervised.py")

# AVH-Align fusion checkpoint (FusionModel MLP head)
AVH_FUSION_CKPT = os.path.join(AVH_DIR, "checkpoints", "AVH-Align_AV1M.pt")

# AV-HuBERT checkpoint + required AV-HuBERT code/data inside this repo
AVH_AVHUBERT_DIR = os.path.join(AVH_DIR, "av_hubert", "avhubert")
AVH_FACE_PREDICTOR = os.path.join(
    AVH_AVHUBERT_DIR,
    "content",
    "data",
    "misc",
    "shape_predictor_68_face_landmarks.dat",
)
AVH_MEAN_FACE = os.path.join(
    AVH_AVHUBERT_DIR,
    "content",
    "data",
    "misc",
    "20words_mean_face.npy",
)
AVH_AVHUBERT_CKPT = os.path.join(AVH_AVHUBERT_DIR, "self_large_vox_433h.pt")

# Grad-CAM evidence script
AVH_GRADCAM_SCRIPT = os.path.join(AVH_DIR, "gradcam_mouth_roi.py")

# Allowed python executables for running AVH subprocesses.
# This prevents arbitrary code execution via user-supplied `python_exe`.
AVH_PYTHON_ALLOWLIST = [
    "/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python",
    os.path.expanduser("~/miniforge3/envs/avh/bin/python"),
    os.path.expanduser("~/miniconda3/envs/avh/bin/python"),
    os.path.expanduser("~/anaconda3/envs/avh/bin/python"),
]

