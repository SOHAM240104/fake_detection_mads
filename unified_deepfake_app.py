"""
Unified Streamlit App: NOMA (Audio-Only) + AVH-Align (Audio-Visual) Deepfake Detection
Panel-ready: interactive explanations, methodology, and live testing for both systems.
"""

import io
import os
import sys
import subprocess
import tempfile
import re

import streamlit as st
import pandas as pd
import altair as alt

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# NOMA model: use the original shipped model by default.
# NOTE: we deliberately ignore notebook-trained artifacts (e.g. fake_audio_detection.joblib)
# because they can pickle custom classes (like L2Normalizer) that are not importable
# from this Streamlit context, leading to AttributeError during joblib.load.
NOMA_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model", "noma-1"),
]
AVH_DIR = os.path.join(PROJECT_ROOT, "AVH")
AVH_TEST_SCRIPT = os.path.join(AVH_DIR, "test_video.py")
AVH_FUSION_CKPT = os.path.join(AVH_DIR, "checkpoints", "AVH-Align_AV1M.pt")
AVH_AVHUBERT_DIR = os.path.join(AVH_DIR, "av_hubert", "avhubert")
AVH_FACE_PREDICTOR = os.path.join(AVH_AVHUBERT_DIR, "content", "data", "misc", "shape_predictor_68_face_landmarks.dat")
AVH_MEAN_FACE = os.path.join(AVH_AVHUBERT_DIR, "content", "data", "misc", "20words_mean_face.npy")
AVH_AVHUBERT_CKPT = os.path.join(AVH_AVHUBERT_DIR, "self_large_vox_433h.pt")

# ─── Page config & CSS ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detection Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1rem; color: #5a6c7d; margin-bottom: 1.5rem; }
    .method-card { padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2563eb; background: #f8fafc; }
    .step-box    { padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 6px; background: #f1f5f9; font-size: 0.95rem; }
    .metric-big  { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    .stExpander  { border: 1px solid #e2e8f0; border-radius: 8px; }

    /* Pipeline flow diagram */
    .pipeline-flow {
        display: flex; align-items: center; justify-content: center; flex-wrap: wrap;
        gap: 4px; margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px; border: 1px solid #bae6fd;
    }
    .flow-step {
        padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; font-weight: 600; text-align: center;
        min-width: 80px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .flow-step.audio   { background: #3b82f6; color: white; }
    .flow-step.process { background: #8b5cf6; color: white; }
    .flow-step.ml      { background: #059669; color: white; }
    .flow-step.output  { background: #dc2626; color: white; }
    .flow-step.visual  { background: #ea580c; color: white; }
    .flow-arrow        { font-size: 1.2rem; color: #64748b; }
    .flow-label        { font-size: 0.7rem; color: #64748b; margin-top: 2px; }

    /* Insight / concept cards */
    .insight-row       { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .insight-card {
        flex: 1; min-width: 140px; padding: 1rem; border-radius: 10px; text-align: center;
        background: white; border: 2px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .insight-card .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .insight-card .title { font-weight: 700; color: #334155; font-size: 0.9rem; }
    .insight-card .desc  { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

    /* How it works timeline */
    .timeline { margin: 1rem 0; padding-left: 1.5rem; border-left: 3px solid #3b82f6; }
    .timeline-step {
        position: relative; margin-bottom: 1rem; padding: 0.75rem 1rem; background: #f8fafc;
        border-radius: 8px; border: 1px solid #e2e8f0; margin-left: 0.5rem;
    }
    .timeline-step::before {
        content: ''; position: absolute; left: -1.6rem; top: 1rem; width: 12px; height: 12px;
        background: #3b82f6; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 0 2px #3b82f6;
    }
    .timeline-step strong { color: #1e40af; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────
def get_noma_model_path():
    for p in NOMA_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def run_noma_prediction(model_path: str, audio_path: str = None, audio_bytes: io.BytesIO = None):
    from fake_audio_detection.model import predict_audio_blocks
    return predict_audio_blocks(model_path, audio_path or "", audio_bytes=audio_bytes)

def check_avh_setup():
    """Return list of (check_name, ok, detail) for AVH setup."""
    checks = []
    checks.append(("AVH folder", os.path.isdir(AVH_DIR), AVH_DIR))
    checks.append(("test_video.py", os.path.isfile(AVH_TEST_SCRIPT), AVH_TEST_SCRIPT))
    checks.append(("Fusion checkpoint (AVH-Align_AV1M.pt)", os.path.isfile(AVH_FUSION_CKPT), AVH_FUSION_CKPT))
    checks.append(("av_hubert/avhubert (clone AV-HuBERT)", os.path.isdir(AVH_AVHUBERT_DIR), AVH_AVHUBERT_DIR))
    checks.append(("Face landmark predictor (dlib)", os.path.isfile(AVH_FACE_PREDICTOR), AVH_FACE_PREDICTOR))
    checks.append(("Mean face (20words_mean_face.npy)", os.path.isfile(AVH_MEAN_FACE), AVH_MEAN_FACE))
    checks.append(("AV-HuBERT checkpoint (self_large_vox_433h.pt)", os.path.isfile(AVH_AVHUBERT_CKPT), AVH_AVHUBERT_CKPT))
    return checks


def run_avh_on_video(video_path: str, timeout: int = 300, python_exe: str = None, keep_temp: bool = False):
    """Run AVH test_video.py on a video file.
    Returns (success, score_or_error_message) or (success, score, audio_path) if keep_temp=True.
    python_exe: use this Python (e.g. conda env where AVH works) instead of current one.
    """
    if not os.path.exists(AVH_TEST_SCRIPT):
        return (False, "AVH test script not found. Ensure AVH repo is present.", None) if keep_temp else (False, "AVH test script not found. Ensure AVH repo is present.")
    if not os.path.exists(AVH_FUSION_CKPT):
        return (False, "AVH-Align checkpoint not found.", None) if keep_temp else (False, "AVH-Align checkpoint not found (checkpoints/AVH-Align_AV1M.pt).")
    if not os.path.exists(AVH_AVHUBERT_CKPT):
        return (False, "AV-HuBERT checkpoint not found.", None) if keep_temp else (False, "AV-HuBERT checkpoint not found (self_large_vox_433h.pt).")
    fusion_path = _get_readable_ckpt_path(AVH_FUSION_CKPT, force_tmp=True)
    avhubert_path = _get_readable_ckpt_path(AVH_AVHUBERT_CKPT, "self_large_vox_433h.pt", force_tmp=True)
    exe = (python_exe or "").strip() or sys.executable
    cmd = [exe, "test_video.py", "--video", os.path.abspath(video_path),
           "--fusion_ckpt", fusion_path, "--avhubert_ckpt", avhubert_path]
    if keep_temp:
        cmd.append("--keep_temp")
    try:
        result = subprocess.run(
            cmd,
            cwd=AVH_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (result.stdout or "") + (result.stderr or "")
        # Parse "Working directory: /path"
        work_dir_match = re.search(r"Working directory:\s+(\S+)", out)
        # Parse "DEEPFAKE SCORE: 6.0319"
        match = re.search(r"DEEPFAKE\s+SCORE:\s+([-\d.]+)", out, re.I)
        if match:
            score = float(match.group(1))
            if keep_temp and work_dir_match:
                work_dir = work_dir_match.group(1).strip()
                audio_path = os.path.join(work_dir, "audio.wav")
                if os.path.isfile(audio_path):
                    return True, score, audio_path
                return True, score, None
            return (True, score, None) if keep_temp else (True, score)
        if result.returncode != 0:
            err = out or f"Exit code {result.returncode}"
            return (False, err, None) if keep_temp else (False, err)
        return (False, out or "Could not parse score from output.", None) if keep_temp else (False, out or "Could not parse score from output.")
    except subprocess.TimeoutExpired:
        err = "AVH pipeline timed out (video may be long or AV-HuBERT not set up)."
        return (False, err, None) if keep_temp else (False, err)
    except Exception as e:
        return (False, str(e), None) if keep_temp else (False, str(e))


def _get_readable_ckpt_path(path: str, tmp_name: str = "AVH-Align_AV1M.pt", force_tmp: bool = False):
    """Return a path we can read. If original raises PermissionError, copy to /tmp.
    If force_tmp, always copy to /tmp (for subprocess; avoids sandbox permission issues).
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


def run_avh_from_npz(npz_bytes: bytes, fusion_ckpt_path: str):
    """Score using only the Fusion checkpoint and pre-extracted .npz (no av_hubert/dlib).
    .npz must have keys 'visual' and 'audio' (arrays shape (T, 1024)). Returns (success, score_or_error).
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


def _show_avh_score_or_error(ok, result):
    """Render AVH score or error in Streamlit (reused for video and .npz paths)."""
    if ok:
        score = result
        st.markdown("#### 🎯 Deepfake score")
        if score < 0:
            score_bg, score_border, score_label = "#dcfce7", "#22c55e", "Likely REAL"
        elif score > 4:
            score_bg, score_border, score_label = "#fee2e2", "#ef4444", "Likely FAKE"
        else:
            score_bg, score_border, score_label = "#fef9c3", "#eab308", "Uncertain"
        st.markdown(f"""
        <div style="padding:1.5rem; border-radius:12px; background:{score_bg}; border:3px solid {score_border}; text-align:center; margin:0.5rem 0;">
            <div style="font-size:2rem; font-weight:800; color:#0f172a;">{score:.4f}</div>
            <div style="font-size:0.95rem; font-weight:600; color:#475569; margin-top:0.25rem;">{score_label}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">Higher score ⇒ more likely deepfake</div>
        </div>
        """, unsafe_allow_html=True)
        if score < 0:
            st.success("Score is negative or low → more consistent with **real** (aligned) video.")
        elif score > 4:
            st.error("Score is high → suggests **deepfake** (audio-visual mismatch).")
        else:
            st.info("Score in middle range → interpret with caution; consider human review.")
    else:
        st.error("AVH pipeline failed. Check **🔧 AVH setup status** or use **Score from .npz** with a pre-extracted file.")
        with st.expander("Error details", expanded=True):
            st.text(result)


# ─── Sidebar: method selection ─────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ Deepfake Detection Lab")
st.sidebar.markdown("Choose a detection method to explore and run.")
method = st.sidebar.radio(
    "**Detection method**",
    ["NOMA (Audio-Only)", "AVH-Align (Audio-Visual)", "Combined (AVH → NOMA)"],
    index=0,
)
# Resolved checkpoint paths (what the app uses)
_noma_path = get_noma_model_path()
with st.sidebar.expander("📁 Model paths", expanded=False):
    st.markdown("**NOMA (audio)**")
    st.code(_noma_path or "— not found —", language=None)
    st.markdown("**AVH-Align (fusion)**")
    st.code(AVH_FUSION_CKPT, language=None)
    if _noma_path:
        st.caption(f"NOMA: {'✅ found' if os.path.isfile(_noma_path) else '❌ missing'}")
    st.caption(f"AVH: {'✅ found' if os.path.isfile(AVH_FUSION_CKPT) else '❌ missing'}")

# Python for AVH video pipeline — MUST use avh conda env (has correct omegaconf, fairseq, etc.)
def _find_avh_python():
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

_avh_default = _find_avh_python()
avh_python_path = st.sidebar.text_input(
    "**Python for AVH video** (required for video upload)",
    value=_avh_default or "",
    placeholder="/path/to/conda/envs/avh/bin/python",
    help="Must point to the avh conda env Python. The venv Python has wrong omegaconf and will fail.",
)

# ─── Shared intro ─────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ Deepfake Detection Lab</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Compare two approaches: <strong>NOMA</strong> (audio-only, lightweight ML) '
    'and <strong>AVH-Align</strong> (audio-visual, deep learning). Use the sidebar to switch methods.</p>',
    unsafe_allow_html=True,
)

with st.expander("📊 Comparison at a glance", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="padding:1rem; border-radius:10px; background:linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%); border:2px solid #3b82f6;">
            <div style="font-size:1.2rem; font-weight:700; color:#1e40af; margin-bottom:0.5rem;">🎧 NOMA</div>
            <div style="font-size:0.85rem; color:#1e3a8a;">Audio-only · Lightweight</div>
            <ul style="margin:0.5rem 0 0 1rem; font-size:0.9rem;">
                <li>Input: WAV / MP3</li>
                <li>Detects: TTS, voice cloning</li>
                <li>Features + SVM → Real/Fake</li>
                <li>Fast (seconds)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="padding:1rem; border-radius:10px; background:linear-gradient(180deg, #fef3c7 0%, #fde68a 100%); border:2px solid #f59e0b;">
            <div style="font-size:1.2rem; font-weight:700; color:#b45309; margin-bottom:0.5rem;">🎬 AVH-Align</div>
            <div style="font-size:0.85rem; color:#92400e;">Audio-visual · Deep learning</div>
            <ul style="margin:0.5rem 0 0 1rem; font-size:0.9rem;">
                <li>Input: Video (MP4)</li>
                <li>Detects: Lip–speech mismatch</li>
                <li>AV-HuBERT + Fusion → Score</li>
                <li>1–3 min (CPU)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    | | **NOMA** | **AVH-Align** |
    |---|---|---|
    | **Input** | Audio (WAV/MP3) | Video (MP4) with face + audio |
    | **Detects** | Synthetic speech | Lip–speech mismatch |
    | **Method** | Features + SVM | AV-HuBERT + Fusion MLP |
    | **Output** | Per-second Real/Fake | Single score (higher = faker) |
    """)

# ═══════════════════════════════════════════════════════════════════════════
#  NOMA (Audio-Only) interface
# ═══════════════════════════════════════════════════════════════════════════
if method == "NOMA (Audio-Only)":
    st.markdown("---")
    st.markdown("## 🎧 NOMA — Audio-Only Fake Detection")

    # Visual: pipeline flow diagram
    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step audio">🎵 Audio</div><div class="flow-label">WAV/MP3</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">✂️ Split</div><div class="flow-label">1s blocks</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">📊 Features</div><div class="flow-label">MFCC, spectral</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">📐 L2 norm</div><div class="flow-label">Normalize</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🤖 SVM</div><div class="flow-label">RBF kernel</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">✅ Verdict</div><div class="flow-label">Real / Fake</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Insight cards: Input | Model | Output
    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎤</div><div class="title">Input</div><div class="desc">Single audio file (speech)</div></div>
        <div class="insight-card"><div class="icon">🧮</div><div class="title">Model</div><div class="desc">Hand-crafted features + SVM</div></div>
        <div class="insight-card"><div class="icon">📋</div><div class="title">Output</div><div class="desc">Per-second Real/Fake + overall verdict</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does NOMA do?", expanded=True):
        st.markdown("""
        **NOMA** (from Mozilla AI’s fake-audio-detection) is a **lightweight, audio-only** system that decides whether 
        an audio clip is **real human speech** or **synthetic/fake** (e.g. TTS, voice cloning). It does **not** use video.
        - **Input:** A single audio file (WAV, MP3, OGG).
        - **Output:** Per-second predictions (Real/Fake) and an overall verdict, plus confidence.
        - **Use case:** Fast screening of speech recordings when you don’t have video (e.g. podcasts, calls).
        """)

    with st.expander("🔄 How does the process work?"):
        st.markdown("**Visual timeline:**")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-step"><strong>1. Load audio</strong> — Resampled to 22.05 kHz mono.</div>
            <div class="timeline-step"><strong>2. Split into 1-second blocks</strong> — Each block is analyzed independently.</div>
            <div class="timeline-step"><strong>3. Feature extraction</strong> — Chroma STFT, spectral centroid/bandwidth/rolloff, MFCCs (20), IMFCCs (13), RMS, ZCR, tonnetz.</div>
            <div class="timeline-step"><strong>4. L2 normalization</strong> — Feature vectors normalized to unit length.</div>
            <div class="timeline-step"><strong>5. SVM classification</strong> — RBF kernel predicts Fake (0) or Real (1) per block.</div>
            <div class="timeline-step"><strong>6. Aggregation</strong> — Overall verdict from block-wise predictions.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="step-box">📊 <strong>Why 1-second blocks?</strong> Balance between temporal detail and enough signal for stable features.</div>', unsafe_allow_html=True)

    with st.expander("🧠 ML/DL behind it"):
        st.markdown("""
        - **Model:** Scikit-learn **Pipeline**: L2 normalizer → **SVC** (RBF kernel, C=1, gamma='scale', probability=True).
        - **Training:** Supervised on labeled Real/Fake audio; features from librosa + custom IMFCCs.
        - **No deep learning:** Hand-crafted features + SVM → interpretable, fast, and small footprint.
        - **Loss:** SVM uses hinge loss; we report accuracy, precision, recall, F1 on a hold-out set.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try NOMA")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:1rem;">
        Upload an audio file below and click <strong>Run NOMA prediction</strong> to see per-second Real/Fake analysis.
    </div>
    """, unsafe_allow_html=True)

    noma_model_path = get_noma_model_path()
    if not noma_model_path:
        st.warning("No NOMA model found. Add one of: `model/fake_audio_detection.joblib` or `model/noma-1`.")
    else:
        upload = st.file_uploader("Upload an audio file (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"], key="noma_upload")
        if upload is not None:
            st.audio(upload, format=f"audio/{upload.name.split('.')[-1].lower()}")
        else:
            st.info("Upload an audio file above to run NOMA.")

        if st.button("Run NOMA prediction", key="noma_btn") and noma_model_path:
            if upload is None:
                st.error("Please upload an audio file first.")
            else:
                with st.spinner("Analyzing audio (feature extraction + SVM)…"):
                    try:
                        audio_bytes = io.BytesIO(upload.getvalue())
                        times, probas = run_noma_prediction(noma_model_path, audio_bytes=audio_bytes)
                        preds = probas.argmax(axis=1)
                        confidences = probas.max(axis=1)
                        preds_str = ["Fake" if i == 0 else "Real" for i in preds]
                        df = pd.DataFrame({
                            "Seconds": times,
                            "Prediction": preds_str,
                            "Confidence": confidences,
                        })
                        df["Confidence Level"] = df.apply(
                            lambda r: "Uncertain" if r["Confidence"] < 0.3 else r["Prediction"],
                            axis=1,
                        )

                        st.markdown("#### 📈 Prediction by 1-second blocks")
                        chart = (
                            alt.Chart(df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Seconds:O", title="Seconds"),
                                y=alt.value(30),
                                color=alt.Color(
                                    "Confidence Level:N",
                                    scale=alt.Scale(
                                        domain=["Fake", "Real", "Uncertain"],
                                        range=["#ef4444", "#22c55e", "#94a3b8"],
                                    ),
                                ),
                                tooltip=["Seconds", "Prediction", "Confidence"],
                            )
                            .properties(width=700, height=150)
                        )
                        st.altair_chart(chart, width="stretch")

                        st.markdown("#### 🎯 Overall verdict")
                        if all(p == "Real" for p in preds_str):
                            st.success("**Verdict: REAL** — No synthetic segments detected.")
                        elif all(p == "Fake" for p in preds_str):
                            st.error("**Verdict: FAKE** — Audio appears synthetic across blocks.")
                        else:
                            st.warning("**Verdict: MIXED** — Some blocks detected as fake.")
                        n_blocks = len(preds_str)
                        mean_conf = float(confidences.mean())
                        st.markdown(f"""
                        <div class="step-box" style="margin-top:0.5rem;">
                            📊 <strong>Summary:</strong> {n_blocks} block(s) analyzed · Mean confidence: {mean_conf:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Analysis: NOMA (Mozilla-style audio-only pipeline).")
                    except Exception as e:
                        st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════
#  AVH-Align (Audio-Visual) interface
# ═══════════════════════════════════════════════════════════════════════════
elif method == "AVH-Align (Audio-Visual)":
    st.markdown("---")
    st.markdown("## 🎬 AVH-Align — Audio-Visual Deepfake Detection")

    # Visual: pipeline flow diagram
    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step visual">🎬 Video</div><div class="flow-label">MP4 + face</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">👤 Face & mouth</div><div class="flow-label">dlib, 96×96 ROI</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🧠 AV-HuBERT</div><div class="flow-label">1024-d features</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🔀 Fusion MLP</div><div class="flow-label">audio+visual</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">📈 Score</div><div class="flow-label">higher = faker</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Insight cards
    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎥</div><div class="title">Input</div><div class="desc">Video with face + audio</div></div>
        <div class="insight-card"><div class="icon">🔗</div><div class="title">Idea</div><div class="desc">Lip–speech alignment</div></div>
        <div class="insight-card"><div class="icon">📊</div><div class="title">Output</div><div class="desc">Single deepfake score</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does AVH-Align do?", expanded=True):
        st.markdown("""
        **AVH-Align** (from [CVPR 2025](https://github.com/SOHAM240104/AVH)) detects deepfakes by checking whether 
        **mouth movements and speech are in sync**. It uses both **video** and **audio**.
        - **Input:** A video file (e.g. MP4) with a visible face and audio.
        - **Output:** A **deepfake score**. **Higher score ⇒ more likely fake** (mouth-speech mismatch).
        - **Use case:** Detecting face-swapped or lip-synced videos (e.g. talking-head deepfakes).
        """)

    with st.expander("🔄 How does the process work?"):
        st.markdown("**Visual timeline:**")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-step"><strong>1. Preprocessing</strong> — Face detection (dlib), 68-point landmarks, mouth ROI crop (96×96). Audio extracted to WAV (16 kHz).</div>
            <div class="timeline-step"><strong>2. AV-HuBERT</strong> — Visual: mouth frames → 1024-d vectors. Audio: log-filterbank → 1024-d vectors (time-aligned).</div>
            <div class="timeline-step"><strong>3. Fusion & scoring</strong> — L2-normalized audio + visual features → MLP → single score (higher when lip–speech is out of sync).</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="step-box">🔬 <strong>Key idea:</strong> Real videos have natural lip–speech alignment; '
            'many deepfakes break this alignment, which AVH-Align learns to detect.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("🧠 ML/DL behind it"):
        st.markdown("""
        - **AV-HuBERT:** Transformer-based audio-visual representation (Facebook Research); pretrained on 433h of LRS3/VoxCeleb.
        - **FusionModel:** Lightweight MLP (visual_proj → 512, audio_proj → 512; concat → 1024 → 512 → 256 → 128 → 1).
        - **Training:** Unsupervised / contrastive-style on audio-visual alignment; fine-tuned on deepfake datasets (e.g. AV-Deepfake1M).
        - **Inference:** L2-normalize features → FusionModel → logsumexp over time → single deepfake score.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try AVH-Align")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:#fffbeb; border:1px solid #fde68a; margin-bottom:1rem;">
        Upload a video (face + audio) below and click <strong>Run AVH-Align</strong>. Processing may take 1–3 minutes on CPU.
    </div>
    """, unsafe_allow_html=True)

    avh_checks = check_avh_setup()
    avh_full_ready = all(ok for _, ok, _ in avh_checks)
    # Video path works if script + fusion ckpt exist; use sidebar "Python for AVH" if you use another env.
    avh_video_ready = os.path.exists(AVH_TEST_SCRIPT) and os.path.exists(AVH_FUSION_CKPT)
    avh_fusion_only = os.path.exists(AVH_FUSION_CKPT)

    with st.expander("🔧 AVH setup status & how to fix", expanded=not avh_full_ready):
        for name, ok, path in avh_checks:
            st.markdown(f"{'✅' if ok else '❌'} **{name}** — `{path}`")
        st.markdown("**Tip:** If you already made AVH work in another env (e.g. `conda activate avh`), set **Python for AVH video** in the sidebar to that Python path — then video upload will use it and no clone is needed in this app's env.")
        if not avh_full_ready:
            st.markdown("---")
            st.markdown("**Full setup (only if you want video → score in this app's env):**")
            st.markdown("""
            1. **Clone AV-HuBERT inside AVH** (required):
               ```bash
               cd AVH && git clone https://github.com/facebookresearch/av_hubert.git && cd av_hubert/avhubert
               git submodule init && git submodule update
               cd ../fairseq && pip install --editable . && cd ../avhubert
               ```
            2. **Install Python deps** (use a Python 3.10 env; fairseq needs it):
               ```bash
               pip install torch torchvision torchaudio
               pip install opencv-python dlib librosa python_speech_features scikit-video sentencepiece
               pip install "numpy<1.24" "omegaconf>=2.1" "hydra-core>=1.1"
               ```
            3. **Download face model & mean face** (from inside `AVH/av_hubert/avhubert`):
               ```bash
               mkdir -p content/data/misc
               curl -L -o content/data/misc/shape_predictor_68_face_landmarks.dat.bz2 \\
                 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
               bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
               curl -L -o content/data/misc/20words_mean_face.npy \\
                 https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
               ```
            4. **Download AV-HuBERT checkpoint** (~1 GB, in `AVH/av_hubert/avhubert`):
               ```bash
               curl -L -o self_large_vox_433h.pt \\
                 https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
               ```
            5. **Install ffmpeg**: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

            Then restart this app and try again. Full details: [AVH README](https://github.com/SOHAM240104/AVH).
            """)

    if not avh_fusion_only:
        st.warning("AVH-Align Fusion checkpoint not found. Add `AVH/checkpoints/AVH-Align_AV1M.pt` to use AVH.")
    else:
        # Video path: use sidebar Python if set (env where AVH works)
        if avh_video_ready:
            st.markdown("**From video:**")
            upload_v = st.file_uploader("Upload a video file (MP4 with face + audio)", type=["mp4", "avi", "mov"], key="avh_upload")
            if upload_v is not None:
                st.video(upload_v)
            if avh_python_path:
                st.caption(f"Using Python: `{avh_python_path}`")
            else:
                st.error("Set **Python for AVH video** in the sidebar to your avh conda Python (e.g. `.../envs/avh/bin/python`). The venv Python will fail with omegaconf errors.")
            if st.button("Run AVH-Align on video", key="avh_btn") and upload_v is not None:
                if not (avh_python_path and os.path.isfile(avh_python_path)):
                    st.error("Set **Python for AVH video** in the sidebar to your avh conda Python. Example: `/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python`")
                else:
                    suffix = os.path.splitext(upload_v.name)[-1] or ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(upload_v.getvalue())
                        tmp_path = tmp.name
                    try:
                        with st.spinner("Running AVH pipeline (preprocess → AV-HuBERT → FusionModel). This may take 1–3 min on CPU…"):
                            ok, result = run_avh_on_video(tmp_path, timeout=900, python_exe=avh_python_path)
                    except Exception as e:
                        ok, result = False, str(e)
                    _show_avh_score_or_error(ok, result)
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            elif upload_v is None and not avh_full_ready:
                st.info("Upload a video above, or use **Score from .npz** below (needs only the Fusion checkpoint).")
        else:
            st.caption("Video pipeline needs `AVH/test_video.py`. Use **Score from .npz** below with just the checkpoint.")

        # .npz path: only Fusion checkpoint (no av_hubert, no dlib)
        st.markdown("**Or score from pre-extracted features (no clone needed):**")
        upload_npz = st.file_uploader("Upload pre-extracted .npz (keys: `visual`, `audio`)", type=["npz"], key="avh_npz")
        if st.button("Score .npz with Fusion model", key="avh_npz_btn") and upload_npz is not None and avh_fusion_only:
            with st.spinner("Scoring with Fusion checkpoint…"):
                ok, result = run_avh_from_npz(upload_npz.getvalue(), AVH_FUSION_CKPT)
            _show_avh_score_or_error(ok, result)
        if avh_fusion_only and not upload_npz:
            st.caption("If you have a .npz from a previous AVH feature extraction, upload it — only the Fusion checkpoint is used.")

# ═══════════════════════════════════════════════════════════════════════════
#  Combined (AVH → NOMA) interface
# ═══════════════════════════════════════════════════════════════════════════
elif method == "Combined (AVH → NOMA)":
    st.markdown("---")
    st.markdown("## 🔀 Combined — Video → AVH → NOMA")

    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step visual">🎬 Video</div><div class="flow-label">MP4</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">🎬 AVH</div><div class="flow-label">Lip–speech</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step audio">🎵 Extract</div><div class="flow-label">Audio WAV</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🤖 NOMA</div><div class="flow-label">TTS detection</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">📊 Both scores</div><div class="flow-label">AVH + NOMA</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎬</div><div class="title">Input</div><div class="desc">Single MP4 video</div></div>
        <div class="insight-card"><div class="icon">🔗</div><div class="title">Flow</div><div class="desc">AVH extracts audio → NOMA analyzes it</div></div>
        <div class="insight-card"><div class="icon">📊</div><div class="title">Output</div><div class="desc">AVH score + NOMA per-second Real/Fake</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does Combined do?", expanded=True):
        st.markdown("""
        **Combined** runs both detectors on a single video:
        1. **AVH-Align** processes the video (face + mouth crop, AV-HuBERT, Fusion) → **AVH score** (lip–speech alignment).
        2. The **extracted audio** from AVH preprocessing is passed to **NOMA** → **per-second Real/Fake** (TTS/synthetic detection).
        - **Use case:** Get both audio-visual and audio-only signals in one run.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try Combined")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%); border:1px solid #f59e0b; margin-bottom:1rem;">
        Upload a video below. AVH runs first, then the extracted audio is sent to NOMA. Total time: ~2–4 min on CPU.
    </div>
    """, unsafe_allow_html=True)

    noma_model_path = get_noma_model_path()
    avh_video_ready = os.path.exists(AVH_TEST_SCRIPT) and os.path.exists(AVH_FUSION_CKPT) and os.path.exists(AVH_AVHUBERT_CKPT)

    if not (noma_model_path and avh_video_ready):
        st.warning("Combined needs both NOMA model and AVH setup. Check sidebar **Model paths** and **Python for AVH video**.")
    elif not (avh_python_path and os.path.isfile(avh_python_path)):
        st.error("Set **Python for AVH video** in the sidebar to your avh conda Python.")
    else:
        upload_combined = st.file_uploader("Upload a video (MP4 with face + audio)", type=["mp4", "avi", "mov"], key="combined_upload")
        if upload_combined is not None:
            st.video(upload_combined)
        if st.button("Run Combined (AVH → NOMA)", key="combined_btn") and upload_combined is not None:
            suffix = os.path.splitext(upload_combined.name)[-1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(upload_combined.getvalue())
                tmp_path = tmp.name
            try:
                with st.spinner("Step 1/2: Running AVH (preprocess + lip–speech score)…"):
                    ok, avh_score, audio_path = run_avh_on_video(tmp_path, timeout=900, python_exe=avh_python_path, keep_temp=True)
                if ok and audio_path:
                    st.success(f"**AVH score:** {avh_score:.4f} (higher = more likely fake)")
                    with st.spinner("Step 2/2: Running NOMA on extracted audio…"):
                        try:
                            times, probas = run_noma_prediction(noma_model_path, audio_path=audio_path)
                            preds = probas.argmax(axis=1)
                            confidences = probas.max(axis=1)
                            preds_str = ["Fake" if i == 0 else "Real" for i in preds]
                            df = pd.DataFrame({
                                "Seconds": times,
                                "Prediction": preds_str,
                                "Confidence": confidences,
                            })
                            st.markdown("#### 📈 NOMA per-second predictions")
                            st.dataframe(df, width="stretch", hide_index=True)
                            fake_pct = 100 * (preds == 0).mean()
                            st.caption(f"Overall: {fake_pct:.1f}% blocks predicted Fake, {100-fake_pct:.1f}% Real")
                        except Exception as e:
                            st.error(f"NOMA failed: {e}")
                    # Clean up AVH temp dir
                    try:
                        work_dir = os.path.dirname(audio_path)
                        import shutil
                        shutil.rmtree(work_dir, ignore_errors=True)
                    except Exception:
                        pass
                elif ok:
                    st.success(f"**AVH score:** {avh_score:.4f}")
                    st.warning("Could not extract audio for NOMA (AVH temp dir not found).")
                else:
                    st.error("AVH pipeline failed.")
                    with st.expander("Error details", expanded=True):
                        st.text(avh_score)
            except Exception as e:
                st.error(str(e))
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        elif upload_combined is None:
            st.info("Upload a video above to run the combined pipeline.")

st.sidebar.markdown("---")
st.sidebar.markdown("**How to run (terminal)**")
st.sidebar.code("streamlit run unified_deepfake_app.py", language="bash")
st.sidebar.markdown("---")
st.sidebar.caption("NOMA: Mozilla-style audio-only · AVH-Align: CVPR 2025 audio-visual")
