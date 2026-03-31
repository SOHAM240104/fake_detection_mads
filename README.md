---
title: Fake Audio Detection
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Lightweight ML method to detect forged / synthetic audio
---

## What this repo contains

This repo has **two** detection pipelines and one unified Streamlit demo:

- **NOMA (audio-only)**: classical ML (hand-crafted features + SVM) via Mozilla’s `fake-audio-detection`.
- **AVH-Align (audio-visual)**: AV-HuBERT feature extraction + fusion model score (lip–speech mismatch).
- **Unified demo**: `unified_deepfake_app.py` exposes:
  - `NOMA (Audio-Only)`
  - `AVH-Align (Audio-Visual)`
  - `Combined (AVH → NOMA)` + optional **Grad-CAM evidence** + optional **robustness delta** + **evidence bundle export**.

## Quickstart (recommended)

### 1) Streamlit (venv) for the unified demo

```bash
cd /Users/soham/fake-audio-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements-venv.txt
streamlit run unified_deepfake_app.py
```

### 2) AVH pipeline (conda env `avh`)

AVH needs a separate environment (fairseq/omegaconf compatibility). Create it from:
- `environment-avh.yml`

Then point Streamlit sidebar **Python for AVH video** to your conda python:
- Example: `/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python`

## Forensics evidence outputs (panel-ready)

In **Combined (AVH → NOMA)**:
- **Grad-CAM**: mouth ROI overlays explain visual sensitivity (evidence, not proof).
- **Robustness delta**: baseline AVH score vs adversarially-perturbed score (feature-space hard misalignment).
- **Evidence bundle export**: download a `.zip` containing:
  - `manifest.json` with sha256 hashes + scores
  - extracted `audio.wav` + `mouth_roi.mp4`
  - Grad-CAM overlays + `index.json`
  - NOMA predictions CSV

## Notes
- `app.py` is an older NOMA-only demo; the main deliverable is `unified_deepfake_app.py`.
- You must download/copy required AVH checkpoints into the expected `AVH/` paths (see sidebar “AVH setup status” in the app).
