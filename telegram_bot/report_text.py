"""
Plain-text summary + JSON export aligned with ui/integrated_verdict (without Streamlit).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from calibration_runtime import avh_score_to_calibrated_p_fake
from config import get_late_fusion_mode


def format_combined_summary_text(res: dict[str, Any]) -> str:
    """Headline metrics after Combined (mirrors _render_combined_demo_summary_from_res)."""
    avh_score = res.get("avh_score")
    avh_p_fake = res.get("p_avh_cal")
    if avh_p_fake is None and isinstance(avh_score, (int, float)):
        avh_p_fake = float(
            avh_score_to_calibrated_p_fake(float(avh_score), use_unsup_avh=bool(res.get("use_unsup_avh")))
        )

    noma_df = res.get("noma_df")
    noma_mean_p_fake = None
    noma_blocks = 0
    if isinstance(noma_df, pd.DataFrame) and "p_fake" in noma_df.columns and len(noma_df) > 0:
        noma_blocks = int(len(noma_df))
        noma_mean_p_fake = float(noma_df["p_fake"].astype(float).mean())

    p_fused = res.get("p_fused")
    fusion_tension = res.get("fusion_tension")
    fusion_verdict = res.get("fusion_verdict")
    fusion_tau = res.get("fusion_tau")
    late_mode = res.get("late_fusion_mode") or get_late_fusion_mode()

    lines = [
        "Results — Combined (AVH → NOMA)",
        f"Late fusion mode: {late_mode}",
        "",
        f"AVH p(fake) (cal.): {avh_p_fake:.3f}" if avh_p_fake is not None else "AVH p(fake): n/a",
        f"NOMA mean p(fake): {noma_mean_p_fake:.3f} ({noma_blocks} blocks)"
        if noma_mean_p_fake is not None
        else "NOMA mean p(fake): n/a",
        f"Late-fused p(fake): {float(p_fused):.3f}" if p_fused is not None else "Late-fused p(fake): n/a",
        f"Fusion tension: {float(fusion_tension):.3f}" if fusion_tension is not None else "Fusion tension: n/a",
        f"Fusion verdict: {fusion_verdict}" if fusion_verdict else "Fusion verdict: n/a",
    ]
    if fusion_tau is not None:
        lines.append(f"Fusion tau: {float(fusion_tau):.3f}")
    lines.append("")
    lines.append("Screening / evidence signals — not a legal verdict.")

    return "\n".join(lines)


def build_verdict_and_export_payload(res: dict[str, Any], *, use_unsup_avh: bool) -> tuple[str, dict[str, Any]]:
    """Match integrated_verdict export_payload + user-facing verdict string."""
    avh_score = res.get("avh_score")
    avh_p_fake = res.get("p_avh_cal")
    if avh_p_fake is None and isinstance(avh_score, (int, float)):
        avh_p_fake = float(
            avh_score_to_calibrated_p_fake(
                float(avh_score),
                use_unsup_avh=bool(res.get("use_unsup_avh", use_unsup_avh)),
            )
        )

    noma_df = res.get("noma_df")
    noma_mean_p_fake = None
    if isinstance(noma_df, pd.DataFrame) and "p_fake" in noma_df.columns and len(noma_df) > 0:
        noma_mean_p_fake = float(noma_df["p_fake"].astype(float).mean())

    p_fused_res = res.get("p_fused")
    fusion_tension = res.get("fusion_tension")
    fusion_tau = res.get("fusion_tau")
    fusion_verdict = res.get("fusion_verdict")

    blended_p_fake = None
    verdict = "Insufficient evidence"
    if p_fused_res is not None:
        blended_p_fake = float(p_fused_res)
        verdict = str(fusion_verdict) if fusion_verdict else "Uncertain"
    else:
        weights = {"avh": 0.55, "noma": 0.45}
        blend_terms = []
        if avh_p_fake is not None:
            blend_terms.append((weights["avh"], avh_p_fake))
        if noma_mean_p_fake is not None:
            blend_terms.append((weights["noma"], noma_mean_p_fake))
        if blend_terms:
            w_sum = sum(w for w, _ in blend_terms)
            blended_p_fake = float(sum(w * v for w, v in blend_terms) / max(w_sum, 1e-9))
        if blended_p_fake is not None:
            if blended_p_fake >= 0.65:
                verdict = "Likely FAKE"
            elif blended_p_fake <= 0.35:
                verdict = "Likely REAL"
            else:
                verdict = "Uncertain"

    inst = res.get("noma_confidence_instability")
    export_payload = {
        "verdict": verdict,
        "avh_score": avh_score,
        "avh_p_fake": avh_p_fake,
        "noma_mean_p_fake": noma_mean_p_fake,
        "blended_p_fake": blended_p_fake,
        "p_fused": float(p_fused_res) if p_fused_res is not None else None,
        "fusion_tension": float(fusion_tension) if fusion_tension is not None else None,
        "fusion_tau": float(fusion_tau) if fusion_tau is not None else None,
        "fusion_verdict": fusion_verdict,
        "cmid_status": res.get("cmid_status"),
        "cii": (float(inst["CII"]) if isinstance(inst, dict) and inst.get("CII") is not None else None),
        "temporal_corroboration": res.get("temporal_corroboration"),
    }
    user_lines = [
        "User summary",
        f"Screening verdict: {verdict}",
        f"Fused p(fake): {blended_p_fake:.3f}" if blended_p_fake is not None else "Fused p(fake): n/a",
        f"AVH vs NOMA tension: {float(fusion_tension):.3f}" if fusion_tension is not None else "Tension: n/a",
    ]
    return "\n".join(user_lines), export_payload


def export_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def split_telegram_chunks(text: str, limit: int = 4000) -> list[str]:
    """Split long markdown for Telegram (4096 max; keep under limit)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return chunks
