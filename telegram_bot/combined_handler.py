"""
Run Combined pipeline and push structured results to a Telegram chat.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
from typing import Any

from telegram import InputFile
from telegram.ext import ContextTypes


def _run_combined_sync(video_path: str, video_name: str) -> dict[str, Any]:
    from orchestrator.combined_runner import run_combined_avh_to_noma
    from telegram_bot.combined_settings import combined_run_kwargs

    kwargs = combined_run_kwargs(video_path=video_path, video_name=video_name)
    return run_combined_avh_to_noma(**kwargs)


def _gemini_sync(
    res: dict[str, Any],
    cam_idx: dict[str, Any] | None,
    *,
    use_unsup_avh: bool,
) -> list[tuple[str, str, str | None]]:
    """Return list of (section_title, markdown_or_empty, error)."""
    from integrations.research_chat.gemini_client import synthesize_ui_guide
    from ui.report_explain_payload import (
        XAI_SECTION_LABELS,
        SECTION_LABELS,
        build_combined_report_guide_payload,
        build_xai_standalone_payload,
    )

    out: list[tuple[str, str, str | None]] = []
    guide = build_combined_report_guide_payload(res, cam_idx, use_unsup_avh=use_unsup_avh)
    full_title = next(t for t in SECTION_LABELS if t[0] == "full")[1]
    text, err = synthesize_ui_guide(section_id="full", section_title=full_title, guide_payload=guide)
    out.append((f"Gemini — {full_title}", text, err))

    for sid, title in XAI_SECTION_LABELS:
        kind = "audio" if sid == "xai_audio" else "video"
        try:
            payload = build_xai_standalone_payload(kind, res, cam_idx)
        except Exception as e:  # noqa: BLE001
            out.append((f"Gemini — {title}", "", str(e)))
            continue
        text2, err2 = synthesize_ui_guide(section_id=sid, section_title=title, guide_payload=payload)
        out.append((f"Gemini — {title}", text2, err2))

    return out


async def process_video_message(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    video_path: str,
    video_name: str,
) -> None:
    """Run Combined pipeline and send results (text, plots, JSON, Gemini)."""
    from telegram_bot.plot_bundle import build_plot_items
    from telegram_bot.report_text import (
        build_verdict_and_export_payload,
        export_json_bytes,
        format_combined_summary_text,
        split_telegram_chunks,
    )
    from telegram_bot.combined_settings import resolve_use_unsup_avh

    loop = asyncio.get_running_loop()
    status_msg = await context.bot.send_message(chat_id, "Processing… Running Combined (AVH → NOMA). This may take several minutes.")

    use_unsup = resolve_use_unsup_avh()

    try:
        res = await loop.run_in_executor(None, lambda: _run_combined_sync(video_path, video_name))
    except Exception as e:  # noqa: BLE001
        await status_msg.edit_text(f"Combined pipeline crashed: {e}")
        return

    if not res.get("avh_ok"):
        err = res.get("avh_error") or "unknown error"
        await status_msg.edit_text(f"AVH stage failed: {err}")
        return

    await status_msg.edit_text("Combined finished. Sending results…")

    cam_idx = res.get("cam_idx") if isinstance(res.get("cam_idx"), dict) else None

    # ── Developer summary ───────────────────────────────────────────────
    t_dev = format_combined_summary_text(res)
    for chunk in split_telegram_chunks(t_dev):
        await context.bot.send_message(chat_id, chunk)

    # ── User summary + JSON ───────────────────────────────────────────
    user_text, export_payload = build_verdict_and_export_payload(res, use_unsup_avh=use_unsup)
    for chunk in split_telegram_chunks(user_text):
        await context.bot.send_message(chat_id, chunk)

    await context.bot.send_document(
        chat_id,
        document=InputFile(io.BytesIO(export_json_bytes(export_payload)), filename="final_combined_report.json"),
        caption="Structured export (same fields as Streamlit download).",
    )

    if res.get("bundle_bytes"):
        await context.bot.send_document(
            chat_id,
            document=InputFile(io.BytesIO(res["bundle_bytes"]), filename="evidence_bundle.zip"),
            caption="Evidence bundle (optional export).",
        )

    # ── Charts / images ────────────────────────────────────────────────
    await context.bot.send_message(chat_id, "Charts & figures (PNG):")
    try:
        items = build_plot_items(res)
    except Exception as e:  # noqa: BLE001
        await context.bot.send_message(chat_id, f"(Could not build all plots: {e})")
        items = []

    for payload, caption in items:
        try:
            if isinstance(payload, str):
                with open(payload, "rb") as f:
                    data = f.read()
            else:
                data = payload
            bio = io.BytesIO(data)
            bio.name = "plot.png"
            await context.bot.send_photo(chat_id, photo=bio, caption=caption[:1024])
            await asyncio.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            await context.bot.send_message(chat_id, f"Skipped one figure ({caption[:80]}…): {e}")

    # ── Gemini explanations ────────────────────────────────────────────
    await context.bot.send_message(chat_id, "Gemini explanations (may take a minute)…")
    try:
        gem_parts = await loop.run_in_executor(
            None,
            lambda: _gemini_sync(res, cam_idx, use_unsup_avh=use_unsup),
        )
    except Exception as e:  # noqa: BLE001
        await context.bot.send_message(chat_id, f"Gemini batch failed: {e}")
        return

    for title, text, err in gem_parts:
        if err:
            await context.bot.send_message(chat_id, f"{title}\nError: {err}")
            continue
        body = (text or "").strip()
        if not body:
            await context.bot.send_message(chat_id, f"{title}\n(empty response)")
            continue
        for chunk in split_telegram_chunks(f"{title}\n\n{body}"):
            await context.bot.send_message(chat_id, chunk)

    await context.bot.send_message(chat_id, "Done.")


async def download_telegram_video_to_temp(update, context) -> tuple[str, str] | None:
    """Download video or video document to a temp file. Returns (path, filename) or None."""
    msg = update.effective_message
    if not msg:
        return None

    if msg.video:
        vf = await msg.video.get_file()
        name = (msg.video.file_name or "video.mp4").strip() or "video.mp4"
        suffix = os.path.splitext(name)[1] or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="tg_combined_")
        os.close(fd)
        await vf.download_to_drive(path)
        return path, name

    if msg.document and (msg.document.mime_type or "").startswith("video/"):
        df = await msg.document.get_file()
        name = (msg.document.file_name or "video.mp4").strip() or "video.mp4"
        suffix = os.path.splitext(name)[1] or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="tg_combined_")
        os.close(fd)
        await df.download_to_drive(path)
        return path, name

    return None
