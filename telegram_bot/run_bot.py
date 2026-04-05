"""
Run the Telegram bot: research chat + Combined deepfake analysis on video upload.

  PYTHONPATH=. python -m telegram_bot.run_bot

Requires: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY (for Gemini explanations).
Optional: AVH_PYTHON, TELEGRAM_USE_UNSUP_AVH, LATE_FUSION_MODE, etc.
"""

from __future__ import annotations

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


async def cmd_start(update, context) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(
        "Commands:\n"
        "• Send a **video** (or video file) to run **Combined** (AVH → NOMA → charts + Gemini).\n"
        "• Send **text** for research: SerpAPI + News + Gemini summary.\n"
        "/help — more info\n"
        "/paths — show NOMA + AVH paths (same resolution as Streamlit)\n\n"
        "Not legal advice; screening signals only."
    )


async def cmd_help(update, context) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(
        "• **Video**: uploads run `run_combined_avh_to_noma` (same pipeline as Streamlit Combined). "
        "You will get numeric summaries, PNG charts, `final_combined_report.json`, and Gemini explanations.\n"
        "• **Text**: research assistant (needs Serp/News keys if configured).\n\n"
        "Paths match Streamlit: NOMA via `get_noma_model_path()`, AVH Python from `AVH_PYTHON` or the same "
        "candidates as the Streamlit sidebar (`config.AVH_PYTHON_ALLOWLIST`). "
        "/paths — show what this bot resolves on this machine.\n\n"
        "Env: `AVH_PYTHON` (optional if auto-find works), `TELEGRAM_INCLUDE_GRADCAM`, "
        "`LATE_FUSION_MODE`, `TELEGRAM_USE_UNSUP_AVH` (optional; default follows Streamlit when unset)."
    )


async def cmd_paths(update, context) -> None:
    from telegram_bot.combined_settings import describe_resolved_paths

    msg = update.effective_message
    if not msg:
        return
    text = describe_resolved_paths()
    await msg.reply_text(text[:4096])


async def on_text(update, context) -> None:
    from integrations.research_chat.chat_orchestrator import run_research_turn

    msg = update.effective_message
    if not msg:
        return
    text = (msg.text or "").strip()
    if not text:
        return

    hist = context.user_data.get("history")
    if not isinstance(hist, list):
        hist = []

    loop = asyncio.get_running_loop()
    turn = await loop.run_in_executor(
        None,
        lambda: run_research_turn(text, detection_context=None, history=hist),
    )

    reply = (turn.text or turn.error or "No response.")[:4096]
    hist.append({"role": "user", "content": text})
    hist.append({"role": "assistant", "content": reply})
    context.user_data["history"] = hist[-24:]

    await msg.reply_text(reply)


async def on_video(update, context) -> None:
    import os as _os

    from telegram_bot.combined_handler import download_telegram_video_to_temp, process_video_message

    msg = update.effective_message
    if not msg:
        return

    dl = await download_telegram_video_to_temp(update, context)
    if not dl:
        await msg.reply_text("Send a video recording or a **video document** (.mp4, etc.).")
        return

    path, name = dl
    chat_id = msg.chat_id
    try:
        await process_video_message(context=context, chat_id=chat_id, video_path=path, video_name=name)
    finally:
        try:
            _os.remove(path)
        except OSError:
            pass


async def _post_init_delete_webhook(application) -> None:
    """Polling conflicts with an active webhook; clear webhook before getUpdates."""
    await application.bot.delete_webhook(drop_pending_updates=False)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init_delete_webhook)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("paths", cmd_paths))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("Telegram bot polling… (video → Combined; text → research)")
    print("Open Telegram → search your bot by @username from BotFather → /start")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
