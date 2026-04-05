"""Tests for telegram_bot.report_text helpers."""

from telegram_bot.report_text import split_telegram_chunks


def test_split_telegram_chunks_respects_limit():
    s = "a\n\n" * 3000
    parts = split_telegram_chunks(s, limit=100)
    assert all(len(p) <= 100 for p in parts)
    assert len(parts) >= 1


def test_split_short_roundtrip():
    s = "hello\n\nworld"
    assert split_telegram_chunks(s, limit=4000) == [s]
