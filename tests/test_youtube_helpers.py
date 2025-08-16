"""Tests for utils.youtube_helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

from src.utils.youtube_helpers import (
    _build_proxy_url,
    _srt_to_text,
    _vtt_to_text,
    channel_key_from_url,
    download_subs,
    filter_pending_urls,
    is_supported,
    vid_from_url,
)


def test_video_id_from_url_simple_id() -> None:
    """Video ID remains unchanged when already an ID."""
    assert vid_from_url("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_video_id_from_url_youtube_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast-path should extract ID from watch URL without yt-dlp."""

    class FakeCompleted:
        def __init__(self) -> None:
            self.stdout = "abc123\n"

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> FakeCompleted:
        del capture_output, text, check
        if "--get-id" not in cmd:
            pytest.fail("--get-id not present in yt-dlp command")
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert vid_from_url("https://www.youtube.com/watch?v=whatever") == "whatever"


def test_download_and_read_subtitles_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return empty string when yt-dlp fails."""

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> NoReturn:
        del cmd, check, capture_output, text
        msg = "yt-dlp failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert download_subs("https://youtu.be/xyz") == ""


def test_video_id_from_url_fast_youtu_be_no_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure fast-path avoids spawning yt-dlp for youtu.be links."""

    def fail_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "yt-dlp should not be called for fast-path"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "run", fail_run)
    assert vid_from_url("https://youtu.be/ID123") == "ID123"


def test_video_id_from_url_fast_watch_no_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure fast-path avoids spawning yt-dlp for watch?v= links."""

    def fail_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "yt-dlp should not be called for fast-path"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "run", fail_run)
    url = "https://www.youtube.com/watch?v=abcDEF123"
    assert vid_from_url(url) == "abcDEF123"


def test_video_id_from_url_shorts_path_no_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure fast-path handles shorts form without spawning yt-dlp."""

    def fail_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "yt-dlp should not be called for fast-path"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "run", fail_run)
    assert vid_from_url("https://www.youtube.com/shorts/SHORT_ID") == "SHORT_ID"


def test_video_id_from_url_live_path_no_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure fast-path handles live form without spawning yt-dlp."""

    def fail_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "yt-dlp should not be called for fast-path"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "run", fail_run)
    assert vid_from_url("https://www.youtube.com/live/LIVE_ID") == "LIVE_ID"


def test_channel_key_from_url_variants() -> None:
    """Extract proper channel key from different channel URL formats."""
    assert channel_key_from_url("https://www.youtube.com/@Handle") == "Handle"
    assert (
        channel_key_from_url("https://www.youtube.com/c/SomeChannel/") == "SomeChannel"
    )
    assert channel_key_from_url("https://www.youtube.com/channel/UCXYZ/") == "UCXYZ"


def test_build_proxy_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return HTTP_URL when present; None when absent."""
    monkeypatch.setenv("HTTP_URL", "http://proxy:8080")
    assert _build_proxy_url() == "http://proxy:8080"
    monkeypatch.delenv("HTTP_URL", raising=False)
    assert _build_proxy_url() is None


def test_filter_pending_urls(tmp_path: Path) -> None:
    """Filter out URLs that already have a non-empty transcript on disk."""
    transcripts_dir = tmp_path
    ch = "https://www.youtube.com/@Foo"
    per_channel = {
        ch: [
            "https://youtu.be/ID_A",
            "https://www.youtube.com/watch?v=ID_B",
            "https://www.youtube.com/shorts/ID_C",
        ],
    }
    # Create Foo channel dir and one existing non-empty transcript (ID_A)
    foo_dir = transcripts_dir / "Foo"
    foo_dir.mkdir(parents=True, exist_ok=True)
    (foo_dir / "ID_A.txt").write_text("already here", encoding="utf-8")
    # Create a zero-byte transcript for ID_C: should NOT be considered existing
    (foo_dir / "ID_C.txt").write_bytes(b"")

    filtered, total = filter_pending_urls(per_channel, transcripts_dir)
    expected_total = 2
    assert total == expected_total
    assert set(filtered[ch]) == {
        "https://www.youtube.com/watch?v=ID_B",
        "https://www.youtube.com/shorts/ID_C",
    }


def test_vtt_to_text_basic() -> None:
    """Strip headers/timestamps and HTML, and collapse tokens for VTT."""
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "<c>Olá</c>\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "<b>mundo</b> mundo\n"
    )
    text = _vtt_to_text(vtt)
    assert text == "Olá mundo"


def test_srt_to_text_basic() -> None:
    """Strip indices/timestamps and HTML, and collapse tokens for SRT."""
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\nOlá\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n<b>mundo</b> mundo\n"
    )
    text = _srt_to_text(srt)
    assert text == "Olá mundo"


def test_download_and_read_subtitles_success_vtt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return parsed VTT text when yt-dlp downloads a VTT file."""

    class FakeCompleted:
        def __init__(self, stdout: str = "", stderr: str = "") -> None:
            self.stdout = stdout
            self.stderr = stderr

    # Monkeypatch the internal runner to create a VTT file in the given cwd
    def fake_run(
        _cmd: list[str],
        *,
        cwd: Path,
        temp_dir: Path,
    ) -> FakeCompleted:
        del temp_dir
        # Extract vid from output template in args (not strictly needed)
        # Instead derive from url: using known input id="id123"
        (Path(cwd) / "id123.en.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nOlá mundo\n",
            encoding="utf-8",
        )
        return FakeCompleted()

    monkeypatch.setattr("src.utils.youtube_helpers._run_yt_dlp", fake_run)
    out = download_subs("https://youtu.be/id123", languages=["pt"])
    assert out == "Olá mundo"


def test_is_supported_video_url_matrix() -> None:
    """Support standard videos, shorts, and live streams; exclude embed/clip."""
    assert is_supported("https://www.youtube.com/watch?v=ID") is True
    assert is_supported("https://youtu.be/ID") is True
    assert is_supported("https://www.youtube.com/shorts/ID") is True
    assert is_supported("https://www.youtube.com/live/ID") is True
    assert is_supported("https://www.youtube.com/embed/ID") is False
    assert is_supported("https://www.youtube.com/clip/ID") is False
    assert is_supported("ID") is True
