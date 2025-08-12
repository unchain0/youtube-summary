"""Tests for utils.youtube_helpers."""

import subprocess
from typing import NoReturn

import pytest

from utils.youtube_helpers import download_and_read_subtitles, video_id_from_url


def test_video_id_from_url_simple_id() -> None:
    """Video ID remains unchanged when already an ID."""
    assert video_id_from_url("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_video_id_from_url_youtube_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extract ID via yt-dlp (--get-id)."""

    class FakeCompleted:
        def __init__(self) -> None:
            self.stdout = "abc123\n"

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> "FakeCompleted":
        del capture_output, text, check
        if "--get-id" not in cmd:
            pytest.fail("--get-id not present in yt-dlp command")
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert video_id_from_url("https://www.youtube.com/watch?v=whatever") == "abc123"


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
    assert download_and_read_subtitles("https://youtu.be/xyz") == ""
