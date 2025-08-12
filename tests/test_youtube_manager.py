"""Tests for `YouTubeTranscriptManager` helpers and I/O."""

import subprocess
from pathlib import Path

import pytest

from src.youtube import YouTubeTranscriptManager


def test_get_video_urls_filters_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filter out restricted videos and keep public ones."""
    output = """
private\thttps://youtu.be/priv
needs_auth\thttps://youtu.be/auth
ok\thttps://youtu.be/ok1
\t https://youtu.be/ok2
https://youtu.be/ok3
""".strip()

    class FakeCompleted:
        """Fake subprocess.CompletedProcess with stdout only."""

        def __init__(self) -> None:
            self.stdout = output

    def fake_run(
        cmd: list[str], *, capture_output: bool, text: bool, check: bool,
    ) -> FakeCompleted:
        """Return our fake process after basic validation."""
        del capture_output, text, check
        assert cmd[0] == "yt-dlp"
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)

    yt = YouTubeTranscriptManager(base_dir="data/transcripts")
    urls = yt.get_video_urls_from_channel("https://www.youtube.com/@handle")
    assert urls == [
        "https://youtu.be/ok1",
        "https://youtu.be/ok2",
        "https://youtu.be/ok3",
    ]


def test_save_transcript_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure `save_transcript` writes the transcript to disk."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "transcripts"))

    def fake_fetch(*args: object, **kwargs: object) -> str:
        """Return a stable fake transcript string."""
        del args, kwargs
        return "hello world"

    monkeypatch.setattr(yt, "fetch_transcript", fake_fetch)
    out = yt.save_transcript("channelX", "https://youtu.be/abc123")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hello world"
