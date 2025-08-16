"""Tests for `YouTubeTranscriptManager` helpers and I/O."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Never

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest

import src.youtube as ymod
from src.exceptions import SkipVideoError
from src.youtube import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptManager,
)

# Tests use a small, explicit number of retry attempts
RETRIES = 3


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
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
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
    tmp_path: Path,
) -> None:
    """Ensure `save_transcript` writes the transcript to disk."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "transcripts"))

    def fake_fetch(*args: object, **kwargs: object) -> str:
        """Return a stable fake transcript string."""
        del args, kwargs
        return "hello world"

    yt.fetch_transcript = fake_fetch
    out = yt.save_transcript("channelX", "https://youtu.be/abc123")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hello world"


def test_save_transcript_skips_existing(
    tmp_path: Path,
) -> None:
    """`save_transcript` should skip rewriting an existing non-empty file."""
    base = tmp_path / "transcripts"
    yt = YouTubeTranscriptManager(base_dir=str(base))
    # Prepare existing file
    out = base / "channelX" / "abc123.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("original", encoding="utf-8")

    # Ensure our fake fetch would be called if not skipped
    def fail_fetch(*_a: object, **_k: object) -> str:
        msg = "fetch_transcript should not be called"
        raise AssertionError(msg)

    yt.fetch_transcript = fail_fetch
    returned = yt.save_transcript("channelX", "https://youtu.be/abc123")
    assert returned == out
    assert out.read_text(encoding="utf-8") == "original"


def test_with_retries_respects_non_retryable(tmp_path: Path) -> None:
    """_with_retries should not retry on blocked/disabled exceptions."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))
    calls = {"n": 0}

    def fn_td() -> int:
        calls["n"] += 1
        msg_disabled = "disabled"
        raise TranscriptsDisabled(msg_disabled)

    with pytest.raises(TranscriptsDisabled):
        yt._with_retries(fn_td, attempts=RETRIES)  # noqa: SLF001
    assert calls["n"] == 1  # no retries

    calls = {"n": 0}

    def fn_rb() -> int:
        calls["n"] += 1
        msg_blocked = "blocked"
        raise RequestBlocked(msg_blocked)

    with pytest.raises(RequestBlocked):
        yt._with_retries(fn_rb, attempts=RETRIES)  # noqa: SLF001
    assert calls["n"] == 1

    calls = {"n": 0}

    def fn_ib() -> int:
        calls["n"] += 1
        msg_ip = "ip blocked"
        raise IpBlocked(msg_ip)

    with pytest.raises(IpBlocked):
        yt._with_retries(fn_ib, attempts=RETRIES)  # noqa: SLF001
    assert calls["n"] == 1


def test_with_retries_retries_generic_then_succeeds(tmp_path: Path) -> None:
    """Retries generic errors and eventually succeeds."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < RETRIES:
            msg_temp = "temp"
            raise RuntimeError(msg_temp)
        return "ok"

    assert yt._with_retries(fn, attempts=RETRIES) == "ok"  # noqa: SLF001
    assert calls["n"] == RETRIES


def test_fetch_transcript_transcripts_disabled_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If listing raises TranscriptsDisabled, subtitles fallback returns text."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    class FakeApi:
        def list(self, _vid: str) -> object:
            msg_disabled = "disabled"
            raise TranscriptsDisabled(msg_disabled)

    monkeypatch.setattr(yt, "_create_ytt_api", lambda: FakeApi())
    # Force subtitle fallback path to return some text
    monkeypatch.setattr(
        ymod,
        "_download_subs",
        lambda _url, _languages=None: "subs",
    )
    assert (
        yt.fetch_transcript("https://youtu.be/abc", use_subs_fallback=True)
        == "subs"
    )


def test_fetch_transcript_transcripts_disabled_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without fallback, TranscriptsDisabled surfaces as SkipVideoError.

    This goes through the `fetch_transcript` path without subtitles fallback.
    """
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    class FakeApi:
        def list(self, _vid: str) -> object:
            msg_disabled = "disabled"
            raise TranscriptsDisabled(msg_disabled)

    monkeypatch.setattr(yt, "_create_ytt_api", lambda: FakeApi())
    with pytest.raises(SkipVideoError) as ei:
        yt.fetch_transcript(
            "https://youtu.be/abc",
            use_subs_fallback=False,
        )
    # SkipVideoError bubbles up
    assert "Transcripts disabled" in str(ei.value)


def test_fetch_entries_no_transcript_found_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If transcript fetch raises NoTranscriptFound, fallback returns subtitles text."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    class FakeTranscript:
        def fetch(
            self,
            _preserve_formatting: bool = True,  # noqa: FBT001, FBT002
        ) -> Never:
            msg_none = "none"
            raise NoTranscriptFound(msg_none)

    class FakeList:
        def __iter__(self) -> Iterator[int]:
            return iter([1])

        def find_manually_created_transcript(
            self, _langs: list[str],
        ) -> FakeTranscript:
            return FakeTranscript()

        def find_generated_transcript(
            self, _langs: list[str],
        ) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

        def find_transcript(
            self, _langs: list[str],
        ) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

    class FakeApi:
        def list(self, _vid: str) -> FakeList:
            return FakeList()

    monkeypatch.setattr(yt, "_create_ytt_api", lambda: FakeApi())
    monkeypatch.setattr(ymod, "_download_subs", lambda _url, _languages=None: "subs")
    assert yt.fetch_transcript("https://youtu.be/abc", use_subs_fallback=True) == "subs"


def test_fetch_entries_no_transcript_found_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When no transcript is found and fallback is disabled, raise RuntimeError."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    class FakeTranscript:
        def fetch(
            self,
            _preserve_formatting: bool = True,  # noqa: FBT001, FBT002
        ) -> Never:
            msg_none = "none"
            raise NoTranscriptFound(msg_none)

    class FakeList:
        def __iter__(self) -> Iterator[int]:
            return iter([1])

        def find_manually_created_transcript(
            self, _langs: list[str],
        ) -> FakeTranscript:
            return FakeTranscript()

        def find_generated_transcript(
            self, _langs: list[str],
        ) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

        def find_transcript(
            self, _langs: list[str],
        ) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

    class FakeApi:
        def list(self, _vid: str) -> FakeList:
            return FakeList()

    monkeypatch.setattr(yt, "_create_ytt_api", lambda: FakeApi())
    with pytest.raises(RuntimeError):
        yt.fetch_transcript("https://youtu.be/abc", use_subs_fallback=False)


def test_select_transcript_order_and_assemble(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selection tries manual, then generated, then generic. Joins snippets."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    class GoodTranscript:
        def fetch(
            self, _preserve_formatting: bool = True,  # noqa: FBT001, FBT002
        ) -> list[dict[str, float | str]]:
            return [
                {"text": "hello", "start": 0.0, "duration": 1.0},
                {"text": "world", "start": 1.0, "duration": 1.0},
            ]

    class FakeList:
        def __iter__(self) -> Iterator[int]:
            return iter([1])

        def find_manually_created_transcript(
            self, _langs: list[str],
        ) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

        def find_generated_transcript(
            self, _langs: list[str],
        ) -> GoodTranscript:
            return GoodTranscript()

        def find_transcript(self, _langs: list[str]) -> Never:
            vid_str = "vid"
            langs = ["pt", "en"]
            data = []
            raise NoTranscriptFound(vid_str, langs, data)

    class FakeApi:
        def list(self, _vid: str) -> FakeList:
            return FakeList()

    monkeypatch.setattr(yt, "_create_ytt_api", lambda: FakeApi())
    assert (
        yt.fetch_transcript("https://youtu.be/abc", use_subs_fallback=False)
        == "hello world"
    )


def test_normalize_entries_variants(tmp_path: Path) -> None:
    """Covers multiple input variants to _normalize_entries()."""
    yt = YouTubeTranscriptManager(base_dir=str(tmp_path / "irrelevant"))

    # 1) FetchedTranscript-like with to_raw_data
    class Fetched:
        def to_raw_data(self) -> list[dict[str, float | str]]:
            return [
                {"text": "a", "start": 0.0, "duration": 1.0},
                {"text": "b", "start": 1.0, "duration": 1.0},
            ]

    out1 = yt._normalize_entries(Fetched())  # noqa: SLF001
    expected_len = 2
    assert len(out1) == expected_len
    assert out1[0]["text"] == "a"

    # 2) Iterable of snippet-like objects
    class Snip:
        def __init__(self, t: str, s: float, d: float) -> None:
            self.text = t
            self.start = s
            self.duration = d

    out2 = yt._normalize_entries([Snip("x", 0.0, 1.0), Snip("y", 1.0, 1.0)])  # noqa: SLF001
    assert [e["text"] for e in out2] == ["x", "y"]

    # 3) Single dict
    out3 = yt._normalize_entries({"text": "one"})  # noqa: SLF001
    assert out3 == [{"text": "one"}]
