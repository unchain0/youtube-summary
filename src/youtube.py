"""YouTube transcript management and retrieval utilities."""

import os
import random
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeVar

from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

try:
    from youtube_transcript_api.proxies import GenericProxyConfig
except ImportError:  # pragma: no cover - optional dependency surface
    GenericProxyConfig = None  # type: ignore[assignment]

# Newer versions also expose IpBlocked; import if available for better handling
try:  # pragma: no cover - optional dependency surface
    from youtube_transcript_api import IpBlocked
except ImportError:  # pragma: no cover - best-effort compatibility

    class IpBlocked(RequestBlocked):
        """Compatibility stub for youtube-transcript-api versions without IpBlocked."""


from .utils.logging_setup import logger
from .utils.youtube_helpers import (
    download_and_read_subtitles as _download_subs,
)
from .utils.youtube_helpers import (
    is_supported_video_url as _is_supported,
)
from .utils.youtube_helpers import (
    video_id_from_url as _vid_from_url,
)

T = TypeVar("T")
Entry = dict[str, Any]


class _Snippet(Protocol):
    """Protocol for a transcript snippet object."""

    text: str
    start: float
    duration: float


class _FetchedTranscriptProtocol(Protocol):
    """Protocol for FetchedTranscript-like objects."""

    def to_raw_data(self) -> list[Entry]:
        ...


class _IterableSnippets(Protocol, Iterable[_Snippet]):
    """Protocol for iterables yielding snippet-like objects."""



class _UseSubsFallbackError(Exception):
    """Internal control-flow exception carrying subtitles text."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__("Use subtitles fallback")


class _HasFetch(Protocol):
    """Minimal protocol for a transcript object supporting fetch()."""

    def fetch(
        self,
        preserve_formatting: bool = ...,  # noqa: FBT001
    ) -> list[Entry] | _FetchedTranscriptProtocol | _IterableSnippets:
        ...


class _HasFinders(Protocol, Iterable[Any]):
    """Minimal protocol for TranscriptList-like objects with finder methods."""

    def find_manually_created_transcript(self, languages: list[str]) -> _HasFetch: ...

    def find_generated_transcript(self, languages: list[str]) -> _HasFetch: ...

    def find_transcript(self, languages: list[str]) -> _HasFetch: ...


class YouTubeTranscriptManager:
    """Fetches and persists YouTube video transcripts for one or more channels.

    - Uses youtube-transcript-api to fetch transcripts.
    - Saves transcripts under data/transcripts/<channel_id or handle>/<video_id>.txt
    - Accepts either full video URLs or raw video IDs.
    """

    def __init__(self, base_dir: str = "data/transcripts") -> None:
        """Initialize manager with the base transcripts directory."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_video_urls_from_channel(self, channel_url: str) -> list[str]:
        """Return video URLs from a channel.

        Skips private/members-only when possible. Uses yt-dlp to print
        availability + URL per entry and filters out entries whose availability
        indicates restricted access.
        """
        # Configure proxy if provided via env (HTTP only)
        proxy_url = os.getenv("HTTP_URL", "").strip()
        command = ["yt-dlp", "--flat-playlist", "--print", "%(availability)s\t%(url)s"]
        if proxy_url:
            command += ["--proxy", proxy_url]
        command.append(channel_url)
        # Safe: invoking yt-dlp with explicit executable and argument list.
        # Constrain temporary files to data/tmp to avoid cluttering project root
        app_tmp = Path("data/tmp")
        app_tmp.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "TMP": str(app_tmp),
            "TEMP": str(app_tmp),
            "TMPDIR": str(app_tmp),
        })
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(app_tmp),
                env=env,
            )
        except TypeError:
            # Test doubles may not accept cwd/env; fallback without them
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        urls: list[str] = []
        blocked = {
            "private",
            "needs_auth",
            "login_required",
            "subscriber_only",
            "premium_only",
        }
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if "\t" in line:
                availability, url = line.split("\t", 1)
                avail = (availability or "").strip().lower()
                if avail in blocked:
                    logger.debug("Restricted video ({}): {}", avail, url)
                    continue
                url = url.strip()
                if not _is_supported(url):
                    logger.debug("Unsupported content skipped: {}", url)
                    continue
                urls.append(url)
            else:
                # Fallback: only URL was printed; keep it
                url = line.strip()
                if not _is_supported(url):
                    logger.debug("Unsupported content skipped: {}", url)
                    continue
                urls.append(url)
        return urls

    # video ID extraction is provided by youtube_helpers.video_id_from_url

    def _create_ytt_api(self) -> YouTubeTranscriptApi:
        """Create a YouTubeTranscriptApi client honoring HTTP proxy if set."""
        http_url = os.getenv("HTTP_URL", "").strip()
        if GenericProxyConfig is not None and http_url:
            return YouTubeTranscriptApi(
                proxy_config=GenericProxyConfig(
                    http_url=http_url or None,
                ),
            )
        return YouTubeTranscriptApi()

    def _with_retries(self, fn: Callable[[], T], *, attempts: int = 3) -> T:
        """Execute a callable with small backoff retries for transient failures.

        Does not retry RequestBlocked to avoid hammering blocked endpoints.
        """
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                return fn()
            except (RequestBlocked, IpBlocked) as e:
                last_err = e
                break
            except Exception as e:  # noqa: BLE001 - retry unknown transient errors
                last_err = e
                if i == attempts - 1:
                    break
                time.sleep([1, 3, 7][i] + random.uniform(0, 0.4))  # noqa: S311
        if last_err is None:
            msg_unreachable = "Unreachable: no exception recorded"
            raise RuntimeError(msg_unreachable)
        raise last_err

    def _list_transcripts_with_retries(
        self,
        api: YouTubeTranscriptApi,
        video_id: str,
    ) -> _HasFinders:
        """List transcripts for a video with retry logic."""
        return self._with_retries(lambda: api.list(video_id))

    def _select_transcript(
        self,
        transcript_list: _HasFinders,
        languages: list[str],
    ) -> _HasFetch | None:
        """Pick the best transcript in order: manual, autogenerated, generic."""
        for finder in (
            transcript_list.find_manually_created_transcript,
            transcript_list.find_generated_transcript,
            transcript_list.find_transcript,
        ):
            try:
                return finder(languages)
            except NoTranscriptFound:
                continue
        return None

    def _fetch_entries_with_retries(self, transcript: _HasFetch) -> list[Entry]:
        """Fetch transcript entries preserving formatting with retry logic."""
        fetched = self._with_retries(
            lambda: transcript.fetch(preserve_formatting=True),
        )
        return self._normalize_entries(fetched)

    def _normalize_entries(self, data: object) -> list[Entry]:
        """Normalize various fetch return types to a list of dict entries.

        Supports both legacy list-of-dicts and the new FetchedTranscript object
        from youtube-transcript-api v1.x, plus iterables of dataclass snippets.
        """
        # New API: FetchedTranscript exposes to_raw_data()
        to_raw = getattr(data, "to_raw_data", None)
        if callable(to_raw):
            try:
                return list(to_raw())
            except Exception as e:  # noqa: BLE001
                logger.debug("to_raw_data() failed: {}", e)
        # Legacy/unknown variants
        if isinstance(data, dict):
            return [data]
        try:
            iterator = iter(data)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return []
        out: list[Entry] = []
        for item in iterator:  # type: ignore[assignment]
            if isinstance(item, dict):
                out.append(item)
            else:
                text = getattr(item, "text", None)
                if text:
                    start = getattr(item, "start", 0.0)
                    duration = getattr(item, "duration", 0.0)
                    out.append(
                        {
                            "text": str(text),
                            "start": float(start or 0.0),
                            "duration": float(duration or 0.0),
                        },
                    )
        return out

    def _assemble_text(self, entries: Sequence[Mapping[str, Any]]) -> str:
        """Join non-empty 'text' fields from transcript entries."""
        return " ".join(
            str(entry.get("text"))  # ensure str join
            for entry in entries
            if entry.get("text")
        ).strip()

    def _fallback_or_raise(
        self,
        url_or_id: str,
        languages: list[str],
        *,
        use_subs_fallback: bool,
        msg: str,
        err: Exception | None = None,
    ) -> NoReturn:
        """Either raise RuntimeError or raise control-flow to use subtitles text."""
        if use_subs_fallback:
            subs_text = _download_subs(url_or_id, languages=languages)
            if subs_text:
                raise _UseSubsFallbackError(subs_text)
        raise RuntimeError(msg) from err

    def fetch_transcript(
        self,
        url_or_id: str,
        languages: list[str] | None = None,
        *,
        use_subs_fallback: bool = False,
    ) -> str:
        """Return transcript text for a given video URL or ID.

        Tries provided languages first, then auto-detects.
        """
        try:
            ytt_api = self._create_ytt_api()
            video_id = _vid_from_url(url_or_id)
            languages = languages or ["pt", "en"]

            # 1) List transcripts
            try:
                transcript_list = self._list_transcripts_with_retries(ytt_api, video_id)
            except (RequestBlocked, IpBlocked) as e:
                msg = f"Transcript request blocked for video {video_id}: {e}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                    err=e,
                )
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to list transcripts for video {video_id}: {e}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                    err=e,
                )

            # 2) Handle empty list
            if not list(transcript_list):
                msg = f"No transcripts available for video {video_id}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                )

            # 3) Select best transcript
            transcript = self._select_transcript(transcript_list, languages)
            if transcript is None:
                msg = f"No valid transcript found for video {video_id}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                )

            # 4) Fetch entries
            try:
                entries = self._fetch_entries_with_retries(transcript)
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                msg = f"Transcript unavailable for video {video_id}: {e}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                    err=e,
                )
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to fetch transcript for video {video_id}: {e}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                    err=e,
                )

            # 5) Assemble text
            text = self._assemble_text(entries)
            if not text:
                msg = f"Empty transcript for video {video_id}"
                self._fallback_or_raise(
                    url_or_id,
                    languages,
                    use_subs_fallback=use_subs_fallback,
                    msg=msg,
                )
            else:
                return text
        except _UseSubsFallbackError as u:
            return u.text

    # Subtitles helpers are imported from youtube_helpers

    def save_transcript(
        self,
        channel_key: str,
        url_or_id: str,
        languages: list[str] | None = None,
        *,
        subs_fallback: bool = False,
    ) -> Path:
        """Fetch and save transcript to a file.

        Returns the file path.

        channel_key: a filesystem-friendly identifier for the channel
        (e.g., handle or custom name).
        """
        video_id = _vid_from_url(url_or_id)
        channel_dir = self.base_dir / channel_key
        channel_dir.mkdir(parents=True, exist_ok=True)
        out_path = channel_dir / f"{video_id}.txt"

        # Skip if the transcript file already exists and is non-empty.
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info("Skipping existing transcript {} -> {}", video_id, out_path)
            return out_path

        text = self.fetch_transcript(
            video_id,
            languages=languages,
            use_subs_fallback=subs_fallback,
        )
        if not text:
            # Treat empty content as a failure so callers can skip it
            msg = f"Empty transcript for video {video_id}"
            raise RuntimeError(msg)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(text)
        logger.success("Transcribed video {} -> {}", video_id, out_path)
        return out_path

    def process_channel(
        self,
        channel_url: str,
        channel_key: str | None = None,
        limit: int | None = None,
        languages: list[str] | None = None,
        *,
        subs_fallback: bool = False,
    ) -> list[Path]:
        """For a given channel URL, retrieve all video URLs and save transcripts.

        Returns list of saved file paths.
        """
        urls = self.get_video_urls_from_channel(channel_url)
        if limit is not None:
            urls = urls[:limit]
        # Derive a default channel key if not provided
        if not channel_key:
            # e.g., https://www.youtube.com/@Handle
            channel_key = (
                channel_url.rstrip("/").split("/")[-1].lstrip("@") or "channel"
            )
        saved: list[Path] = []
        for url in urls:
            try:
                saved.append(
                    self.save_transcript(
                        channel_key,
                        url,
                        languages=languages,
                        subs_fallback=subs_fallback,
                    ),
                )
            except (RuntimeError, OSError) as e:
                logger.warning("Failed transcript for {}: {}", url, e)
        return saved
