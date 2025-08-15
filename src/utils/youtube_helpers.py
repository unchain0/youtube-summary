"""Helpers for YouTube URL parsing, proxies, and subtitles via yt-dlp."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import urllib.parse as _up
from pathlib import Path
from typing import TYPE_CHECKING

from src.exceptions import YtDlpError
from src.utils.logging_setup import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


def _run_yt_dlp(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    temp_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run yt-dlp safely and return the completed process.

    Ensures the first argument is the expected executable and uses a list of args.
    Constrains temporary files to a known directory to avoid cluttering project root.
    """
    if not cmd or cmd[0] != "yt-dlp":
        msg = "First arg must be 'yt-dlp'"
        raise ValueError(msg)
    # Default temp/cwd: data/tmp
    app_tmp = temp_dir or Path("data/tmp")
    app_tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TMP": str(app_tmp),
            "TEMP": str(app_tmp),
            "TMPDIR": str(app_tmp),
        },
    )
    try:
        return subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd or app_tmp),
            env=env,
        )
    except TypeError:
        # Some test doubles may not accept cwd/env; fallback without them
        try:
            return subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as err:
            msg = "yt-dlp execution failed"
            raise YtDlpError(msg) from err
    except Exception as err:
        msg = "yt-dlp execution failed"
        raise YtDlpError(msg) from err


def _build_proxy_url() -> str | None:
    """Build a proxy URL from environment variables if provided.

    Env vars:
      - HTTP_URL: e.g. http://host:port or http://user:pass@host:port
    Returns a single proxy URL suitable for yt-dlp's --proxy.
    """
    http_url = os.getenv("HTTP_URL", "").strip()
    if http_url:
        return http_url
    return None


def _extract_id_from_parsed(parsed: _up.ParseResult) -> str | None:
    """Extract video ID from a parsed YouTube URL if possible.

    Supports:
    - youtu.be/<id>
    - youtube.com/watch?v=<id>
    - youtube.com/shorts/<id>
    - youtube.com/live/<id>
    Returns None when not extractable.
    """
    host = parsed.netloc
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate or None
    if host.endswith("youtube.com"):
        qs = _up.parse_qs(parsed.query)
        vid = qs.get("v", [""])[0]
        if vid:
            return vid
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] in {"shorts", "live"} and len(parts) > 1:
            return parts[1]
    return None


def channel_key_from_url(url: str) -> str:
    """Return a filesystem-friendly channel key derived from a channel URL."""
    return url.rstrip("/").split("/")[-1].lstrip("@") or "channel"


def video_id_from_url(url_or_id: str) -> str:
    """Extract a YouTube video ID or return the input if already an ID.

    Fast-path: parse the URL to avoid spawning yt-dlp per video. Falls back to
    yt-dlp only if heuristics fail.
    """
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed = _up.urlparse(url_or_id)
        fast = _extract_id_from_parsed(parsed)
        if fast:
            return fast
        # Fallback: robust extraction via yt-dlp as last resort
        try:
            proxy_url = _build_proxy_url()
            cmd = ["yt-dlp", "--get-id"]
            if proxy_url:
                cmd += ["--proxy", proxy_url]
            cmd.append(url_or_id)
            result = _run_yt_dlp(cmd)
            lines = result.stdout.strip().splitlines()
            vid = lines[0] if lines else ""
            if vid:
                return vid
        except (YtDlpError, IndexError) as err:
            logger.debug(
                "yt-dlp --get-id failed for {}: {}: {}",
                url_or_id,
                type(err).__name__,
                err,
            )
        # Last fallback: return original input
        return url_or_id
    # Input is already an ID or a non-YouTube URL: return as-is
    return url_or_id


def is_supported_video_url(url: str) -> bool:
    """Return True for supported YouTube URLs, False otherwise.

    Supported forms:
    - https://youtu.be/<id>
    - https://www.youtube.com/watch?v=<id>
    - https://www.youtube.com/shorts/<id>
    - https://www.youtube.com/live/<id>

    Explicitly excluded: embed and clip forms. Raw IDs (no URL) are supported.
    """
    if "youtube.com" not in url and "youtu.be" not in url:
        # If looks like a URL but not YouTube, reject; else treat as raw ID
        return not ("://" in url or url.startswith("www."))
    parsed = _up.urlparse(url)
    host = parsed.netloc
    if host.endswith("youtu.be"):
        return True
    if host.endswith("youtube.com"):
        qs = _up.parse_qs(parsed.query)
        if qs.get("v", [""])[0]:
            return True
        parts = [p for p in parsed.path.split("/") if p]
        return bool(parts and parts[0] in {"shorts", "live"} and len(parts) > 1)
    return False


def filter_pending_urls(
    per_channel_urls: dict[str, list[str]],
    transcripts_dir: Path,
) -> tuple[dict[str, list[str]], int]:
    """Filter out URLs that already have non-empty transcripts saved.

    Returns a tuple of (filtered_mapping, total_pending_count).
    """
    filtered: dict[str, list[str]] = {}
    total_pending = 0
    for ch, urls in per_channel_urls.items():
        ch_key = channel_key_from_url(ch)
        pending: list[str] = []
        for url in urls:
            # Skip unsupported content types (embed, clip, etc.)
            if not is_supported_video_url(url):
                continue
            vid = video_id_from_url(url)
            out_path = transcripts_dir / ch_key / f"{vid}.txt"
            try:
                if out_path.exists() and out_path.stat().st_size > 0:
                    continue
            except OSError:
                # Keep pending on filesystem errors
                pass
            pending.append(url)
        if pending:
            filtered[ch] = pending
            total_pending += len(pending)
    return filtered, total_pending


def _clean_caption_line(s: str) -> str:
    # Remove simple HTML/markup tags and inline cue formatting like <c>...</c>
    return re.sub(r"<[^>]+>", "", s)


def _collapse_consecutive_tokens(text: str) -> str:
    toks = text.split()
    if not toks:
        return ""
    out: list[str] = [toks[0]]
    for t in toks[1:]:
        if t != out[-1]:
            out.append(t)
    return " ".join(out)


def _vtt_to_text(vtt: str) -> str:
    lines: list[str] = []
    for line in vtt.splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip headers and meta
        if s.startswith("WEBVTT"):
            continue
        if s.lower().startswith(("kind:", "language:", "style:", "region:")):
            continue
        if "-->" in s:
            continue
        if s.isdigit():
            continue
        lines.append(_clean_caption_line(s))
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return _collapse_consecutive_tokens(joined)


def _srt_to_text(srt: str) -> str:
    lines: list[str] = []
    for line in srt.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+$", s):
            continue
        if "-->" in s:
            continue
        lines.append(_clean_caption_line(s))
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return _collapse_consecutive_tokens(joined)


def download_and_read_subtitles(  # noqa: C901, PLR0911, PLR0912
    url_or_id: str,
    languages: Iterable[str] | None = None,
) -> str:
    """Download subtitles (manual/auto) via yt-dlp and return plain text.

    Tries VTT and SRT. Returns empty string on failure.
    """
    langs = list(languages) if languages else ["pt", "en"]
    # Expand to include region-specific variants (e.g., pt-BR) and exact codes
    lang_patterns: list[str] = []
    for lang in langs:
        lang_patterns.append(lang)
        lang_patterns.append(f"{lang}.*")
    # De-duplicate while preserving order
    lang_patterns = list(dict.fromkeys(lang_patterns))
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        vid = video_id_from_url(url_or_id)
        out_tpl = str(tdir / f"{vid}.%(ext)s")
        lang_arg = ",".join(lang_patterns)
        proxy_url = _build_proxy_url()
        cmd = ["yt-dlp"]
        if proxy_url:
            cmd += ["--proxy", proxy_url]
        cmd += [
            "--skip-download",
            "--write-sub",
            "--write-auto-sub",
            "--sub-langs",
            lang_arg,
            "--retries",
            "3",
            "--sleep-requests",
            "1",
            "--socket-timeout",
            "20",
            "-o",
            out_tpl,
            url_or_id,
        ]
        try:
            # Confine yt-dlp temp and working dir to the temporary folder
            res = _run_yt_dlp(cmd, cwd=tdir, temp_dir=tdir)
        except YtDlpError as e:
            logger.debug(
                "yt-dlp failed to get subs for {} (langs={}): {}: {}",
                url_or_id,
                lang_arg,
                type(e).__name__,
                e,
            )
            return ""

        # Find downloaded subtitle file (prefer VTT then SRT)
        cand: Path | None = None
        for ext in ("vtt", "srt"):
            for fp in tdir.glob(f"{vid}*.{ext}"):
                cand = fp
                break
            if cand:
                break
        if not cand:
            # Retry by fetching any available subtitles, regardless of language
            cmd_any = ["yt-dlp"]
            if proxy_url:
                cmd_any += ["--proxy", proxy_url]
            cmd_any += [
                "--skip-download",
                "--write-sub",
                "--write-auto-sub",
                "--sub-langs",
                "all",
                "--retries",
                "3",
                "--sleep-requests",
                "1",
                "--socket-timeout",
                "20",
                "-o",
                out_tpl,
                url_or_id,
            ]
            try:
                res_any = _run_yt_dlp(cmd_any, cwd=tdir, temp_dir=tdir)
            except YtDlpError as e:
                logger.debug(
                    "yt-dlp failed to get any subs for {}: {}: {}",
                    url_or_id,
                    type(e).__name__,
                    e,
                )
                return ""
            for ext in ("vtt", "srt"):
                for fp in tdir.glob(f"{vid}*.{ext}"):
                    cand = fp
                    break
                if cand:
                    break
            if not cand:
                # Log outputs to help diagnose missing subs
                stderr_snip = (res.stderr or "").strip() if "res" in locals() else ""
                stderr_snip_any = (res_any.stderr or "").strip()
                stdout_snip = (res.stdout or "").strip() if "res" in locals() else ""
                stdout_snip_any = (res_any.stdout or "").strip()
                logger.debug(
                    (
                        "No subtitle files found for {} (langs={}). "
                        "yt-dlp stderr: {} | stdout: {} | "
                        "retry(all) stderr: {} | stdout: {}"
                    ),
                    url_or_id,
                    lang_arg,
                    stderr_snip[:500],
                    stdout_snip[:500],
                    stderr_snip_any[:500],
                    stdout_snip_any[:500],
                )
                return ""
        try:
            raw = cand.read_text(encoding="utf-8", errors="ignore")  # type: ignore[arg-type]
        except OSError:
            return ""
        if cand.suffix.lower() == ".vtt":  # type: ignore[union-attr]
            return _vtt_to_text(raw)
        if cand.suffix.lower() == ".srt":  # type: ignore[union-attr]
            return _srt_to_text(raw)
        return ""
