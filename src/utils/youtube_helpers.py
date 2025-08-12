"""Helpers for YouTube URL parsing, proxies, and subtitles via yt-dlp."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import urllib.parse as _up
from pathlib import Path
from typing import TYPE_CHECKING

from .logging_setup import logger

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
    env.update({
        "TMP": str(app_tmp),
        "TEMP": str(app_tmp),
        "TMPDIR": str(app_tmp),
    })
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
            raise RuntimeError(msg) from err
    except Exception as err:
        msg = "yt-dlp execution failed"
        raise RuntimeError(msg) from err


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


def video_id_from_url(url_or_id: str) -> str:
    """Extract a YouTube video ID or return the input if already an ID.

    Uses yt-dlp when available for robust extraction; falls back to URL parsing.
    """
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        try:
            proxy_url = _build_proxy_url()
            cmd = ["yt-dlp", "--get-id"]
            if proxy_url:
                cmd += ["--proxy", proxy_url]
            cmd.append(url_or_id)
            # Use wrapper that confines temp files and falls back for test doubles
            result = _run_yt_dlp(cmd)
            lines = result.stdout.strip().splitlines()
            vid = lines[0] if lines else ""
            if vid:
                return vid
        except (RuntimeError, IndexError) as err:
            logger.debug(
                "yt-dlp --get-id failed for {}: {}: {}",
                url_or_id,
                type(err).__name__,
                err,
            )

        parsed = _up.urlparse(url_or_id)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.lstrip("/")
        qs = _up.parse_qs(parsed.query)
        return qs.get("v", [url_or_id])[0]
    return url_or_id


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
        except RuntimeError as e:
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
            except RuntimeError as e:
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
