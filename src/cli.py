"""Command-line interface for fetching transcripts, indexing, and querying."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from src.rag import TranscriptRAG
from src.utils.logging_setup import logger
from src.youtube import YouTubeTranscriptManager


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch YouTube transcripts, index (FastEmbed), "
            "and query with Groq."
        ),
    )
    parser.add_argument(
        "channels",
        nargs="+",
        help=(
            "One or more YouTube channel URLs "
            "(e.g., https://www.youtube.com/@Handle)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos per channel to fetch",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="pt,en",
        help="Comma-separated language preferences for transcripts",
    )
    parser.add_argument(
        "--subs",
        action="store_true",
        help=(
            "Try to download subtitles (auto/manual) via yt-dlp "
            "when transcripts are not available"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from all transcripts instead of incremental add",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional question to query against the indexed transcripts",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="data/transcripts",
        help="Directory where transcripts are saved",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="data/vector_store",
        help="Directory where the Chroma vector store will persist",
    )
    return parser


def fetch_transcripts(
    channels: Iterable[str],
    transcripts_dir: Path,
    languages: list[str],
    limit: int | None,
    *,
    subs: bool,
) -> dict[str, list[Path]]:
    """Fetch transcripts for channels and save them under ``transcripts_dir``.

    Returns a mapping of channel URL to a list of saved transcript file paths.
    """
    yt = YouTubeTranscriptManager(base_dir=str(transcripts_dir))
    saved: dict[str, list[Path]] = {}
    for ch in channels:
        logger.info("Fetching transcripts for {} ...", ch)
        files = yt.process_channel(
            ch,
            limit=limit,
            languages=languages,
            subs_fallback=subs,
        )
        saved[ch] = files
        logger.success("Saved {} transcripts for {}", len(files), ch)
    return saved


def build_or_update_vector_store(
    channels: Iterable[str],
    transcripts_dir: Path,
    vector_dir: Path,
    *,
    rebuild: bool,
) -> TranscriptRAG:
    """Build or update the vector store from the available transcripts."""
    rag = TranscriptRAG(vector_dir=vector_dir)
    if rebuild:
        logger.info("Rebuilding vector store from all transcripts ...")
        if vector_dir.exists():
            logger.info("Removing existing vector store at {}", vector_dir)
            shutil.rmtree(vector_dir, ignore_errors=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        rag.index_transcripts(transcripts_dir)
        logger.success("Vector store rebuilt")
    else:
        logger.info("Incrementally updating vector store ...")
        for ch in channels:
            channel_key = ch.rstrip("/").split("/")[-1].lstrip("@") or "channel"
            ch_dir = transcripts_dir / channel_key
            rag.add_channel(ch_dir)
        logger.success("Vector store updated")
    return rag


def query_vector_store(rag: TranscriptRAG, question: str, k: int = 4) -> str:
    """Query the vector store and return the answer text."""
    logger.info("Question: {}", question)
    out = rag.query(question, k=k)
    answer = str(out.get("answer", ""))
    logger.info("Answer:\n{}", answer)
    return answer


# Aliases with camelCase as requested
def fetchTranscripts(  # noqa: N802
    channels: Iterable[str],
    transcripts_dir: Path,
    languages: list[str],
    limit: int | None,
    *,
    subs: bool,
) -> dict[str, list[Path]]:
    """Alias for :func:`fetch_transcripts`."""
    return fetch_transcripts(
        channels=channels,
        transcripts_dir=transcripts_dir,
        languages=languages,
        limit=limit,
        subs=subs,
    )


def buildOrUpdateVectorStore(  # noqa: N802
    channels: Iterable[str],
    transcripts_dir: Path,
    vector_dir: Path,
    *,
    rebuild: bool,
) -> TranscriptRAG:
    """Alias for :func:`build_or_update_vector_store`."""
    return build_or_update_vector_store(
        channels=channels,
        transcripts_dir=transcripts_dir,
        vector_dir=vector_dir,
        rebuild=rebuild,
    )


def queryVectorStore(rag: TranscriptRAG, question: str, k: int = 4) -> str:  # noqa: N802
    """Alias for :func:`query_vector_store`."""
    return query_vector_store(rag, question, k)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and execute the workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    transcripts_base = Path(args.transcripts_dir)
    fetch_transcripts(
        channels=args.channels,
        transcripts_dir=transcripts_base,
        languages=langs,
        limit=args.limit,
        subs=args.subs,
    )

    rag = build_or_update_vector_store(
        channels=args.channels,
        transcripts_dir=transcripts_base,
        vector_dir=Path(args.vector_dir),
        rebuild=args.rebuild,
    )

    if args.query:
        query_vector_store(rag, args.query, k=4)


if __name__ == "__main__":
    main()
