"""Project-wide custom exceptions.

Centralizes custom exception classes for reuse and organization.
"""

from __future__ import annotations


class _UseSubsFallbackError(Exception):
    """Internal control-flow exception carrying subtitles text.

    Raised to indicate that subtitle text should be used as a fallback path.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__("Use subtitles fallback")


class SkipVideoError(Exception):
    """Signal to callers that a video should be skipped without error."""


class ModelInitializationError(Exception):
    """Raised when an external model client fails to initialize."""


class VectorStoreAddError(Exception):
    """Raised when adding documents to the vector store fails."""


class VectorStoreDeleteError(Exception):
    """Raised when deleting documents from the vector store fails."""


class YtDlpError(Exception):
    """Raised when yt-dlp subprocess execution fails."""


__all__ = [
    "ModelInitializationError",
    "SkipVideoError",
    "VectorStoreAddError",
    "VectorStoreDeleteError",
    "YtDlpError",
    "_UseSubsFallbackError",
]
