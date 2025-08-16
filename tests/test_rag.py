"""Tests for RAG indexing and retrieval helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

import src.rag as rag_mod


class FakeEmbeddings:
    """Fake embeddings for tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """No-op initializer for fake embeddings."""
        del args, kwargs


class FakeChroma:
    """A minimal in-memory stand-in for Chroma used in tests."""

    _REGISTRY: ClassVar[dict[str, set[str]]] = {}

    def __init__(
        self,
        embedding_function: object | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the fake vector store container."""
        del embedding_function, kwargs
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.docs: list[object] = []
        self.persisted = False
        # Maintain IDs per (persist_directory, collection_name) so subsequent
        # instances share the same underlying collection state.
        key = f"{persist_directory}|{collection_name}"
        reg: dict[str, set[str]] = FakeChroma._REGISTRY
        if key not in reg:
            reg[key] = set()
        self._key = key
        self._ids = reg[key]

    @classmethod
    def from_documents(
        cls,
        documents: list[object],
        embedding: object,
        persist_directory: str,
        collection_name: str,
    ) -> FakeChroma:
        """Construct a FakeChroma from provided documents."""
        inst = cls(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        inst.docs.extend(documents)
        return inst

    def add_documents(
        self,
        documents: list[object],
        ids: list[str] | None = None,
    ) -> None:
        """Append documents to the in-memory store, recording optional IDs."""
        self.docs.extend(documents)
        if ids is not None:
            self._ids.update(ids)

    def persist(self) -> None:
        """Mark the store as persisted."""
        self.persisted = True

    def get(self, include: list[str] | None = None) -> dict[str, list[str]]:
        """Return a dict containing existing IDs (like chromadb)."""
        del include
        return {"ids": sorted(self._ids)}

    def as_retriever(self, search_kwargs: dict[str, int]) -> object:
        """Return a simple retriever exposing an `invoke()` method."""

        class R:
            def __init__(self, docs: list[object]) -> None:
                self._docs = docs

            def invoke(self, _query: object) -> list[object]:
                k = int(search_kwargs.get("k", 4))
                return self._docs[:k]

        return R(self.docs)


def test_index_transcripts_empty_initializes_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Indexing on empty directory initializes the vector DB and persists it."""
    monkeypatch.setattr(rag_mod, "TogetherEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag_mod, "Chroma", FakeChroma)

    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True)

    r = rag_mod.TranscriptRAG(vector_dir=tmp_path / "vec")
    r.index_transcripts(transcripts_dir)
    assert isinstance(r.db, FakeChroma)
    assert r.db.persisted is True


def test_add_channel_no_docs_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Adding a channel with no docs still initializes the DB."""
    monkeypatch.setattr(rag_mod, "TogetherEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag_mod, "Chroma", FakeChroma)

    r = rag_mod.TranscriptRAG(vector_dir=tmp_path / "vec")

    def fake_docs(_self: object, _dir: Path) -> list[object]:
        return []

    monkeypatch.setattr(rag_mod.TranscriptRAG, "_docs_from_transcripts", fake_docs)
    r.add_channel(tmp_path / "transcripts")
    assert isinstance(r.db, FakeChroma)


def test_deduplicate_chunk_ids_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure repeated indexing doesn't create duplicate chunk IDs."""
    monkeypatch.setattr(rag_mod, "TogetherEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag_mod, "Chroma", FakeChroma)

    base = tmp_path / "transcripts"
    chan = base / "channelX"
    chan.mkdir(parents=True, exist_ok=True)
    f = chan / "vid1.txt"
    # Create content long enough to produce multiple chunks with small chunk_size
    content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
    f.write_text(content, encoding="utf-8")

    r = rag_mod.TranscriptRAG(
        vector_dir=tmp_path / "vec",
        chunk_size=64,
        chunk_overlap=0,
    )
    # First full indexing
    r.index_transcripts(base)
    assert isinstance(r.db, FakeChroma)
    data1 = r.db.get()
    first_ids = set(data1.get("ids", []))
    assert len(first_ids) > 0

    # Incremental add of the same channel should skip all existing IDs
    r.add_channel(chan)
    data2 = r.db.get()
    second_ids = set(data2.get("ids", []))
    assert second_ids == first_ids
