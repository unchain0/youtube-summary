"""RAG utilities for indexing and querying YouTube transcripts."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # typing-only import to satisfy Ruff TCH rules
    from collections.abc import Callable

# Optional imports with safe fallbacks for testing without heavy deps installed
try:
    from chromadb.config import (
        Settings as ChromaSettings,  # type: ignore[import-not-found]
    )
except ImportError:

    class ChromaSettings:  # type: ignore[no-redef]
        """Stub for chromadb.config.Settings used only for construction."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize ChromaSettings stub."""
            del args, kwargs


try:
    from langchain.chains import RetrievalQA  # type: ignore[import-not-found]
except ImportError:

    class RetrievalQA:  # type: ignore[no-redef]
        """Minimal stub for RetrievalQA supporting .from_chain_type() and .invoke()."""

        def __init__(self) -> None:
            """Construct a trivial QA chain stub."""

        @classmethod
        def from_chain_type(
            cls,
            llm: object,
            chain_type: str,
            retriever: object,
            *,
            chain_type_kwargs: dict[str, object] | None = None,
            return_source_documents: bool | None = None,
        ) -> RetrievalQA:
            """Return a new stub instance regardless of inputs."""
            # Mark parameters as used to satisfy Ruff ARG002 while preserving signature
            del llm, chain_type, retriever, chain_type_kwargs, return_source_documents
            return cls()

        def invoke(self, _inputs: dict[str, object]) -> dict[str, str]:
            """Return an empty result string like a degenerate chain."""
            return {"result": ""}


try:
    from langchain.prompts import ChatPromptTemplate  # type: ignore[import-not-found]
except ImportError:

    class ChatPromptTemplate:  # type: ignore[no-redef]
        """Stub for ChatPromptTemplate exposing from_messages()."""

        @classmethod
        def from_messages(
            cls,
            _messages: list[tuple[str, str]],
        ) -> ChatPromptTemplate:
            """Return a trivial template stub."""
            return cls()


try:
    from langchain.schema import Document  # type: ignore[import-not-found]
except ImportError:

    class Document:  # type: ignore[no-redef]
        """Lightweight document object with content and metadata."""

        def __init__(
            self,
            page_content: str,
            metadata: dict[str, object] | None = None,
        ) -> None:
            """Initialize a document stub."""
            self.page_content = page_content
            self.metadata = metadata or {}


try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,  # type: ignore[import-not-found]
    )
except ImportError:

    class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
        """Simple text splitter that produces overlapping fixed-size chunks."""

        def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 50) -> None:
            """Configure chunk size and overlap for the splitter."""
            self.chunk_size = int(max(1, chunk_size))
            self.chunk_overlap = int(max(0, min(chunk_overlap, self.chunk_size - 1)))

        def split_documents(self, docs: list[Document]) -> list[Document]:
            """Split each document into chunks preserving metadata."""
            out: list[Document] = []
            for doc in docs:
                text = doc.page_content
                start = 0
                step = max(1, self.chunk_size - self.chunk_overlap)
                while start < len(text):
                    end = min(len(text), start + self.chunk_size)
                    chunk = text[start:end]
                    out.append(
                        Document(
                            page_content=chunk,
                            metadata=dict(doc.metadata),
                        ),
                    )
                    if end == len(text):
                        break
                    start += step
            return out


try:
    from langchain_chroma import Chroma  # type: ignore[import-not-found]
except ImportError:

    class Chroma:  # type: ignore[no-redef]
        """In-memory stand-in for Chroma with minimal surface used by tests."""

        def __init__(
            self,
            embedding_function: object | None = None,
            persist_directory: str | None = None,
            collection_name: str | None = None,
            client_settings: object | None = None,
        ) -> None:
            """Initialize in-memory Chroma stub."""
            # Unused, but keep names compatible with real API/call sites
            del embedding_function, client_settings
            self._docs: list[Document] = []
            self._ids: set[str] = set()
            self.persist_directory = persist_directory
            self.collection_name = collection_name

        @classmethod
        def from_documents(
            cls,
            documents: list[Document],
            embedding: object,
            persist_directory: str,
            collection_name: str,
        ) -> Chroma:
            """Create a new collection initialized with provided documents."""
            inst = cls(
                embedding_function=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            inst._docs.extend(documents)
            return inst

        def add_documents(
            self,
            documents: list[Document],
            ids: list[str] | None = None,
        ) -> None:
            """Append documents and record optional IDs."""
            self._docs.extend(documents)
            if ids:
                self._ids.update(ids)

        def get(self, include: list[str] | None = None) -> dict[str, list[str]]:
            """Return a dict-like view including existing IDs."""
            del include
            return {"ids": sorted(self._ids)}

        def persist(self) -> None:
            """No-op persist for stub implementation."""

        def as_retriever(self, search_kwargs: dict[str, object]) -> object:
            """Return a minimal retriever exposing invoke()."""

            class _R:
                def __init__(self, docs: list[Document]) -> None:
                    self._docs = docs

                def invoke(self, _q: object) -> list[Document]:
                    k = int(search_kwargs.get("k", 4))
                    return self._docs[:k]

            return _R(self._docs)

        def delete(self, *args: object, **kwargs: object) -> None:
            """No-op delete in stub implementation."""


try:
    from langchain_groq import ChatGroq  # type: ignore[import-not-found]
except ImportError:

    class ChatGroq:  # type: ignore[no-redef]
        """Stub for ChatGroq model class."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Construct a no-op model stub."""


try:
    from langchain_together.embeddings import (  # type: ignore[import-not-found]
        TogetherEmbeddings,
    )
except ImportError:

    class TogetherEmbeddings:  # type: ignore[no-redef]
        """Stub for TogetherEmbeddings used in tests unless monkeypatched."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Construct a no-op embeddings stub."""


from src.exceptions import ModelInitializationError, VectorStoreAddError
from src.utils.logging_setup import logger


class TranscriptRAG:
    """Build a vector store from transcript files and provide QA.

    - Embeddings: Together AI (default: togethercomputer/m2-bert-80M-32k-retrieval;
      override via TOGETHER_EMBEDDINGS_MODEL)
    - Vector store: Chroma (local, persisted under data/vector_store)
    - LLM: Groq chat model (configurable via env GROQ_MODEL; lazy-initialized)
    """

    def __init__(
        self,
        vector_dir: str | Path = "data/vector_store",
        chunk_size: int = 12000,
        chunk_overlap: int = 1200,
        embed_model_name: str | None = None,
        groq_model: str | None = None,
    ) -> None:
        """Initialize RAG components.

        Args:
            vector_dir: Directory where the Chroma DB will persist.
            chunk_size: Max characters per chunk for splitting.
            chunk_overlap: Overlap between adjacent chunks.
            embed_model_name: Optional Together embeddings model name; overrides env.
            groq_model: Optional Groq model name; overrides env.

        """
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embed_model_name = embed_model_name or os.getenv(
            "TOGETHER_EMBEDDINGS_MODEL",
            "togethercomputer/m2-bert-80M-32k-retrieval",
        )
        # Instantiate Together embeddings
        self.embeddings = TogetherEmbeddings(model=self._embed_model_name)
        safe_model = self._embed_model_name.replace("-", "_").replace("/", "_")
        self.collection_name = f"transcripts_together_{safe_model}"
        self.db: Chroma | None = None
        self.groq_model_name: str = groq_model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )
        self.model: ChatGroq | None = None
        # Root directory for transcripts to compute stable relative paths
        self.transcripts_root: Path | None = None
        # Disable Chroma anonymized telemetry to avoid noisy logs and external calls
        self._chroma_settings = ChromaSettings(anonymized_telemetry=False)

    def _system_prompt(self) -> str:
        """Return the system prompt for Groq RAG, optionally overridden by env.

        Env var: RAG_SYSTEM_PROMPT
        """
        env_prompt = os.getenv("RAG_SYSTEM_PROMPT")
        if env_prompt:
            return env_prompt
        return (
            "Você é o autor dos conteúdos presentes nas transcrições fornecidas. "
            "Escreva em primeira pessoa, em português do Brasil, com tom claro, "
            "confiante e direto. Responda somente com base no CONTEXTO "
            "fornecido; não invente informações. Não mencione “fontes”, "
            "“contexto”, nem use expressões como “de acordo com as fontes” ou "
            "“segundo o autor”. Não mencione que é um sistema de IA.\n\n"
            "Se a informação solicitada não estiver no CONTEXTO, diga que esse "
            "ponto não foi abordado nos materiais e, se fizer sentido, sugira "
            "revisar os vídeos relevantes ou refinar a pergunta.\n\n"
            "Estruture a resposta como um texto autoral coeso; quando útil, "
            "utilize listas curtas e objetivas. Evite redundâncias, desculpas e "
            "qualificadores desnecessários. Mantenha foco, precisão e "
            "consistência com o que está nos trechos."
        )

    def _build_chat_prompt(self) -> ChatPromptTemplate:
        """Compose a chat prompt with system instructions and placeholders.

        Works with RetrievalQA chain_type="stuff" expecting {context} and {question}.
        """
        system = self._system_prompt()
        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Pergunta: {question}\n\n"
                    "Contexto (trechos relevantes):\n"
                    "{context}\n\n"
                    "Siga as instruções do sistema.",
                ),
            ],
        )

    def _ensure_model(self) -> ChatGroq:
        if self.model is None:
            try:
                self.model = ChatGroq(model=self.groq_model_name, temperature=0)
            except Exception as err:
                msg = (
                    "Failed to initialize Groq model. "
                    f"Model='{self.groq_model_name}'. Ensure GROQ_API_KEY is set and "
                    "the model exists and is accessible. You can set GROQ_MODEL to a "
                    "supported model (e.g., 'mixtral-8x7b-32768', "
                    "'llama-3.3-70b-versatile')."
                )
                raise ModelInitializationError(msg) from err
        return self.model

    def _generate_chunk_ids(self, chunks: list[Document]) -> list[str]:
        """Generate deterministic IDs for each chunk to avoid duplicates.

        ID format: "{source}::{local_index}::{sha1_12(page_content)}"
        where local_index increments per source to remain stable across runs.
        """
        counters: dict[str, int] = {}
        ids: list[str] = []
        for chunk in chunks:
            source = str(chunk.metadata.get("source", "unknown"))
            idx = counters.get(source, 0)
            counters[source] = idx + 1
            text_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[
                :12
            ]
            ids.append(f"{source}::{idx}::{text_hash}")
        return ids

    def _add_chunks_with_ids(self, chunks: list[Document]) -> tuple[int, int]:
        """Add chunks to the vector store using deterministic IDs, skipping existing.

        Returns a tuple (added_count, skipped_count).
        """
        db = self._ensure_db()
        ids = self._generate_chunk_ids(chunks)

        existing = self._existing_ids(db)

        docs_to_add: list[Document] = []
        ids_to_add: list[str] = []
        for doc, _id in zip(chunks, ids, strict=True):
            if _id not in existing:
                docs_to_add.append(doc)
                ids_to_add.append(_id)

        added = 0
        if docs_to_add:
            try:
                db.add_documents(docs_to_add, ids=ids_to_add)  # type: ignore[arg-type]
                added = len(docs_to_add)
            except Exception as e:
                logger.warning(
                    "Failed to add documents to Chroma collection. This often "
                    "happens when the existing collection was created with a "
                    "different embedding dimension or due to duplicate IDs. Try "
                    "rebuilding the index. Cause: {}: {}",
                    type(e).__name__,
                    e,
                )
                msg = (
                    "Falha ao adicionar documentos ao índice vetorial. "
                    "Possível incompatibilidade de dimensão de embeddings "
                    "ou IDs duplicados. "
                    "Recrie o índice do zero."
                )
                raise VectorStoreAddError(msg) from e

        skipped = len(chunks) - added
        if hasattr(db, "persist"):
            try:
                db.persist()  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Chroma persist() failed: {}: {}",
                    type(e).__name__,
                    e,
                )
        return added, skipped

    def _existing_ids(self, db: Chroma) -> set[str]:
        """Return existing IDs from the Chroma collection, best-effort."""
        # Prefer public API if available
        if hasattr(db, "get"):
            try:
                try:
                    result = db.get(include=[])  # type: ignore[attr-defined]
                except TypeError:
                    result = db.get()  # type: ignore[attr-defined]
                ids = result.get("ids") if isinstance(result, dict) else None
                return set(ids or [])
            except Exception:  # noqa: BLE001
                return set()

        # Fallback to underlying chromadb collection
        collection = getattr(db, "_collection", None)
        if collection is None or not hasattr(collection, "get"):
            return set()
        try:
            try:
                result = collection.get(include=[])  # type: ignore[attr-defined]
            except TypeError:
                result = collection.get()  # type: ignore[attr-defined]
            ids = result.get("ids") if isinstance(result, dict) else None
            return set(ids or [])
        except Exception:  # noqa: BLE001
            return set()

    def _docs_from_transcripts(self, transcripts_dir: str | Path) -> list[Document]:
        # Use a stable root if available to make metadata["source"] consistent
        base = Path(self.transcripts_root or transcripts_dir)
        docs: list[Document] = []
        for fpath in base.rglob("*.txt"):
            text = fpath.read_text(encoding="utf-8").strip()
            if not text:
                continue
            rel = fpath.relative_to(base).as_posix()
            docs.append(Document(page_content=text, metadata={"source": rel}))
        return docs

    def add_transcript_file(self, transcript_path: str | Path) -> tuple[int, int]:
        """Incrementally index a single transcript .txt file.

        Returns a tuple (added, skipped) referring to chunk vectors added/skipped.
        Requires that ``self.transcripts_root`` is set to the root directory under
        which ``transcript_path`` resides (e.g., ``data/transcripts``) to keep
        ``metadata['source']`` stable across runs.
        """
        fpath = Path(transcript_path)
        if not fpath.exists() or not fpath.is_file():
            return (0, 0)
        try:
            text = fpath.read_text(encoding="utf-8").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to read transcript {}: {}: {}",
                fpath,
                type(e).__name__,
                e,
            )
            return (0, 0)
        if not text:
            return (0, 0)
        # Determine base for stable relative source
        base = Path(self.transcripts_root or fpath.parent.parent)
        try:
            rel = fpath.relative_to(base).as_posix()
        except ValueError:
            # Fallback: just use filename
            rel = fpath.name
        doc = Document(page_content=text, metadata={"source": rel})
        chunks = self.text_splitter.split_documents([doc])
        return self._add_chunks_with_ids(chunks)

    def index_transcripts(
        self,
        transcripts_dir: str | Path,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int | float]:
        """Build or rebuild the vector store from transcript .txt files.

        If ``on_progress`` is provided, it will be called as
        ``on_progress(current_file_index, total_files, relative_path)`` after each
        file is processed.

        Returns a summary dict with keys: ``files``, ``added``, ``skipped``,
        ``duration_s``.
        """
        start = time.perf_counter()
        self.transcripts_root = Path(transcripts_dir)
        base = self.transcripts_root
        files = list(base.rglob("*.txt"))
        total = len(files)

        logger.info(
            "Indexing started. files={}, model={}, collection={}",
            total,
            self._embed_model_name,
            self.collection_name,
        )

        if total == 0:
            # Initialize empty store and persist
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
                client_settings=self._chroma_settings,
            )
            if hasattr(self.db, "persist"):
                try:
                    self.db.persist()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Chroma persist() failed: {}: {}",
                        type(e).__name__,
                        e,
                    )
            elapsed = time.perf_counter() - start
            logger.info(
                "Indexing finished. files=0 added=0 skipped=0 time={:.2f}s",
                elapsed,
            )
            return {"files": 0, "added": 0, "skipped": 0, "duration_s": elapsed}

        # Ensure DB exists once
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_dir),
            collection_name=self.collection_name,
            client_settings=self._chroma_settings,
        )

        processed, added_total, skipped_total = self._index_files(
            files,
            base,
            total,
            on_progress,
        )

        # Persist at end
        if hasattr(self.db, "persist"):
            try:
                self.db.persist()  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                logger.warning("Chroma persist() failed: {}: {}", type(e).__name__, e)

        elapsed = time.perf_counter() - start
        logger.info(
            "Indexing finished. files={} added={} skipped={} time={:.2f}s",
            processed,
            added_total,
            skipped_total,
            elapsed,
        )
        return {
            "files": processed,
            "added": added_total,
            "skipped": skipped_total,
            "duration_s": elapsed,
        }

    def _index_files(
        self,
        files: list[Path],
        base: Path,
        total: int,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> tuple[int, int, int]:
        """Index individual transcript files and return counters.

        Returns (processed, added_total, skipped_total).
        """
        added_total = 0
        skipped_total = 0
        processed = 0
        for fpath in files:
            try:
                text = fpath.read_text(encoding="utf-8").strip()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to read transcript {}: {}: {}",
                    fpath,
                    type(e).__name__,
                    e,
                )
                processed += 1
                if on_progress:
                    rel = fpath.relative_to(base).as_posix()
                    on_progress(processed, total, rel)
                continue

            if not text:
                processed += 1
                if on_progress:
                    rel = fpath.relative_to(base).as_posix()
                    on_progress(processed, total, rel)
                continue

            rel = fpath.relative_to(base).as_posix()
            doc = Document(page_content=text, metadata={"source": rel})
            chunks = self.text_splitter.split_documents([doc])
            a, s = self._add_chunks_with_ids(chunks)
            added_total += a
            skipped_total += s
            processed += 1
            if on_progress:
                on_progress(processed, total, rel)
        return processed, added_total, skipped_total

    def add_channel(self, channel_transcripts_dir: str | Path) -> tuple[int, int]:
        """Incrementally add a channel's transcripts into the current vector store.

        Returns a tuple (added, skipped).
        """
        t0 = time.perf_counter()
        if self.db is None:
            # Lazy init empty store if not present
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
                client_settings=self._chroma_settings,
            )
        # Ensure stable transcripts root (parent of the channel dir) if not set yet
        if self.transcripts_root is None:
            self.transcripts_root = Path(channel_transcripts_dir).parent
        docs = self._docs_from_transcripts(Path(channel_transcripts_dir))
        if not docs:
            return (0, 0)
        chunks: list[Document] = []
        for doc in docs:
            chunks.extend(self.text_splitter.split_documents([doc]))
        added, skipped = self._add_chunks_with_ids(chunks)
        logger.info(
            "Channel indexing completed. added={} skipped={} time={:.2f}s",
            added,
            skipped,
            (time.perf_counter() - t0),
        )
        return added, skipped

    def remove_channel_from_index(self, channel_name: str) -> int:
        """Remove all vectors for a given channel from the index.

        Best effort: tries to find IDs matching metadata.source starting with
        "channel_name/" and delete them. Returns count of IDs attempted to delete.
        """
        db = self._ensure_db()
        # Try to fetch matching IDs using underlying collection when possible
        ids_to_delete: list[str] = []
        where = {"source": {"$contains": f"{channel_name}/"}}
        collection = getattr(db, "_collection", None)
        if collection is not None and hasattr(collection, "get"):
            try:
                result = collection.get(where=where)  # type: ignore[attr-defined]
                got = result.get("ids") if isinstance(result, dict) else None
                if got:
                    ids_to_delete = list(got)
            except Exception:  # noqa: BLE001
                ids_to_delete = []
        # If not collected, we still try a best-effort delete by where below
        try:
            # Prefer deleting by where if supported (will be ignored otherwise)
            db.delete(where=where)  # type: ignore[arg-type]
        except Exception as err:  # noqa: BLE001
            # Fallback: delete by explicit ids if we managed to fetch any
            if ids_to_delete:
                try:
                    db.delete(ids=ids_to_delete)  # type: ignore[arg-type]
                except Exception as err2:  # noqa: BLE001
                    logger.warning(
                        "Delete by ids failed: {}: {}",
                        type(err2).__name__,
                        err2,
                    )
            else:
                logger.warning(
                    "Delete by where failed: {}: {}",
                    type(err).__name__,
                    err,
                )
        return len(ids_to_delete)

    def _ensure_db(self) -> Chroma:
        if self.db is None:
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
                client_settings=self._chroma_settings,
            )
        return self.db

    def query(self, question: str, k: int = 4, score_threshold: float = 0.7) -> str:
        """Perform retrieval-augmented QA using RetrievalQA chain.

        Args:
            question: The question to answer.
            k: Maximum number of documents to retrieve.
            score_threshold: Minimum similarity score (0-1). Only documents with
                score >= threshold are returned. Default 0.7.

        """
        db = self._ensure_db()
        llm = self._ensure_model()
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        qa = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self._build_chat_prompt()},
        )
        result = qa.invoke({"query": question})
        if isinstance(result, dict) and "result" in result:
            return str(result["result"])
        return str(result)

    def query_with_sources(
        self,
        question: str,
        k: int = 4,
        score_threshold: float = 0.7,
    ) -> tuple[str, list[Document]]:
        """RAG QA returning both answer text and source documents.

        Args:
            question: The question to answer.
            k: Maximum number of documents to retrieve.
            score_threshold: Minimum similarity score (0-1). Only documents with
                score >= threshold are returned. Default 0.7.

        Returns:
            A tuple of (answer, source_documents).

        """
        db = self._ensure_db()
        llm = self._ensure_model()
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        qa = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self._build_chat_prompt()},
        )
        result = qa.invoke({"query": question})
        answer: str
        sources: list[Document]
        if isinstance(result, dict):
            answer = str(result.get("result", ""))
            sources = result.get("source_documents") or []  # type: ignore[assignment]
        else:
            answer = str(result)
            sources = []
        logger.info(
            "Query done. k={} sources={} chars={}",
            k,
            len(sources),
            len(answer),
        )
        return answer, sources
