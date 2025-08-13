"""RAG utilities for indexing and querying YouTube transcripts."""

import hashlib
import os
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_together import TogetherEmbeddings

from .utils.logging_setup import logger


class TranscriptRAG:
    """Build a vector store from transcript files and provide QA.

    - Embeddings: Together AI (Multilingual E5 Large Instruct)
    - Vector store: Chroma (local, persisted under data/vector_store)
    - LLM: Groq chat model (configurable via env GROQ_MODEL; lazy-initialized)
    """

    def __init__(
        self,
        vector_dir: str | Path = "data/vector_store",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        groq_model: str | None = None,
    ) -> None:
        """Initialize RAG components.

        Args:
            vector_dir: Directory where the Chroma DB will persist.
            chunk_size: Max characters per chunk for splitting.
            chunk_overlap: Overlap between adjacent chunks.
            groq_model: Optional Groq model name; defaults to env or a sensible default.

        """
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        provider = "together"
        model_name = "intfloat/multilingual-e5-large-instruct"
        self.embeddings = TogetherEmbeddings(model=model_name)
        safe_model = model_name.replace("-", "_").replace("/", "_")
        self.collection_name = f"transcripts_{provider}_{safe_model}"
        self.db: Chroma | None = None
        self.groq_model_name: str = groq_model or os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )
        self.model: ChatGroq | None = None
        # Root directory for transcripts to compute stable relative paths
        self.transcripts_root: Path | None = None

    def _ensure_model(self) -> ChatGroq:
        if self.model is None:
            try:
                self.model = ChatGroq(model=self.groq_model_name)
            except Exception as err:
                msg = (
                    "Failed to initialize Groq model. "
                    f"Model='{self.groq_model_name}'. Ensure GROQ_API_KEY is set and "
                    "the model exists and is accessible. You can set GROQ_MODEL to a "
                    "supported model (e.g., 'mixtral-8x7b-32768', "
                    "'llama-3.3-70b-versatile')."
                )
                raise RuntimeError(msg) from err
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
                    "running with --rebuild to recreate the index. Cause: {}: {}",
                    type(e).__name__,
                    e,
                )
                raise

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

    def index_transcripts(self, transcripts_dir: str | Path) -> None:
        """Build or rebuild the vector store from transcript .txt files."""
        # Load existing transcript docs
        self.transcripts_root = Path(transcripts_dir)
        docs = self._docs_from_transcripts(transcripts_dir)
        if not docs:
            # No transcripts: initialize empty store and persist
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
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
            return
        # Non-empty transcripts: create docs using our stable relative source
        chunks = self.text_splitter.split_documents(docs)
        # Ensure DB exists, then add chunks with deterministic IDs to avoid duplicates
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_dir),
            collection_name=self.collection_name,
        )
        added, skipped = self._add_chunks_with_ids(chunks)
        logger.info(
            "Indexing completed. Added {} chunks, skipped {} already present.",
            added,
            skipped,
        )

    def add_channel(self, channel_transcripts_dir: str | Path) -> None:
        """Incrementally add a channel's transcripts into the current vector store."""
        if self.db is None:
            # Lazy init empty store if not present
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
            )
        # Ensure stable transcripts root (parent of the channel dir) if not set yet
        if self.transcripts_root is None:
            self.transcripts_root = Path(channel_transcripts_dir).parent
        docs = self._docs_from_transcripts(Path(channel_transcripts_dir))
        if not docs:
            return
        chunks: list[Document] = []
        for doc in docs:
            chunks.extend(self.text_splitter.split_documents([doc]))
        added, skipped = self._add_chunks_with_ids(chunks)
        logger.info(
            "Channel indexing completed. Added {} chunks, skipped {} already present.",
            added,
            skipped,
        )

    def _ensure_db(self) -> Chroma:
        if self.db is None:
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name=self.collection_name,
            )
        return self.db

    def query(self, question: str, k: int = 4) -> str:
        """Perform retrieval-augmented QA using RetrievalQA chain."""
        db = self._ensure_db()
        llm = self._ensure_model()
        retriever = db.as_retriever(search_kwargs={"k": k})
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
        return qa.run(question)
