"""Indexing (RAG) page to rebuild or update the vector store.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""

# ruff: noqa: N999
from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import TranscriptRAG
from src.utils.logging_setup import get_logger, setup_logging


def _set_favicon(url: str) -> None:
    """Inject favicon links so the tab icon renders correctly."""
    st.markdown(
        f"""
        <link rel=\"icon\" href=\"{url}\" sizes=\"16x16\" />
        <link rel=\"icon\" href=\"{url}\" sizes=\"32x32\" />
        <link rel=\"shortcut icon\" href=\"{url}\" />
        """,
        unsafe_allow_html=True,
    )


def _ensure_logging() -> None:
    """Initialize logging once per Streamlit session."""
    if not st.session_state.get("_logging_initialized", False):
        setup_logging()
        st.session_state["_logging_initialized"] = True


logger = get_logger("index")


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("rag", None)


def _get_rag() -> TranscriptRAG:
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(
            vector_dir=ss.vector_dir,
        )
        logger.info(
            "Initialized TranscriptRAG",
            extra={"vector_dir": str(ss.vector_dir)},
        )
    return ss.rag


def _list_channels(transcripts_dir: Path) -> list[str]:
    if not transcripts_dir.exists():
        return []
    channels = [p.name for p in sorted(transcripts_dir.iterdir()) if p.is_dir()]
    logger.debug("Listed channels", extra={"count": len(channels)})
    return channels




def _configure_page() -> None:
    """Load env, set page config and hide Streamlit default menu."""
    load_dotenv()
    st.set_page_config(page_title="Indexação (RAG)", layout="wide")

    hide_streamlit_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_menu, unsafe_allow_html=True)
    # Proper favicon to avoid oversized/clipped emoji in the browser tab
    twemoji_base = (
        "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/"
    )
    _set_favicon(twemoji_base + "1f9e0.png")  # brain
    _ensure_state()
    logger.info("Index page configured")


def _render_sidebar_nav() -> None:
    """Render left sidebar navigation."""
    with st.sidebar:
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()


def _render_paths() -> None:
    """Show transcript and vector store paths."""
    with st.expander("Caminhos", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text("Transcrições:")
            st.code(str(st.session_state.transcripts_dir))
        with col2:
            st.text("Índice vetorial:")
            st.code(str(st.session_state.vector_dir))


    # (Embeddings selection UI removed; Groq model selection moved to Chat page.)


def _update_channels(rag: TranscriptRAG, channel_names: list[str]) -> None:
    """Update channels incrementally without any artificial pause."""
    for ch in channel_names:
        ch_dir = st.session_state.transcripts_dir / ch
        rag.add_channel(ch_dir)


def _render_rebuild_section() -> None:
    """Reindex from scratch section."""
    st.subheader("Reindexar do zero")
    if st.button("Reindexar agora"):
        try:
            rag = _get_rag()
            with st.spinner("Reindexando transcrições..."):
                logger.info("Reindex start")
                progress = st.progress(0)
                status = st.empty()

                total_holder = {"total": 1}

                def on_progress(curr: int, total: int, rel: str) -> None:
                    total_holder["total"] = max(1, total)
                    progress.progress(min(1.0, curr / max(1, total)))
                    status.write(f"Processando: {rel} ({curr}/{total})")

                summary = rag.index_transcripts(
                    st.session_state.transcripts_dir,
                    on_progress=on_progress,
                )
                progress.progress(1.0)
                status.write("Concluído.")
        except Exception as e:
            logger.exception("Reindex failed")
            st.error(f"Falha ao reindexar: {e}")
        else:
            logger.info(
                "Reindex success",
                extra={
                    "files": int(summary.get("files", 0)),
                    "added": int(summary.get("added", 0)),
                    "skipped": int(summary.get("skipped", 0)),
                    "duration_s": float(summary.get("duration_s", 0.0)),
                },
            )
            st.success(
                "Índice refeito com sucesso. "
                f"Arquivos: {int(summary.get('files', 0))} | "
                f"Adicionados: {int(summary.get('added', 0))} | "
                f"Pulados: {int(summary.get('skipped', 0))} | "
                f"Tempo: {summary.get('duration_s', 0.0):.2f}s",
            )


def _render_incremental_update_section() -> None:
    """Incremental update section with per-channel or all-channels options."""
    st.divider()
    st.subheader("Atualizar canais (incremental)")

    channels = _list_channels(st.session_state.transcripts_dir)
    if not channels:
        st.info("Nenhum canal encontrado em data/transcripts.")
        return

    selected = st.multiselect("Selecione canais para atualizar", options=channels)
    col1, col2 = st.columns(2)
    with col1:
        update = st.button("Atualizar selecionados", use_container_width=True)
    with col2:
        update_all = st.button("Atualizar todos os canais", use_container_width=True)

    if update and selected:
        try:
            rag = _get_rag()
            with st.spinner("Atualizando canais selecionados..."):
                logger.info("Incremental update selected", extra={"channels": selected})
                total_added = 0
                total_skipped = 0
                for ch in selected:
                    a, s = rag.add_channel(st.session_state.transcripts_dir / ch)
                    total_added += a
                    total_skipped += s
        except Exception as e:
            logger.exception("Incremental update (selected) failed")
            st.error(f"Falha ao atualizar: {e}")
        else:
            logger.info(
                "Incremental update selected success",
                extra={"added": total_added, "skipped": total_skipped},
            )
            st.success(
                "Canais atualizados. "
                f"Adicionados: {total_added} | Pulados: {total_skipped}",
            )

    if update_all:
        try:
            rag = _get_rag()
            with st.spinner("Atualizando todos os canais..."):
                logger.info(
                    "Incremental update all start",
                    extra={"channels": channels},
                )
                total_added = 0
                total_skipped = 0
                for ch in channels:
                    a, s = rag.add_channel(st.session_state.transcripts_dir / ch)
                    total_added += a
                    total_skipped += s
        except Exception as e:
            logger.exception("Incremental update all failed")
            st.error(f"Falha ao atualizar todos: {e}")
        else:
            logger.info(
                "Incremental update all success",
                extra={"added": total_added, "skipped": total_skipped},
            )
            st.success(
                "Todos os canais atualizados. "
                f"Adicionados: {total_added} | Pulados: {total_skipped}",
            )


def _render_channel_management() -> None:
    """List, reindex or remove channels from the vector index."""
    st.divider()
    st.subheader("Gerenciar Canais")
    channels = _list_channels(st.session_state.transcripts_dir)
    if not channels:
        st.info("Nenhum canal encontrado em data/transcripts.")
        return
    for ch in channels:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(ch)
        with col2:
            if st.button("Reindexar", key=f"reindex_{ch}"):
                try:
                    logger.info("Reindex single channel start", extra={"channel": ch})
                    a, s = _get_rag().add_channel(st.session_state.transcripts_dir / ch)
                except Exception as e:
                    logger.exception(
                        "Reindex single channel failed",
                        extra={"channel": ch},
                    )
                    st.error(f"Falha ao reindexar {ch}: {e}")
                else:
                    logger.info(
                        "Reindex single channel success",
                        extra={"channel": ch, "added": a, "skipped": s},
                    )
                    st.success(f"{ch}: adicionados {a}, pulados {s}")
        with col3:
            if st.button("Remover do índice", key=f"remove_{ch}"):
                try:
                    logger.info(
                        "Remove channel from index start",
                        extra={"channel": ch},
                    )
                    removed = _get_rag().remove_channel_from_index(ch)
                except Exception as e:
                    logger.exception(
                        "Remove channel from index failed",
                        extra={"channel": ch},
                    )
                    st.error(f"Falha ao remover {ch}: {e}")
                else:
                    logger.warning(
                        "Channel removed from index",
                        extra={"channel": ch, "removed_ids": removed},
                    )
                    st.warning(
                        f"{ch}: removido do índice (ids encontradas: {removed}).",
                    )


def main() -> None:
    """Index and update the vector store."""
    _ensure_logging()
    _configure_page()
    _render_sidebar_nav()

    st.title("Indexação (RAG)")
    st.caption("Refaça o índice do zero ou atualize por canal.")

    _render_paths()
    _render_rebuild_section()
    _render_incremental_update_section()
    _render_channel_management()
    logger.info("Index page rendered")


if __name__ == "__main__":
    main()
