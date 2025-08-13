"""Indexing (RAG) page to rebuild or update the vector store.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""
# ruff: noqa: N999
from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import TranscriptRAG


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("rag", None)


def _get_rag() -> TranscriptRAG:
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(vector_dir=ss.vector_dir)
    return ss.rag


def _list_channels(transcripts_dir: Path) -> list[str]:
    if not transcripts_dir.exists():
        return []
    return [p.name for p in sorted(transcripts_dir.iterdir()) if p.is_dir()]


def _configure_page() -> None:
    """Load env, set page config and hide Streamlit default menu."""
    load_dotenv()
    st.set_page_config(page_title="Indexação (RAG)", page_icon="🧠", layout="wide")

    hide_streamlit_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_menu, unsafe_allow_html=True)
    _ensure_state()


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


def _render_rebuild_section() -> None:
    """Reindex from scratch section."""
    st.subheader("Reindexar do zero")
    if st.button("Reindexar agora"):
        try:
            with st.spinner("Reindexando transcrições..."):
                _get_rag().index_transcripts(st.session_state.transcripts_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao reindexar: {e}")
        else:
            st.success("Índice refeito com sucesso.")


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
                for ch in selected:
                    ch_dir = st.session_state.transcripts_dir / ch
                    rag.add_channel(ch_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao atualizar: {e}")
        else:
            st.success("Canais atualizados no índice.")

    if update_all:
        try:
            rag = _get_rag()
            with st.spinner("Atualizando todos os canais..."):
                for ch in channels:
                    ch_dir = st.session_state.transcripts_dir / ch
                    rag.add_channel(ch_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao atualizar todos: {e}")
        else:
            st.success("Todos os canais atualizados no índice.")


def main() -> None:
    """Index and update the vector store."""
    _configure_page()
    _render_sidebar_nav()

    st.title("Indexação (RAG)")
    st.caption("Refaça o índice do zero ou atualize por canal.")

    _render_paths()
    _render_rebuild_section()
    _render_incremental_update_section()


if __name__ == "__main__":
    main()
