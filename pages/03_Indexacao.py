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


def main() -> None:
    """Index and update the vector store."""
    load_dotenv()
    st.set_page_config(page_title="Indexação (RAG)", page_icon="🧠", layout="wide")

    # Hide default Streamlit menu
    hide_streamlit_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_menu, unsafe_allow_html=True)

    _ensure_state()
    with st.sidebar:
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()

    st.title("Indexação (RAG)")
    st.caption("Refaça o índice do zero ou atualize por canal.")

    with st.expander("Caminhos", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text("Transcrições:")
            st.code(str(st.session_state.transcripts_dir))
        with col2:
            st.text("Índice vetorial:")
            st.code(str(st.session_state.vector_dir))

    st.subheader("Reindexar do zero")
    if st.button("Reindexar agora"):
        try:
            with st.spinner("Reindexando transcrições..."):
                _get_rag().index_transcripts(st.session_state.transcripts_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao reindexar: {e}")
        else:
            st.success("Índice refeito com sucesso.")

    st.divider()
    st.subheader("Atualizar canais (incremental)")

    channels = _list_channels(st.session_state.transcripts_dir)
    if not channels:
        st.info("Nenhum canal encontrado em data/transcripts.")
        return

    selected = st.multiselect("Selecione canais para atualizar", options=channels)
    update = st.button("Atualizar selecionados")
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


if __name__ == "__main__":
    main()
