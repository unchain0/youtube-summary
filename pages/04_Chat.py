# ruff: noqa: N999
"""Streamlit chat page using st.chat_message and st.chat_input.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import TranscriptRAG


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("rag", None)
    ss.setdefault("chat", [])  # list of {role, content}
    ss.setdefault("top_k", 4)


def _get_rag() -> TranscriptRAG:
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(vector_dir=ss.vector_dir)
    return ss.rag


def main() -> None:
    """Render a simple chat interface backed by the RAG index."""
    load_dotenv()
    st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

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

    st.title("Conversar com os vídeos")
    st.caption("Pergunte algo; recuperamos passagens do índice e respondemos.")

    st.session_state.top_k = st.slider(
        "Número de passagens recuperadas (k)", 1, 10, st.session_state.top_k,
    )

    # Clear conversation
    if st.button("Limpar conversa"):
        st.session_state.chat = []
        st.rerun()

    # Render chat history
    for msg in st.session_state.chat[-50:]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    # Chat input
    prompt = st.chat_input("Pergunte algo sobre os vídeos indexados...")
    if prompt:
        # Echo user message
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            answer = _get_rag().query(prompt.strip(), k=st.session_state.top_k)
        except Exception as e:  # noqa: BLE001
            err = (
                f"Falha ao consultar. Verifique GROQ_API_KEY e o modelo. Detalhes: {e}"
            )
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.chat.append({"role": "user", "content": prompt})
            st.session_state.chat.append({"role": "assistant", "content": err})
        else:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat.append({"role": "user", "content": prompt})
            st.session_state.chat.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
