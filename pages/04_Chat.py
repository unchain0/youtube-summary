# ruff: noqa: N999
"""Streamlit chat page using st.chat_message and st.chat_input.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import TranscriptRAG

# Optional Groq SDK import for dynamic model listing
try:
    from groq import Groq as _Groq
except Exception:  # noqa: BLE001
    _Groq = None  # type: ignore[assignment]


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("rag", None)
    ss.setdefault("chat", [])  # list of {role, content}
    ss.setdefault("top_k", 4)
    ss.setdefault(
        "groq_model_name",
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


def _get_rag() -> TranscriptRAG:
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(
            vector_dir=ss.vector_dir,
            groq_model=ss.groq_model_name,
        )
    return ss.rag


@st.cache_data(ttl=900, show_spinner=False)
def _list_groq_models(api_key: str | None) -> list[str]:
    """Fetch Groq models via SDK. Returns [] on error or missing key."""
    try:
        if not api_key or _Groq is None:
            return []
        client = _Groq(api_key=api_key)
        resp = client.models.list()
        data = getattr(resp, "data", None)
        items = data if data is not None else getattr(resp, "models", [])
        ids = [str(getattr(m, "id", "")) for m in items]
        return sorted({i for i in ids if i})
    except Exception:  # noqa: BLE001
        return []


def _render_groq_listing_hint(dyn_groq: list[str], groq_key: str | None) -> None:
    """Render helper messages when dynamic Groq model list is empty."""
    if not dyn_groq:
        if not groq_key:
            st.info("Defina GROQ_API_KEY para listar modelos Groq.")
        else:
            st.warning(
                "Não foi possível recuperar modelos do Groq agora. Verifique sua "
                "chave/permissões ou tente novamente com 'Atualizar modelos Groq'.",
            )


def _render_groq_model_selector() -> None:
    """Render Groq LLM model selector in the sidebar and update session state."""
    st.subheader("Modelo LLM (Groq)")
    groq_presets = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "Custom…"]
    groq_key = os.getenv("GROQ_API_KEY")
    dyn_groq = _list_groq_models(groq_key)
    groq_options = [
        *sorted(set(groq_presets[:-1]) | set(dyn_groq)),
        "Custom…",
    ]
    if st.button("🔄 Atualizar modelos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    _render_groq_listing_hint(dyn_groq, groq_key)

    current_groq = st.session_state.groq_model_name
    if current_groq in groq_options:
        idx_g = groq_options.index(current_groq)
    else:
        idx_g = len(groq_options) - 1  # Custom…
    sel_g = st.selectbox(
        "Modelo",
        groq_options,
        index=idx_g,
        label_visibility="collapsed",
    )
    if sel_g == "Custom…":
        current_groq = st.text_input(
            "Nome do modelo Groq",
            value=st.session_state.groq_model_name,
        ).strip()
    else:
        current_groq = sel_g

    if current_groq != st.session_state.groq_model_name:
        st.session_state.groq_model_name = current_groq
        st.session_state.rag = None  # force re-instantiation with new model
        st.info(
            "Modelo Groq atualizado. Novas consultas usarão a nova configuração.",
        )


def _render_sidebar() -> None:
    """Render left sidebar navigation and model selector."""
    with st.sidebar:
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()
        _render_groq_model_selector()


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
    _render_sidebar()

    st.title("Conversar com os vídeos")
    st.caption("Pergunte algo; recuperamos passagens do índice e respondemos.")

    st.session_state.top_k = st.slider(
        "Número de passagens recuperadas (k)",
        1,
        10,
        st.session_state.top_k,
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
        with st.chat_message("assistant"):
            try:
                with st.spinner("Consultando o índice e gerando resposta..."):
                    answer, sources = _get_rag().query_with_sources(
                        prompt.strip(),
                        k=st.session_state.top_k,
                    )
            except Exception as e:  # noqa: BLE001
                err = (
                    "Falha ao consultar. Verifique GROQ_API_KEY e o modelo. "
                    f"Detalhes: {e}"
                )
                st.error(err)
                st.session_state.chat.append({"role": "user", "content": prompt})
                st.session_state.chat.append({"role": "assistant", "content": err})
            else:
                st.markdown(answer)
                with st.expander("Fontes consultadas", expanded=False):
                    if sources:
                        for i, doc in enumerate(sources, 1):
                            meta = getattr(doc, "metadata", {})
                            src = str(meta.get("source", "desconhecida"))
                            snippet = str(getattr(doc, "page_content", ""))[:300]
                            st.markdown(f"{i}. `{src}`")
                            st.code(snippet)
                    else:
                        st.caption("Nenhuma fonte retornada pelo pipeline.")
                st.caption("Concluído.")
                st.session_state.chat.append({"role": "user", "content": prompt})
                st.session_state.chat.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
