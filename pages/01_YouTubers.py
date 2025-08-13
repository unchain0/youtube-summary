"""YouTubers page listing downloaded transcripts.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""
# ruff: noqa: N999
from __future__ import annotations

from pathlib import Path

import streamlit as st


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))


def _list_channels(transcripts_dir: Path) -> list[tuple[str, int, list[str]]]:
    out: list[tuple[str, int, list[str]]] = []
    if not transcripts_dir.exists():
        return out
    for p in sorted(transcripts_dir.iterdir()):
        if p.is_dir():
            files = sorted([f.name for f in p.glob("*.txt")])
            out.append((p.name, len(files), files))
    return out


def main() -> None:
    """Render the page that lists downloaded YouTubers and their transcripts."""
    st.set_page_config(page_title="YouTubers", page_icon="📺", layout="wide")

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
        st.page_link("pages/01_YouTubers.py", label="YouTubers", icon="📺")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()
    st.title("YouTubers baixados")

    rows = _list_channels(st.session_state.transcripts_dir)
    if not rows:
        st.info("Nenhum transcript baixado ainda.")
        st.write(
            "Vá até a página 'Baixar Notas' para iniciar o download de transcrições.",
        )
        return

    for ch_key, count, files in rows:
        with st.expander(f"{ch_key} — {count} arquivos"):
            if not files:
                st.write("(vazio)")
            else:
                for name in files:
                    st.write(f"- {name}")


if __name__ == "__main__":
    main()
