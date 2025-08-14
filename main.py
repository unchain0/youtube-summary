"""Streamlit app replacing the CLI for YouTube transcript RAG.

Features (parity with CLI):
- Fetch transcripts for one or more channels (with languages, limit, subs fallback)
- Build/Update the vector store (rebuild or incremental)
- Query the vector store (chat)
- View downloaded YouTubers and transcript counts
- Show download progress and allow chatting while downloading (background thread)
Progress updates from a background thread do not automatically rerun the
Streamlit script.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


def _set_favicon(url: str) -> None:
    """Inject favicon links so the tab icon renders crisply without clipping."""
    st.markdown(
        f"""
        <link rel=\"icon\" href=\"{url}\" sizes=\"16x16\" />
        <link rel=\"icon\" href=\"{url}\" sizes=\"32x32\" />
        <link rel=\"shortcut icon\" href=\"{url}\" />
        """,
        unsafe_allow_html=True,
    )


def _ensure_state() -> None:
    """Initialize minimal session state for presentation-only dashboard."""
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("vector_dir", Path("data/vector_store"))


def _list_channels(transcripts_dir: Path) -> list[tuple[str, int]]:
    """Return [(channel_key, num_files)] for downloaded channels."""
    out: list[tuple[str, int]] = []
    if not transcripts_dir.exists():
        return out
    for p in sorted(transcripts_dir.iterdir()):
        if p.is_dir():
            count = len(list(p.glob("*.txt")))
            out.append((p.name, count))
    return out


def _setup_page() -> None:
    """Shared page setup: env, page config, style, state, heading."""
    load_dotenv()
    st.set_page_config(
        page_title="YouTube Summary",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Hide default Streamlit menu
    hide_streamlit_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_menu, unsafe_allow_html=True)
    # Use a proper PNG favicon to avoid oversized/clipped emoji icons in the tab
    _set_favicon(
        "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f3e0.png",
    )
    _ensure_state()
    st.title("YouTube Summary • Dashboard")
    st.caption(
        "Visão geral do projeto e métricas. As ações estão nas páginas laterais.",
    )


def _render_sidebar() -> None:
    """Sidebar navigation only (no actions on the dashboard)."""
    with st.sidebar:
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()


def _render_channels_summary() -> None:
    """Summary of downloaded channels and file counts."""
    with st.expander("YouTubers baixados (resumo)", expanded=False):
        rows = _list_channels(st.session_state.transcripts_dir)
        if not rows:
            st.info("Nenhum transcript baixado ainda.")
        else:
            for ch_key, count in rows:
                st.write(f"• {ch_key}: {count} arquivos")


def _dir_size_bytes(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _render_overview_metrics() -> None:
    rows = _list_channels(st.session_state.transcripts_dir)
    total_channels = len(rows)
    total_transcripts = sum(c for _, c in rows)
    vdir = st.session_state.vector_dir
    v_exists = vdir.exists()
    v_size = _dir_size_bytes(vdir) if v_exists else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Canais", total_channels)
    with c2:
        st.metric("Transcrições", total_transcripts)
    with c3:
        human = (
            f"{v_size / (1024 * 1024):.1f} MB"
            if v_size >= 1024 * 1024
            else f"{v_size} B"
        )
        st.metric("Tamanho do índice (aprox.)", human)


def _render_channels_chart_or_table() -> None:
    rows = _list_channels(st.session_state.transcripts_dir)
    if not rows:
        st.info("Nenhum transcript encontrado em data/transcripts.")
        return
    st.subheader("Canais por quantidade de transcrições")
    if pd is not None:
        df = pd.DataFrame(rows, columns=["Canal", "Transcrições"]).sort_values(
            by="Transcrições",
            ascending=False,
        )
        st.bar_chart(
            df.set_index("Canal"),
            horizontal=True,
            x_label="Transcrições",
            y_label="Canal",
            color="#8f8f8f",
        )
    else:
        for ch, cnt in sorted(rows, key=lambda x: x[1], reverse=True)[:20]:
            st.write(f"• {ch}: {cnt}")


def _render_navigation_cta() -> None:
    st.divider()
    st.subheader("Próximas ações")
    st.write("Use as páginas laterais para executar ações:")
    st.write("- ⬇️ Baixar novas transcrições")
    st.write("- 🧠 Indexar/Reindexar o acervo")
    st.write("- 💬 Conversar com os vídeos")


# (All download/chat UI removed from dashboard)


def main() -> None:
    """Render the Streamlit dashboard with presentation-only content."""
    _setup_page()
    _render_sidebar()
    _render_overview_metrics()
    _render_channels_chart_or_table()
    _render_channels_summary()
    _render_navigation_cta()


if __name__ == "__main__":
    main()
