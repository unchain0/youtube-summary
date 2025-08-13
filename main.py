"""Streamlit app replacing the CLI for YouTube transcript RAG.

UI language: Portuguese (pt-BR). Code comments/logs: English.

Features (parity with CLI):
- Fetch transcripts for one or more channels (with languages, limit, subs fallback)
- Build/Update the vector store (rebuild or incremental)
- Query the vector store (chat)
- View downloaded YouTubers and transcript counts
- Show download progress and allow chatting while downloading (background thread)
Progress updates from a background thread do not automatically rerun the
Streamlit script; enable auto-refresh below to see live progress.

"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import TranscriptRAG
from src.youtube import YouTubeTranscriptManager


# ----------------------------- Helpers (backend) -----------------------------
def _ensure_state() -> None:
    """Initialize Streamlit session state keys lazily."""
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("languages", ["pt", "en"])  # default preference
    ss.setdefault("subs_fallback", True)
    ss.setdefault("limit", None)
    ss.setdefault("channels_input", "")
    ss.setdefault("rag", None)
    ss.setdefault("groq_model", "")  # empty -> use env/default inside RAG
    # Download progress state
    ss.setdefault("download_running", False)
    ss.setdefault("download_total", 0)
    ss.setdefault("download_done", 0)
    ss.setdefault("download_errors", [])
    ss.setdefault("download_last", "")
    ss.setdefault("download_saved", [])
    # Chat history
    ss.setdefault("chat", [])  # list[dict(role, content)]
    ss.setdefault("top_k", 4)


def _channel_key_from_url(url: str) -> str:
    """Derive a filesystem-friendly channel key from a YouTube channel URL."""
    return url.rstrip("/").split("/")[-1].lstrip("@") or "channel"


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


def _download_worker(
    channels: list[str],
    transcripts_dir: Path,
    languages: list[str],
    *,
    limit: int | None,
    subs_fallback: bool,
) -> None:
    """Background download worker that updates session_state for progress.

    Notes:
    - Avoids using Streamlit APIs directly inside the thread; only session_state.
    - Pre-fetches video URL lists for accurate progress totals.

    """
    ss = st.session_state
    ss.download_running = True
    ss.download_errors = []
    ss.download_saved = []
    ss.download_last = ""
    yt = YouTubeTranscriptManager(base_dir=str(transcripts_dir))

    # Pre-compute URL lists per channel for accurate totals
    per_channel_urls: dict[str, list[str]] = {}
    total = 0
    try:
        for ch in channels:
            urls = yt.get_video_urls_from_channel(ch)
            if limit is not None:
                urls = urls[: int(limit)]
            per_channel_urls[ch] = urls
            total += len(urls)
    except Exception as e:  # noqa: BLE001
        ss.download_errors.append(f"Falha ao listar vídeos: {e}")
    ss.download_total = total
    ss.download_done = 0

    # Download loop
    for ch, urls in per_channel_urls.items():
        channel_key = _channel_key_from_url(ch)
        for url in urls:
            if not ss.download_running:
                break  # allow cancellation in the future
            try:
                out = yt.save_transcript(
                    channel_key,
                    url,
                    languages=languages,
                    subs_fallback=subs_fallback,
                )
                ss.download_saved.append(str(out))
                ss.download_last = out.stem
            except Exception as e:  # noqa: BLE001
                ss.download_errors.append(f"{url}: {e}")
            finally:
                ss.download_done += 1
                # Gentle pace to keep UI responsive and avoid rate issues
                time.sleep(0.05)

    ss.download_running = False


def _get_rag() -> TranscriptRAG:
    """Return a cached TranscriptRAG instance bound to current vector_dir."""
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(vector_dir=ss.vector_dir)
    return ss.rag


# ------------------------------- UI Definition -------------------------------
def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Render the Streamlit dashboard to download, index, and chat."""
    load_dotenv()
    st.set_page_config(
        page_title="YouTube Summary",
        page_icon="🏠",
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
    _ensure_state()

    st.title("YouTube Summary • Streamlit")
    st.caption("Baixe notas, indexe e converse — tudo em um só lugar.")

    # Sidebar settings
    with st.sidebar:
        # Navigation with icons
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()

        st.header("Configurações")

        langs = st.multiselect(
            "Idiomas preferidos",
            options=["pt", "en", "es", "fr", "de", "it"],
            default=st.session_state.languages,
            help="Ordem de preferência para transcrições",
        )
        st.session_state.languages = langs or ["pt", "en"]

        limit = st.number_input(
            "Limite de vídeos por canal (opcional)",
            min_value=0,
            step=1,
            value=st.session_state.limit or 0,
            help="0 = sem limite",
        )
        st.session_state.limit = int(limit) if limit > 0 else None

        st.session_state.subs_fallback = st.checkbox(
            "Usar legendas via yt-dlp quando não houver transcript",
            value=st.session_state.subs_fallback,
        )

        st.divider()
        st.subheader("Indexação (RAG)")
        if st.button("Reindexar do zero (rebuild)"):
            try:
                _get_rag().index_transcripts(st.session_state.transcripts_dir)
                st.success("Índice refeito com sucesso.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Falha ao reindexar: {e}")

    # Summary of downloaded channels
    with st.expander("YouTubers baixados (resumo)", expanded=False):
        rows = _list_channels(st.session_state.transcripts_dir)
        if not rows:
            st.info("Nenhum transcript baixado ainda.")
        else:
            for ch_key, count in rows:
                st.write(f"• {ch_key}: {count} arquivos")

    # Two-column layout: Download (left) and Chat (right)
    col_left, col_right = st.columns(2)

    # Left: Download Notes with progress
    with col_left:
        st.subheader("Baixar notas (transcrições)")
        st.session_state.channels_input = st.text_area(
            "URLs dos canais (um por linha)",
            value=st.session_state.channels_input,
            placeholder="https://www.youtube.com/@Canal1\nhttps://www.youtube.com/@Canal2",
            height=120,
        )
        start_download = st.button(
            "Iniciar download em segundo plano",
            disabled=st.session_state.download_running,
        )
        if start_download:
            channels = [
                x.strip()
                for x in st.session_state.channels_input.splitlines()
                if x.strip()
            ]
            if not channels:
                st.warning("Informe pelo menos um canal.")
            else:
                t = threading.Thread(
                    target=_download_worker,
                    args=(
                        channels,
                        st.session_state.transcripts_dir,
                        st.session_state.languages,
                    ),
                    kwargs={
                        "limit": st.session_state.limit,
                        "subs_fallback": st.session_state.subs_fallback,
                    },
                    daemon=True,
                )
                t.start()

        # Progress widgets
        total = st.session_state.download_total
        done = st.session_state.download_done
        running = st.session_state.download_running
        progress = 0.0 if total == 0 else min(1.0, done / max(1, total))
        st.progress(progress, text=f"Progresso: {done}/{total}")
        if running:
            st.info("Baixando... você pode usar o chat ao lado enquanto isso.")
            auto = st.checkbox(
                "Auto-atualizar progresso",
                value=True,
                key="auto_refresh_main",
                help=(
                    "Quando ativo, a página se atualiza a cada segundo enquanto baixa."
                ),
            )
            if auto:
                time.sleep(1.0)
                st.experimental_rerun()
        if st.session_state.download_last:
            st.caption(f"Último vídeo processado: {st.session_state.download_last}")
        if st.session_state.download_errors:
            with st.expander("Erros (clique para ver)"):
                for err in st.session_state.download_errors[-50:]:
                    st.write(f"- {err}")
        if st.session_state.download_saved:
            with st.expander("Arquivos salvos (recentes)"):
                for p in st.session_state.download_saved[-20:]:
                    st.write(p)

    with col_right:  # Right: Chat
        st.subheader("Conversar com os vídeos")
        st.session_state.top_k = st.slider(
            "Número de passagens recuperadas (k)",
            1,
            10,
            st.session_state.top_k,
        )
        # Render chat history using Streamlit chat components
        for msg in st.session_state.chat[-20:]:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))

        # Chat input (recommended API)
        prompt = st.chat_input("Pergunte algo sobre os vídeos indexados...")
        if prompt:
            # Show the user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                answer = _get_rag().query(prompt.strip(), k=st.session_state.top_k)
            except Exception as e:  # noqa: BLE001
                err = (
                    "Falha ao consultar. Verifique GROQ_API_KEY e o modelo. "
                    f"Detalhes: {e}"
                )
                with st.chat_message("assistant"):
                    st.error(err)
                # persist history
                st.session_state.chat.append({"role": "user", "content": prompt})
                st.session_state.chat.append({"role": "assistant", "content": err})
            else:
                # Show assistant message and persist history
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.chat.append({"role": "user", "content": prompt})
                st.session_state.chat.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
