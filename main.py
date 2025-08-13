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

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.rag import TranscriptRAG
from src.utils.youtube_helpers import channel_key_from_url, filter_pending_urls
from src.youtube import YouTubeTranscriptManager

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


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


# Removed local channel key helper; use utils.channel_key_from_url


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
    # Filter out URLs that already have transcripts persisted
    per_channel_urls, total_pending = filter_pending_urls(
        per_channel_urls, transcripts_dir,
    )
    ss.download_total = total_pending
    ss.download_done = 0

    # Download loop
    for ch, urls in per_channel_urls.items():
        channel_key = channel_key_from_url(ch)
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


# Attach Streamlit run context to a thread when available, then start it.
def _start_thread_with_streamlit_ctx(t: threading.Thread) -> None:
    """Start a thread with Streamlit ScriptRunContext to avoid console warnings."""
    add_script_run_ctx(t)
    t.start()


# ------------------------------- UI Definition -------------------------------
def _setup_page() -> None:
    """Shared page setup: env, page config, style, state, heading."""
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


def _render_sidebar() -> None:
    """Sidebar navigation, settings, and indexing actions."""
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
                with st.spinner("Reindexando transcrições..."):
                    _get_rag().index_transcripts(st.session_state.transcripts_dir)
            except Exception as e:  # noqa: BLE001
                st.error(f"Falha ao reindexar: {e}")
            else:
                st.success("Índice refeito com sucesso.")


def _render_channels_summary() -> None:
    """Summary of downloaded channels and file counts."""
    with st.expander("YouTubers baixados (resumo)", expanded=False):
        rows = _list_channels(st.session_state.transcripts_dir)
        if not rows:
            st.info("Nenhum transcript baixado ainda.")
        else:
            for ch_key, count in rows:
                st.write(f"• {ch_key}: {count} arquivos")


def _download_text_input() -> None:
    """Render channel URLs input area."""
    st.session_state.channels_input = st.text_area(
        "URLs dos canais (um por linha)",
        value=st.session_state.channels_input,
        placeholder="https://www.youtube.com/@Canal1\nhttps://www.youtube.com/@Canal2",
        height=120,
    )


def _handle_download_start_button() -> None:
    """Start background download when user clicks the button."""
    start_download = st.button(
        "Iniciar download em segundo plano",
        disabled=st.session_state.download_running,
    )
    if not start_download:
        return
    channels = [
        x.strip() for x in st.session_state.channels_input.splitlines() if x.strip()
    ]
    if not channels:
        st.warning("Informe pelo menos um canal.")
        return
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
    with st.spinner("Iniciando download em segundo plano..."):
        _start_thread_with_streamlit_ctx(t)
        time.sleep(0.2)
    st.success("Download iniciado. Acompanhe o progresso abaixo.")


def _render_progress_widgets() -> None:
    """Show progress bar and a running info message while downloading."""
    total = st.session_state.download_total
    done = st.session_state.download_done
    running = st.session_state.download_running
    progress = 0.0 if total == 0 else min(1.0, done / max(1, total))
    st.progress(progress, text=f"Progresso: {done}/{total}")
    if running:
        st.info("Baixando... você pode usar o chat ao lado enquanto isso.")


def _render_completion_and_details() -> None:
    """Show completion message, last processed item, errors and saved files."""
    total = st.session_state.download_total
    done = st.session_state.download_done
    running = st.session_state.download_running
    if total > 0 and not running and done >= total:
        st.success("Download concluído.")
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


def _render_download_section(col_left: DeltaGenerator) -> None:
    """Left column: download UI + progress and results."""
    with col_left:
        st.subheader("Baixar notas (transcrições)")
        _download_text_input()
        _handle_download_start_button()
        _render_progress_widgets()
        _render_completion_and_details()


def _render_chat_section(col_right: DeltaGenerator) -> None:
    """Right column: chat UI and behavior."""
    with col_right:
        st.subheader("Conversar com os vídeos")
        st.session_state.top_k = st.slider(
            "Número de passagens recuperadas (k)",
            1,
            10,
            st.session_state.top_k,
        )
        for msg in st.session_state.chat[-20:]:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))
        prompt = st.chat_input("Pergunte algo sobre os vídeos indexados...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Consultando o índice e gerando resposta..."):
                        answer = _get_rag().query(
                            prompt.strip(),
                            k=st.session_state.top_k,
                        )
                except Exception as e:  # noqa: BLE001
                    err = (
                        "Falha ao consultar. Verifique GROQ_API_KEY e o modelo. "
                        f"Detalhes: {e}"
                    )
                    st.error(err)
                    st.session_state.chat.append(
                        {"role": "user", "content": prompt},
                    )
                    st.session_state.chat.append(
                        {"role": "assistant", "content": err},
                    )
                else:
                    st.markdown(answer)
                    st.caption("Concluído.")
                    st.session_state.chat.append(
                        {"role": "user", "content": prompt},
                    )
                    st.session_state.chat.append(
                        {"role": "assistant", "content": answer},
                    )


def _render_two_column_layout() -> None:
    """Two-column main layout: download (left) and chat (right)."""
    col_left, col_right = st.columns(2)
    _render_download_section(col_left)
    _render_chat_section(col_right)


def main() -> None:
    """Render the Streamlit dashboard to download, index, and chat."""
    _setup_page()
    _render_sidebar()
    _render_channels_summary()
    _render_two_column_layout()


if __name__ == "__main__":
    main()
