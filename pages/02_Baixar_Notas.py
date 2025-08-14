"""Download transcripts page (Baixar Notas).

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""

# ruff: noqa: N999
from __future__ import annotations

import queue
import threading
import time
from contextlib import suppress
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.rag import TranscriptRAG
from src.utils.youtube_helpers import channel_key_from_url, filter_pending_urls
from src.youtube import YouTubeTranscriptManager


def _start_thread_with_streamlit_ctx(t: threading.Thread) -> None:
    """Start a thread with Streamlit ScriptRunContext to avoid console warnings."""
    add_script_run_ctx(t)
    t.start()


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("languages", ["pt", "en"])  # default
    ss.setdefault("subs_fallback", True)
    ss.setdefault("limit", None)
    ss.setdefault("channels_input", "")
    # Progress
    ss.setdefault("download_running", False)
    ss.setdefault("download_errors", [])
    ss.setdefault("download_last", "")
    ss.setdefault("download_saved", [])
    # Streaming indexing state
    ss.setdefault("stream_indexing", False)
    ss.setdefault("index_running", False)
    ss.setdefault("index_errors", [])
    ss.setdefault("index_added", 0)
    ss.setdefault("index_skipped", 0)
    # Queue will be created on start when enabled
    ss.setdefault("index_queue", None)


def _download_worker(  # noqa: PLR0913
    channels: list[str],
    transcripts_dir: Path,
    languages: list[str],
    limit: int | None,
    *,
    subs_fallback: bool,
    index_queue: queue.Queue | None = None,
) -> None:
    ss = st.session_state
    ss.download_running = True
    ss.download_errors = []
    ss.download_saved = []
    ss.download_last = ""

    download_errors = []
    download_saved = []
    download_last = ""

    yt = YouTubeTranscriptManager(base_dir=str(transcripts_dir))

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
        download_errors.append(f"Falha ao listar vídeos: {e}")

    # Filter out URLs that already have transcripts persisted
    per_channel_urls, _ = filter_pending_urls(
        per_channel_urls,
        transcripts_dir,
    )

    ss.download_errors = download_errors

    for ch, urls in per_channel_urls.items():
        channel_key = channel_key_from_url(ch)
        for url in urls:
            if not ss.download_running:
                break
            try:
                out = yt.save_transcript(
                    channel_key,
                    url,
                    languages=languages,
                    subs_fallback=subs_fallback,
                )
                download_saved.append(str(out))
                download_last = out.stem
                # Push to indexing queue if enabled
                if index_queue is not None:
                    with suppress(queue.Full):
                        index_queue.put(out, timeout=0.1)
            except Exception as e:  # noqa: BLE001
                download_errors.append(f"{url}: {e}")
            finally:
                ss.download_last = download_last
                ss.download_saved = download_saved
                ss.download_errors = download_errors
                time.sleep(0.05)

    ss.download_running = False
    # Signal index worker to finish when queue is empty
    if index_queue is not None:
        with suppress(queue.Full):
            index_queue.put(None, timeout=0.1)


def _index_worker(transcripts_root: Path, q: queue.Queue) -> None:
    ss = st.session_state
    ss.index_running = True
    ss.index_errors = []
    rag = TranscriptRAG()
    rag.transcripts_root = transcripts_root
    # Ensure DB is created lazily by add_transcript_file -> _ensure_db
    while True:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            # Periodically check if download stopped and queue is likely empty
            if not ss.download_running:
                # drain quickly if empty
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
            else:
                continue
        if item is None:
            break
        try:
            added, skipped = rag.add_transcript_file(item)
            ss.index_added += int(added)
            ss.index_skipped += int(skipped)
        except Exception as e:  # noqa: BLE001
            errs = ss.index_errors
            errs.append(f"{item}: {e}")
            ss.index_errors = errs[-200:]
        finally:
            with suppress(ValueError):
                q.task_done()
    ss.index_running = False


def _render_sidebar_nav() -> None:
    """Render sidebar navigation links."""
    with st.sidebar:
        st.header("Navegação")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/02_Baixar_Notas.py", label="Baixar Notas", icon="⬇️")
        st.page_link("pages/03_Indexacao.py", label="Indexação", icon="🧠")
        st.page_link("pages/04_Chat.py", label="Chat", icon="💬")
        st.divider()


def _render_title_and_caption() -> None:
    """Render title and caption for the page."""
    st.title("Baixar notas (transcrições)")
    st.caption("Você pode ir ao Chat enquanto baixa.")


def _render_inputs() -> None:
    """Render input controls and persist values to session_state."""
    ss = st.session_state
    ss.channels_input = st.text_area(
        "URLs dos canais (um por linha)",
        value=ss.channels_input,
        placeholder="https://www.youtube.com/@Canal1\nhttps://www.youtube.com/@Canal2",
        height=160,
    )
    cols = st.columns(3)
    with cols[0]:
        ss.subs_fallback = st.checkbox(
            "Usar legendas quando necessário",
            value=ss.subs_fallback,
        )
    with cols[1]:
        limit = st.number_input(
            "Limite por canal (0=ilimitado)",
            min_value=0,
            step=1,
            value=ss.limit or 0,
        )
        ss.limit = int(limit) if limit > 0 else None
    with cols[2]:
        langs = st.multiselect(
            "Idiomas",
            options=["pt", "en", "es", "fr", "de", "it"],
            default=ss.languages,
        )
        ss.languages = langs or ["pt", "en"]
    ss.stream_indexing = st.checkbox(
        "Indexar enquanto baixa (beta)",
        value=ss.stream_indexing,
        help=(
            "Indexa cada transcrição assim que for salva, reduzindo o tempo total "
            "até o chat ficar utilizável. Requer chave da Together AI válida."
        ),
    )


def _maybe_start_download() -> None:
    """Start background download thread if triggered by the user."""
    ss = st.session_state
    start_download = st.button(
        "Iniciar download em segundo plano",
        disabled=ss.download_running,
    )
    if not start_download:
        return
    channels = [x.strip() for x in ss.channels_input.splitlines() if x.strip()]
    if not channels:
        st.warning("Informe pelo menos um canal.")
        return
    # Prepare optional indexing worker
    q: queue.Queue | None = None
    if ss.stream_indexing:
        q = queue.Queue()
        ss.index_queue = q
        ss.index_added = 0
        ss.index_skipped = 0
        ss.index_errors = []
        t_index = threading.Thread(
            target=_index_worker,
            args=(ss.transcripts_dir, q),
            daemon=True,
        )
        _start_thread_with_streamlit_ctx(t_index)

    t = threading.Thread(
        target=_download_worker,
        args=(
            channels,
            ss.transcripts_dir,
            ss.languages,
            ss.limit,
        ),
        kwargs={"subs_fallback": ss.subs_fallback, "index_queue": q},
        daemon=True,
    )
    with st.spinner("Iniciando download em segundo plano..."):
        _start_thread_with_streamlit_ctx(t)
        time.sleep(0.2)


def _render_status_notice() -> None:
    """Render a simple status notice without progress bar."""
    ss = st.session_state
    if ss.stream_indexing and (ss.download_running or ss.index_running):
        st.info(
            (
                f"Indexando em paralelo... adicionados={ss.index_added} "
                f"pulados={ss.index_skipped}"
            ),
        )


def _render_results_section() -> None:
    """Render the last processed item, errors and saved files."""
    ss = st.session_state
    if ss.download_last:
        st.caption(f"Último vídeo processado: {ss.download_last}")
    cols2 = st.columns(2)
    with cols2[0]:
        if ss.download_errors:
            with st.expander("Erros (recentes)"):
                for err in ss.download_errors[-50:]:
                    st.write(f"- {err}")
    with cols2[1]:
        if ss.download_saved:
            with st.expander("Arquivos salvos (recentes)"):
                for p in ss.download_saved[-20:]:
                    st.write(p)
    if ss.stream_indexing:
        cols3 = st.columns(2)
        with cols3[0]:
            if ss.index_errors:
                with st.expander("Erros de indexação (recentes)"):
                    for err in ss.index_errors[-50:]:
                        st.write(f"- {err}")
        with cols3[1]:
            st.caption(
                f"Resumo da indexação: adicionados={ss.index_added}, "
                f"pulados={ss.index_skipped}",
            )


def main() -> None:
    """Render the page to download YouTube transcripts with progress."""
    st.set_page_config(page_title="Baixar Notas", page_icon="⬇️", layout="wide")

    hide_streamlit_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_menu, unsafe_allow_html=True)

    _ensure_state()
    _render_sidebar_nav()
    _render_title_and_caption()
    _render_inputs()
    _maybe_start_download()
    _render_status_notice()
    _render_results_section()


if __name__ == "__main__":
    main()
