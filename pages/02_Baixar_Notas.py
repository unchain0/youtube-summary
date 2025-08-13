"""Download transcripts page (Baixar Notas).

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""
# ruff: noqa: N999
from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st

from src.youtube import YouTubeTranscriptManager


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("languages", ["pt", "en"])  # default
    ss.setdefault("subs_fallback", True)
    ss.setdefault("limit", None)
    ss.setdefault("channels_input", "")
    # Progress
    ss.setdefault("download_running", False)
    ss.setdefault("download_total", 0)
    ss.setdefault("download_done", 0)
    ss.setdefault("download_errors", [])
    ss.setdefault("download_last", "")
    ss.setdefault("download_saved", [])


def _channel_key_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].lstrip("@") or "channel"


def _download_worker(
    channels: list[str],
    transcripts_dir: Path,
    languages: list[str],
    limit: int | None,
    *,
    subs_fallback: bool,
) -> None:
    ss = st.session_state
    ss.download_running = True
    ss.download_errors = []
    ss.download_saved = []
    ss.download_last = ""

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
        ss.download_errors.append(f"Falha ao listar vídeos: {e}")
    ss.download_total = total
    ss.download_done = 0

    for ch, urls in per_channel_urls.items():
        channel_key = _channel_key_from_url(ch)
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
                ss.download_saved.append(str(out))
                ss.download_last = out.stem
            except Exception as e:  # noqa: BLE001
                ss.download_errors.append(f"{url}: {e}")
            finally:
                ss.download_done += 1
                time.sleep(0.05)

    ss.download_running = False


def main() -> None:  # noqa: PLR0915
    """Render the page to download YouTube transcripts with progress."""
    st.set_page_config(page_title="Baixar Notas", page_icon="⬇️", layout="wide")

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

    st.title("Baixar notas (transcrições)")
    st.caption("O progresso é mostrado abaixo; você pode ir ao Chat enquanto baixa.")

    st.session_state.channels_input = st.text_area(
        "URLs dos canais (um por linha)",
        value=st.session_state.channels_input,
        placeholder="https://www.youtube.com/@Canal1\nhttps://www.youtube.com/@Canal2",
        height=160,
    )

    cols = st.columns(3)
    with cols[0]:
        st.session_state.subs_fallback = st.checkbox(
            "Usar legendas quando necessário",
            value=st.session_state.subs_fallback,
        )
    with cols[1]:
        limit = st.number_input(
            "Limite por canal (0=ilimitado)",
            min_value=0,
            step=1,
            value=st.session_state.limit or 0,
        )
        st.session_state.limit = int(limit) if limit > 0 else None
    with cols[2]:
        langs = st.multiselect(
            "Idiomas",
            options=["pt", "en", "es", "fr", "de", "it"],
            default=st.session_state.languages,
        )
        st.session_state.languages = langs or ["pt", "en"]

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
                    st.session_state.limit,
                ),
                kwargs={"subs_fallback": st.session_state.subs_fallback},
                daemon=True,
            )
            t.start()

    total = st.session_state.download_total
    done = st.session_state.download_done
    running = st.session_state.download_running
    progress = 0.0 if total == 0 else min(1.0, done / max(1, total))
    st.progress(progress, text=f"Progresso: {done}/{total}")
    if running:
        auto = st.checkbox(
            "Auto-atualizar progresso",
            value=True,
            key="auto_refresh_download_page",
            help="Quando ativo, a página se atualiza a cada segundo enquanto baixa.",
        )
        if auto:
            time.sleep(1.0)
            st.experimental_rerun()

    if st.session_state.download_last:
        st.caption(f"Último vídeo processado: {st.session_state.download_last}")

    cols2 = st.columns(2)
    with cols2[0]:
        if st.session_state.download_errors:
            with st.expander("Erros (recentes)"):
                for err in st.session_state.download_errors[-50:]:
                    st.write(f"- {err}")
    with cols2[1]:
        if st.session_state.download_saved:
            with st.expander("Arquivos salvos (recentes)"):
                for p in st.session_state.download_saved[-20:]:
                    st.write(p)


if __name__ == "__main__":
    main()
