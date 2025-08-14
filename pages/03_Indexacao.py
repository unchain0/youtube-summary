"""Indexing (RAG) page to rebuild or update the vector store.

UI language: Portuguese (pt-BR). Code/docstrings/logs in English.
"""

# ruff: noqa: N999
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from together import Together

from src.rag import TranscriptRAG


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("transcripts_dir", Path("data/transcripts"))
    ss.setdefault("vector_dir", Path("data/vector_store"))
    ss.setdefault("rag", None)
    # Model selections (overrides env)
    ss.setdefault(
        "embed_model_name",
        os.getenv(
            "TOGETHER_EMBEDDINGS_MODEL",
            "intfloat/multilingual-e5-large-instruct",
        ),
    )
    ss.setdefault(
        "groq_model_name",
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


def _get_rag() -> TranscriptRAG:
    ss = st.session_state
    if ss.rag is None or not isinstance(ss.rag, TranscriptRAG):
        ss.rag = TranscriptRAG(
            vector_dir=ss.vector_dir,
            embed_model_name=ss.embed_model_name,
            groq_model=ss.groq_model_name,
        )
    return ss.rag


def _list_channels(transcripts_dir: Path) -> list[str]:
    if not transcripts_dir.exists():
        return []
    return [p.name for p in sorted(transcripts_dir.iterdir()) if p.is_dir()]


@st.cache_data(ttl=900, show_spinner=False)
def _list_together_embedding_models(api_key: str | None) -> list[str]:
    """Fetch TogetherAI models and filter to embeddings.

    Returns an empty list on error or when API key is missing.
    """
    try:
        if not api_key:
            return []
        client = Together(api_key=api_key)
        resp = client.models.list()
        items = getattr(resp, "data", None) or getattr(resp, "models", None) or resp
        ids: list[str] = []
        for m in items:
            # Support both object-like and dict-like entries
            mid = (
                str(getattr(m, "id", ""))
                if not isinstance(m, dict)
                else str(m.get("id", ""))
            )
            mtype = (
                str(getattr(m, "type", ""))
                if not isinstance(m, dict)
                else str(m.get("type", ""))
            )
            if mid and (
                mtype.lower() in {"embedding", "embeddings"} or "embed" in mid.lower()
            ):
                ids.append(mid)
        if ids:
            return sorted(set(ids))
    except Exception:  # noqa: BLE001
        ids = []

    # REST fallback when SDK returns empty or fails
    try:
        req = urllib.request.Request(
            url="https://api.together.xyz/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            body = resp.read()
        payload = json.loads(body)
        items = payload.get("data", [])
        ids_http: list[str] = []
        for m in items:
            mid = str(m.get("id", ""))
            mtype = str(m.get("type", ""))
            if mid and (
                mtype.lower() in {"embedding", "embeddings"} or "embed" in mid.lower()
            ):
                ids_http.append(mid)
        return sorted(set(ids_http))
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(ttl=900, show_spinner=False)
def _list_groq_models(api_key: str | None) -> list[str]:
    """Fetch Groq models via SDK.

    Returns an empty list on error or when API key is missing.
    """
    try:
        if not api_key:
            return []
        client = Groq(api_key=api_key)
        resp = client.models.list()
        data = getattr(resp, "data", None)
        items = data if data is not None else getattr(resp, "models", [])
        ids = [str(getattr(m, "id", "")) for m in items]
        return sorted({i for i in ids if i})
    except Exception:  # noqa: BLE001
        return []


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


def _render_model_settings() -> None:
    """Configure Together Embeddings and Groq model from the UI."""
    st.divider()
    st.subheader("Configurações de Modelos")
    st.caption(
        "Trocar o modelo de embeddings pode exigir refazer o índice "
        "(reindexar do zero).",
    )

    # Presets + dynamic lists
    embed_presets = ["intfloat/multilingual-e5-large-instruct", "Custom…"]
    groq_presets = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "Custom…"]

    together_key = os.getenv("TOGETHER_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    dyn_embeds = _list_together_embedding_models(together_key)
    dyn_groq = _list_groq_models(groq_key)

    embed_options = [
        *sorted(set(embed_presets[:-1]) | set(dyn_embeds)),
        "Custom…",
    ]
    groq_options = [
        *sorted(set(groq_presets[:-1]) | set(dyn_groq)),
        "Custom…",
    ]

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        if st.button("Atualizar lista de modelos"):
            st.cache_data.clear()
            st.rerun()

    _render_model_listing_hints(dyn_embeds, dyn_groq, together_key, groq_key)

    c1, c2 = st.columns(2)
    with c1:
        current_embed = st.session_state.embed_model_name
        if current_embed in embed_options:
            idx = embed_options.index(current_embed)
        else:
            idx = len(embed_options) - 1  # Custom…
        sel = st.selectbox("Embeddings (Together)", embed_options, index=idx)
        if sel == "Custom…":
            current_embed = st.text_input(
                "Nome do modelo de embeddings (Together)",
                value=st.session_state.embed_model_name,
            ).strip()
        else:
            current_embed = sel
        changed_embed = current_embed != st.session_state.embed_model_name

    with c2:
        current_groq = st.session_state.groq_model_name
        if current_groq in groq_options:
            idx_g = groq_options.index(current_groq)
        else:
            idx_g = len(groq_options) - 1  # Custom…
        sel_g = st.selectbox("LLM (Groq)", groq_options, index=idx_g)
        if sel_g == "Custom…":
            current_groq = st.text_input(
                "Nome do modelo Groq",
                value=st.session_state.groq_model_name,
            ).strip()
        else:
            current_groq = sel_g
        changed_groq = current_groq != st.session_state.groq_model_name

    if changed_embed or changed_groq:
        st.session_state.embed_model_name = current_embed
        st.session_state.groq_model_name = current_groq
        # Drop cached RAG to force re-instantiation with new models
        st.session_state.rag = None
        st.info(
            "Modelos atualizados. Novas indexações/consultas usarão as novas "
            "configurações.",
        )


def _render_model_listing_hints(
    dyn_embeds: list[str],
    dyn_groq: list[str],
    together_key: str | None,
    groq_key: str | None,
) -> None:
    """Render helper messages when dynamic model lists are empty."""
    if not dyn_embeds:
        if not together_key:
            st.info(
                "Defina TOGETHER_API_KEY para listar modelos de embeddings do "
                "Together.",
            )
        else:
            st.warning(
                "Não foi possível recuperar modelos do Together agora. Verifique "
                "sua chave/permissões ou tente novamente com 'Atualizar lista de "
                "modelos'.",
            )
    if not dyn_groq:
        if not groq_key:
            st.info("Defina GROQ_API_KEY para listar modelos Groq.")
        else:
            st.warning(
                "Não foi possível recuperar modelos do Groq agora. Verifique sua "
                "chave/permissões ou tente novamente com 'Atualizar lista de "
                "modelos'.",
            )


def _update_channels(rag: TranscriptRAG, channel_names: list[str]) -> None:
    """Update channels incrementally without any artificial pause."""
    for ch in channel_names:
        ch_dir = st.session_state.transcripts_dir / ch
        rag.add_channel(ch_dir)


def _render_rebuild_section() -> None:
    """Reindex from scratch section."""
    st.subheader("Reindexar do zero")
    if st.button("Reindexar agora"):
        try:
            rag = _get_rag()
            with st.spinner("Reindexando transcrições..."):
                progress = st.progress(0)
                status = st.empty()

                total_holder = {"total": 1}

                def on_progress(curr: int, total: int, rel: str) -> None:
                    total_holder["total"] = max(1, total)
                    progress.progress(min(1.0, curr / max(1, total)))
                    status.write(f"Processando: {rel} ({curr}/{total})")

                summary = rag.index_transcripts(
                    st.session_state.transcripts_dir,
                    on_progress=on_progress,
                )
                progress.progress(1.0)
                status.write("Concluído.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao reindexar: {e}")
        else:
            st.success(
                "Índice refeito com sucesso. "
                f"Arquivos: {int(summary.get('files', 0))} | "
                f"Adicionados: {int(summary.get('added', 0))} | "
                f"Pulados: {int(summary.get('skipped', 0))} | "
                f"Tempo: {summary.get('duration_s', 0.0):.2f}s",
            )


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
                total_added = 0
                total_skipped = 0
                for ch in selected:
                    a, s = rag.add_channel(st.session_state.transcripts_dir / ch)
                    total_added += a
                    total_skipped += s
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao atualizar: {e}")
        else:
            st.success(
                "Canais atualizados. "
                f"Adicionados: {total_added} | Pulados: {total_skipped}",
            )

    if update_all:
        try:
            rag = _get_rag()
            with st.spinner("Atualizando todos os canais..."):
                total_added = 0
                total_skipped = 0
                for ch in channels:
                    a, s = rag.add_channel(st.session_state.transcripts_dir / ch)
                    total_added += a
                    total_skipped += s
        except Exception as e:  # noqa: BLE001
            st.error(f"Falha ao atualizar todos: {e}")
        else:
            st.success(
                "Todos os canais atualizados. "
                f"Adicionados: {total_added} | Pulados: {total_skipped}",
            )


def _render_channel_management() -> None:
    """List, reindex or remove channels from the vector index."""
    st.divider()
    st.subheader("Gerenciar Canais")
    channels = _list_channels(st.session_state.transcripts_dir)
    if not channels:
        st.info("Nenhum canal encontrado em data/transcripts.")
        return
    for ch in channels:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(ch)
        with col2:
            if st.button("Reindexar", key=f"reindex_{ch}"):
                try:
                    a, s = _get_rag().add_channel(st.session_state.transcripts_dir / ch)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Falha ao reindexar {ch}: {e}")
                else:
                    st.success(f"{ch}: adicionados {a}, pulados {s}")
        with col3:
            if st.button("Remover do índice", key=f"remove_{ch}"):
                try:
                    removed = _get_rag().remove_channel_from_index(ch)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Falha ao remover {ch}: {e}")
                else:
                    st.warning(
                        f"{ch}: removido do índice (ids encontradas: {removed}).",
                    )


def main() -> None:
    """Index and update the vector store."""
    _configure_page()
    _render_sidebar_nav()

    st.title("Indexação (RAG)")
    st.caption("Refaça o índice do zero ou atualize por canal.")

    _render_paths()
    _render_model_settings()
    _render_rebuild_section()
    _render_incremental_update_section()
    _render_channel_management()


if __name__ == "__main__":
    main()
