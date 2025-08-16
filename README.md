# YouTube Summary – Transcript + RAG

App multipáginas em Streamlit para buscar transcrições de vídeos de canais do YouTube, indexar com embeddings Together AI (Chroma) e consultar via Groq (RAG). Projetado para Python 3.13, `uv`, `ruff`, `pytest`. Caminhos usando `pathlib.Path`.

## Requisitos

- Python 3.13
- uv (gerenciador de pacotes)
- Chaves de API em `.env`:
  - `TOGETHER_API_KEY` (obrigatório para embeddings Together)
  - `GROQ_API_KEY` (obrigatório para LLM Groq)
  - `GROQ_MODEL` (opcional; padrão: `llama-3.3-70b-versatile`)
  - `TOGETHER_EMBEDDINGS_MODEL` (opcional; padrão: `intfloat/multilingual-e5-large-instruct`)
  - `HTTP_URL` (opcional; proxy HTTP p/ reduzir bloqueios)
- `yt-dlp` (já vem como dependência Python)

Notas de suporte a conteúdo YouTube:

- São suportados: vídeos padrão (`youtube.com/watch` e `youtu.be`), Shorts e lives gravadas (`youtube.com/live`).
- São excluídos: embeds e clips (não processados em nenhuma etapa).

## Instalação

Instalar dependências

```bash
uv sync
```

Configurar variáveis de ambiente

- Copie `.env.example` para `.env` e preencha:

```bash
TOGETHER_API_KEY=...
GROQ_API_KEY=...
# Modelos (opcionais)
GROQ_MODEL=llama-3.3-70b-versatile
TOGETHER_EMBEDDINGS_MODEL=intfloat/multilingual-e5-large-instruct
# Proxy (opcional)
HTTP_URL=http://user:pass@proxy.example.org:8080
```

## Estrutura do projeto

- `src/youtube.py`
  - Classe `YouTubeTranscriptManager` para extrair URLs e salvar transcrições em `data/transcripts/<canal>/<video_id>.txt`.
- `src/rag.py`
  - Classe `TranscriptRAG` para indexar (Chroma) e consultar (Groq + Together Embeddings).
- `main.py`
  - Dashboard Streamlit (apenas apresentação/overview e navegação lateral).
- `pages/01_YouTubers.py`
  - Lista canais baixados e transcrições disponíveis.
- `pages/02_Baixar_Notas.py`
  - Baixa transcrições por canal com progresso, limites, idiomas e fallback de legendas.
- `pages/03_Indexacao.py`
  - Indexa/reindexa o vetor (Chroma) e faz atualizações incrementais.
- `pages/04_Chat.py`
  - Interface de chat para consultar o índice RAG com seleção dinâmica de modelo Groq.
- `data/transcripts/`
  - Transcrições por canal.
- `data/vector_store/`
  - Persistência local da base vetorial (Chroma).

## Uso rápido (Streamlit)

### Rodar o app

```bash
uv run streamlit run main.py
```

### Navegação (sidebar)

- Dashboard, YouTubers, Download, Indexação, Chat.

### Fluxo típico

- Adicione um ou mais canais em `Download` e clique em “Baixar”.
- Faça a indexação em `Indexação` (reconstruir do zero ou atualizar incrementalmente).
- Faça perguntas em `Chat` (escolha o modelo Groq na barra lateral se necessário).

## Páginas do app (Streamlit)

- YouTubers (`pages/01_YouTubers.py`)
  - Lista canais com transcrições salvas; expanda para ver os arquivos `.txt`.
- Download (`pages/02_Baixar_Notas.py`)
  - Baixa transcrições com thread em background, progresso, limites por canal, seleção de idiomas (`pt`, `en`) e opção de fallback de legendas (yt-dlp) quando não houver transcript.
  - Suporta “baixar novos vídeos para todos os canais existentes”.
- Indexação (`pages/03_Indexacao.py`)
  - Recria o índice (Chroma) do zero a partir de `data/transcripts/` ou faz atualização incremental por canal.
  - Permite reindexar/remover canal do índice. A configuração de embeddings vem do ambiente.
- Chat (`pages/04_Chat.py`)
  - Chat RAG com seleção dinâmica do modelo Groq na sidebar (com refresh e entrada customizada). Atualiza `st.session_state.groq_model_name` e reinicializa `TranscriptRAG` quando muda.

## Fluxos comuns (no app)

- Baixar primeiros vídeos de um canal (com fallback de legendas) e indexar:
  - Vá em `Download` → cole a(s) URL(s) do canal → defina `limit` → ative `fallback de legendas` se quiser → “Baixar”.
  - Em `Indexação` → “Recriar do zero” para montar a coleção inicial.
  - Em `Chat` → escolha o modelo Groq e pergunte.
- Atualizações recorrentes:
  - Em `Download` → “Baixar novos vídeos de todos os canais”.
  - Em `Indexação` → “Atualizar incrementalmente (todos ou canal selecionado)”.

## Como funciona (resumo)

- `YouTubeTranscriptManager.get_video_urls_from_channel()` usa `yt-dlp` para listar URLs e ignora vídeos com disponibilidade restrita (privado/membros) quando detectável via `availability`. Apenas conteúdos suportados são processados (watch, youtu.be, Shorts, lives). Embeds e clips são ignorados.
- `YouTubeTranscriptManager.fetch_transcript()` usa `youtube-transcript-api` e tenta idiomas em ordem (`pt`, `en`). Quando indisponível e habilitado, tenta fallback de legendas (VTT/SRT) via `yt-dlp`.
- Transcrições são salvas em `data/transcripts/<canal>/<video_id>.txt`.
- `TranscriptRAG.index_transcripts()` lê `.txt`, quebra em chunks e embeda com Together (`TOGETHER_EMBEDDINGS_MODEL`) persistindo em `Chroma`.
- O LLM é Groq (`GROQ_MODEL`, padrão `llama-3.3-70b-versatile`), inicializado sob demanda.
- `TranscriptRAG.query()` e `query_with_sources()` executam a consulta e retornam resposta (e fontes, quando solicitado).

## Troubleshooting

- Streamlit não abre no navegador
  - Verifique o terminal para a URL local (ex.: http://localhost:8501) e abra manualmente.
- Erro ao inicializar o modelo Groq
  - Confirme `GROQ_API_KEY` e se o nome em `GROQ_MODEL` existe e está acessível.
- Embeddings Together não inicializam
  - Confirme `TOGETHER_API_KEY` e (opcionalmente) `TOGETHER_EMBEDDINGS_MODEL`.
- `yt-dlp` não funciona
  - Confirme que o comando está acessível no ambiente virtual (use `uv run ...`).
- Chroma: erro de dimensão/coleção
  - O nome da coleção depende do modelo de embedding. Se você trocou de embedding/modelo e ocorrer erro ao adicionar documentos, recrie o índice do zero em `Indexação` ou apague `data/vector_store/` manualmente.

## Proxies (reduce 429 / IP bans)

- Set `HTTP_URL` in your `.env`.
- Both `youtube-transcript-api` and `yt-dlp` are configured automatically:
  - `youtube-transcript-api`: uses `GenericProxyConfig` internally.
  - `yt-dlp`: gets `--proxy <url>` for all calls (`--get-id`, subtitle download).
- Example:

```bash
# .env
HTTP_URL=http://user:pass@proxy.example.org:8080
```

Notes:

- Proxies do not fully eliminate blocks; prefer providers with IP rotation.
- Use `--limit` and run in batches to reduce request rate.
