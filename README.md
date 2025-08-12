# YouTube Summary – Transcript + RAG

Busca transcrições de vídeos de canais do YouTube, indexa com embeddings locais (FastEmbed) e responde perguntas via modelo Groq (RAG). Projetado para Python 3.13, `uv`, `ruff`, `pytest`, `pyrefly`. Caminhos usando `pathlib.Path`.

## Requisitos

- Python 3.13
- uv (gerenciador de pacotes)
- Chaves de API em `.env`:
  - `GROQ_API_KEY`
  - `GROQ_MODEL` (opcional, padrão: `llama3-70b-8192`)
  - `YT_PROXY_URL` (opcional, e.g. `http://host:port` ou `http://user:pass@host:port`)
  - `YT_PROXY_USERNAME` (opcional, usado se não embutir user:pass em `YT_PROXY_URL`)
  - `YT_PROXY_PASSWORD` (opcional, usado se não embutir user:pass em `YT_PROXY_URL`)
- `yt-dlp` (instalado como dependência Python)

## Instalação

Instalar dependências

```bash
uv sync
```

Configurar variáveis de ambiente

- Copie `.env.example` para `.env` e preencha:

```bash
GROQ_API_KEY=...
# Proxy (opcional)
HTTP_URL=http://proxy.myisp.com:3128
```

## Estrutura do projeto

- `src/youtube.py`
  - Classe `YouTubeTranscriptManager` para extrair URLs e salvar transcrições em `data/transcripts/<canal>/<video_id>.txt`.
- `src/rag.py`
  - Classe `TranscriptRAG` para indexar (Chroma) e consultar (Groq + FastEmbed).
- `main.py`
  - CLI com todos os comandos do fluxo (buscar, indexar, consultar).
- `data/transcripts/`
  - Transcrições por canal.
- `data/vector_store/`
  - Persistência local da base vetorial (Chroma).

## Uso rápido

Use o `main.py` como entrypoint do CLI.

- Ajuda:

```bash
python main.py -h
```

- Buscar transcrições, reindexar e consultar:

```bash
python main.py https://www.youtube.com/@PapaHardware \
  --limit 3 --rebuild --languages pt,en --subs \
  --query "Quais são os principais temas do canal?"
```

Observações:

- Idiomas válidos: apenas `pt` e `en`.
- `--limit` controla quantos vídeos por canal baixar.
- `--rebuild` recria a base Chroma a partir de todos os `.txt`.
- Sem `--rebuild`, o CLI adiciona incrementalmente novos arquivos.
- `--subs` tenta baixar legendas (auto/manuais) via `yt-dlp` quando não houver transcrição.

## Opções do CLI

- `channels` (posicional)
  - Um ou mais URLs de canal do YouTube (ex.: `https://www.youtube.com/@Handle`)
- `--limit <int>`
  - Limita o número de vídeos por canal
- `--languages <str>`
  - Idiomas separados por vírgula (ex.: `pt,en`)
- `--rebuild`
  - Reconstrói todo o índice a partir de `data/transcripts/`
- `--query "<pergunta>"`
  - Faz uma pergunta usando o RAG
- `--transcripts-dir <pasta>`
  - Padrão: `data/transcripts`
- `--vector-dir <pasta>`
  - Padrão: `data/vector_store`
- `--subs`
  - Baixa legendas (automáticas/manuais) via `yt-dlp` como fallback

## Fluxos comuns

- Baixar transcrições (sem indexar):

```bash
python main.py https://www.youtube.com/@Handle --limit 5 --languages pt,en
```

- Indexar tudo (rebuild):

```bash
python main.py https://www.youtube.com/@Handle --rebuild
```

- Atualizar incrementalmente:

```bash
python main.py https://www.youtube.com/@Handle --limit 10
```

- Perguntar após atualizar índice:

```bash
python main.py https://www.youtube.com/@Handle --limit 5 \
  --query "Resumo dos últimos vídeos"
```

## Como funciona (resumo)

- `YouTubeTranscriptManager.get_video_urls_from_channel()` usa `yt-dlp` para listar URLs e ignora vídeos com disponibilidade restrita (privado/membros), quando detectável via `availability`.
- `YouTubeTranscriptManager.fetch_transcript()` usa `youtube-transcript-api` e tenta idiomas em ordem (`pt`, `en`). Caso indisponível, com `--subs` tenta baixar legendas (VTT/SRT).
- Transcrições são salvas em `data/transcripts/<canal>/<video_id>.txt`.
- `TranscriptRAG.index_transcripts()` lê `.txt`, quebra em chunks, embeda com FastEmbed e persiste no `Chroma`.
- `TranscriptRAG.as_chain()` cria a cadeia com Groq (modelo definido por `GROQ_MODEL`, padrão `llama3-70b-8192`) e o prompt em PT-BR.
- `TranscriptRAG.query()` executa a pergunta no pipeline e retorna a resposta.

## Troubleshooting

- Módulos não encontrados (dotenv, langchain, etc.)
  - Rode: `uv sync`
- `yt-dlp` não funciona
  - Confirme que o comando está acessível no ambiente virtual ativado pelo `uv`.
- Imports LangChain/Groq
  - Dependências incluem `langchain-openai`, `langchain-community`, `langchain-groq`, `chromadb`.
- Chroma: erro de dimensão/coleção
  - O projeto agora usa `collection_name` atrelado ao modelo de embedding para evitar conflitos. Se você trocou de embedding/modelo e ocorrer erro ao adicionar documentos, rode com `--rebuild` para recriar o índice ou apague `data/vector_store/` manualmente.

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

## Videos without transcript: fallback

- `--subs`: tries to download auto/manual subtitles via `yt-dlp` and converts VTT/SRT to text.

## Qualidade (opcional)

- Lint:

```bash
ruff check .
```

- Format:

```bash
ruff format .
```

- Type-check:

```bash
pyrefly .
```

## Notas

- Use apenas `pt` e `en` como idiomas de transcrição.
- `pathlib.Path` é usado para todos os caminhos; converta para `str` apenas quando bibliotecas exigirem.
