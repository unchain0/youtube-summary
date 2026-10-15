# Sub-features: Build/Update the vector store

## How to get to it (user POV)
Run `python main.py --rebuild` or `python main.py --incremental` to index transcripts into the Chroma vector store.

## Driving it with the harness
- Launch the app: `python main.py --rebuild` or `python main.py --incremental`
- Doctor check: process exits with code 0, vector store directory updates
- Drive: invoke `python main.py` with --rebuild or --incremental flag
- Expected end state: `data/vector_store/` contains updated Chroma index with new/embedded documents
- Gotchas: `--rebuild` destroys existing Chroma data and rebuilds from all `.txt` files; `--incremental` only adds new files. Both require `TOGETHER_API_KEY` and `GROQ_API_KEY` to be set.

## Proof standards
- Exercise the real user path: run `python main.py --rebuild` after fetching transcripts
- Capture the action: Chroma index is updated/created at `data/vector_store/`
- Verify side effects: check Chroma collection document count, embedding dimensions
- Mocks: none needed; API keys are production boundaries

## Sub-features
- [ ] Rebuild vector store from all transcripts (`--rebuild`)
- [ ] Incrementally add new transcripts (`--incremental`)