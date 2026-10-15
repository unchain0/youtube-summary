# Sub-features: Query the vector store (chat)

## How to get to it (user POV)
Run `python main.py --query "your question here"` or use the Streamlit dashboard chat interface.

## Driving it with the harness
- Launch the app: `python main.py --query "your question here"`
- Doctor check: process is ready to accept query input
- Drive: invoke `python main.py --query "your question here"` or interact with Streamlit chat
- Expected end state: Groq model returns a response based on the vector store context
- Gotchas: Requires both `TOGETHER_API_KEY` and `GROQ_API_KEY`; query without prior index build returns no context

## Proof standards
- Exercise the real user path: run `python main.py --query "test question"` after building the vector store
- Capture the action: Groq response is returned in terminal
- Verify side effects: check that response relates to the indexed video content
- Mocks: API keys are production boundaries; mock only if a production boundary already isolates the external system

## Sub-features
- [ ] Query via CLI (`--query` flag)
- [ ] Query via Streamlit dashboard chat