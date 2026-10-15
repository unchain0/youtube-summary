# Sub-features: Show download progress and chat while downloading

## How to get to it (user POV)
Streamlit dashboard shows download progress and allows chatting while transcripts are being downloaded (background thread).

## Driving it with the harness
- Launch the app: `streamlit run main.py`
- Doctor check: Streamlit server starts, page loads without error
- Drive: interact with Streamlit UI, initiate a download, observe progress updates
- Expected end state: Progress updates appear in the UI; chat is available during download
- Gotchas: Progress updates from background thread do not automatically rerun the Streamlit script; may need to manually refresh or interact with the UI to see updates

## Proof standards
- Exercise the real user path: start `streamlit run main.py`, initiate a download with `--limit 1`, observe progress
- Capture the action: progress updates appear in the Streamlit UI
- Verify side effects: check that transcript files are eventually created in `data/transcripts/`
- Mocks: none needed

## Sub-features
- [ ] Streamlit dashboard loads without error
- [ ] Download progress visible in UI during CLI-driven fetch
- [ ] Chat interface available during download