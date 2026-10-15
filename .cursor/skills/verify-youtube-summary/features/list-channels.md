# Sub-features: View downloaded YouTubers and transcript counts

## How to get to it (user POV)
Run `python main.py list-channels` or use the Streamlit dashboard sidebar.

## Driving it with the harness
- Launch the app: `python main.py list-channels`
- Doctor check: process lists channels without error
- Drive: invoke `python main.py list-channels`
- Expected end state: Output lists channel names and file counts
- Gotchas: Requires `data/transcripts/` directory to exist; empty directory returns empty list

## Proof standards
- Exercise the real user path: run `python main.py list-channels` after downloading transcripts
- Capture the action: terminal output lists channels and counts
- Verify side effects: check that counts match actual file count in `data/transcripts/`
- Mocks: none needed

## Sub-features
- [ ] List channels via CLI (`list-channels` command)
- [ ] View counts in Streamlit dashboard sidebar