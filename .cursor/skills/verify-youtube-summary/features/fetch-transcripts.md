# Sub-features: Fetch transcripts for YouTube channels

## How to get to it (user POV)
Run `python main.py <channel_url> [--limit N] [--languages pt,en] [--subs]` to fetch transcripts for a YouTube channel.

## Driving it with the harness
- Launch the app: `python main.py <channel_url> [--limit N] [--languages pt,en] [--subs]`
- Doctor check: process exits with code 0, transcript files appear in `data/transcripts/<channel>/`
- Drive: invoke `python main.py` with channel URL and flags
- Expected end state: `.txt` files created under `data/transcripts/<channel>/` for each video
- Gotchas: `--subs` flag attempts to download subtitles via yt-dlp when no native transcript exists; this may fail if yt-dlp or proxy is misconfigured. Without `--subs`, videos without captions are silently skipped.

## Proof standards
- Exercise the real user path: run `python main.py` with a real channel URL and `--limit 1`
- Capture the action: transcript `.txt` files appear in `data/transcripts/<channel>/`
- Verify side effects: check file timestamps and content headers
- Mocks: none needed; `yt-dlp` and `youtube-transcript-api` are production boundaries

## Sub-features
- [ ] Fetch with `--limit 1` (single video)
- [ ] Fetch with `--limit 5` (multiple videos)
- [ ] Fetch with `--languages pt,en` (language preference)
- [ ] Fetch with `--subs` (subtitle fallback via yt-dlp)