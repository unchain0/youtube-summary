"""CLI entrypoint wiring for the YouTube transcript summarizer."""

from dotenv import load_dotenv

from src.cli import main as cli_main


def run() -> None:
    """Load environment variables and invoke the CLI main entrypoint."""
    load_dotenv()
    cli_main()


if __name__ == "__main__":
    run()
