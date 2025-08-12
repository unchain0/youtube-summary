"""Pytest configuration to ensure src-layout imports work in tests.

Adds the project root and the `src/` directory to `sys.path` so modules can be
imported both as `src.*` and `utils.*` during test runs.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so the 'src' package can be imported during tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also add the 'src' directory itself so imports like 'utils.*' work in a src-layout
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
