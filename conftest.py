"""
conftest.py — pytest bootstrap for the src/ layout.

Adds `prototype/src` to sys.path so `import vc_realtime.*` works without a
pip install. Also exposes shared session-scoped fixtures.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
