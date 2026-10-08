"""
vc_realtime — Lightweight real-time CPU voice conversion package.

Public API:
    from vc_realtime import Encoder, KNNRetrieval, Decoder, StreamingInfer

Package layout follows PEP 621 (``src/`` layout). Install with::

    pip install -e ".[dev]"

The Protocol/ABC interfaces that lock the v1.0 Python ↔ v2.0 Rust swap
boundary live in :mod:`vc_realtime.interfaces`.
"""

from .decoder import Decoder
from .encoder import Encoder
from .knn_retrieval import KNNRetrieval
from .streaming import StreamingInfer

__all__ = ["Encoder", "Decoder", "KNNRetrieval", "StreamingInfer"]
__version__ = "0.1.0"
