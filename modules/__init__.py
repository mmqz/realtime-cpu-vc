"""
modules/__init__.py — Lightweight real-time VC prototype
=================================================================
Public API:
    from modules import Encoder, KNNRetrieval, Decoder, StreamingInfer
"""
from .encoder import Encoder
from .decoder import Decoder
from .knn_retrieval import KNNRetrieval
from .streaming import StreamingInfer

__all__ = ['Encoder', 'Decoder', 'KNNRetrieval', 'StreamingInfer']
__version__ = '0.1.0'
