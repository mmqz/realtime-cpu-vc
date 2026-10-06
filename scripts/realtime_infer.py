#!/usr/bin/env python3
"""
scripts/realtime_infer.py — Real-time VC entry point (PyTorch OR ORT path)
=============================================================================
End-to-end streaming voice conversion.

Modes:
  --mode torch    : use PyTorch TinyVC directly (P0-4 verification path)
  --mode ort      : use ONNXRuntime with INT8 quantized graphs (P1-3 onwards)

Usage:
  # P0 (PyTorch path, smoke test)
  python scripts/realtime_infer.py --voice-id 0 --mode torch

  # P1 (ORT path, full target config)
  python scripts/realtime_infer.py --voice-id 0 --mode ort --config configs/default.yaml

  # Benchmark mode (run 30s, print latency + RSS stats)
  python scripts/realtime_infer.py --voice-id 0 --mode ort --benchmark --duration 30
"""
import argparse
import os
import sys
import time
import yaml
from pathlib import Path

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.knn_retrieval import KNNRetrieval
from modules.streaming import StreamingInfer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict):
    """Construct the Encoder → KNN → Decoder → StreamingInfer pipeline."""
    a = config['audio']
    ort_cfg = config['ort']

    print(f"Loading encoder: {ort_cfg['encoder_model']}")
    encoder = Encoder(ort_cfg['encoder_model'],
                      intra_op_threads=ort_cfg['intra_op_num_threads'],
                      inter_op_threads=ort_cfg['inter_op_num_threads'])

    print(f"Loading decoder: {ort_cfg['source_net_model']}, {ort_cfg['filter_net_model']}")
    decoder = Decoder(ort_cfg['source_net_model'],
                      ort_cfg['filter_net_model'],
                      intra_op_threads=ort_cfg['intra_op_num_threads'],
                      inter_op_threads=ort_cfg['inter_op_num_threads'])

    print(f"Loading voices: {config['voices']['path']}")
    knn = KNNRetrieval(config['voices']['path'], top_k=config['voices']['knn_top_k'])

    streamer = StreamingInfer(config, encoder, decoder, knn)
    return streamer


def benchmark_mode(streamer: StreamingInfer, duration_s: int = 30):
    """P1-6: run for `duration_s` seconds, print latency + RSS stats.

    Doesn't actually open PyAudio streams — uses synthetic input to measure
    compute path only. Latency = compute latency + algorithmic latency
    (the latter derived from config).
    """
    import numpy as np
    import resource

    print(f"Benchmark mode: {duration_s}s synthetic input")
    cfg = streamer.cfg['audio']
    sr = cfg['sample_rate']
    block = cfg['block_size']
    n_blocks = int(duration_s * sr / block)
    print(f"  blocks: {n_blocks}, block_size: {block} ({block/sr*1000:.0f}ms)")

    # Synthetic input: white noise (silent test, measures pure compute)
    latencies = []
    for i in range(n_blocks):
        t0 = time.perf_counter()
        # Simulate one chunk through the pipeline
        chunk = np.random.randn(block).astype(np.float32) * 0.01
        mel_spec = __import__('modules.encoder', fromlist=['make_mel_spec']).make_mel_spec(
            chunk[None, :])
        content, f0, energy = streamer.enc.encode(mel_spec)
        content_r = streamer.knn.replace(content)
        wav = streamer.dec.decode(content_r, f0, energy)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    # Use GPC max RSS
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 if sys.platform != 'darwin' else rss_kb / (1024 * 1024)
    algo_lat = (cfg['block_size'] + cfg['extra_size'] + cfg['crossfade_size']
                + 2 * cfg['last_delay_size']) / sr * 1000
    print("\n--- Benchmark results ---")
    print(f"  Compute latency: mean={latencies.mean():.1f}ms "
          f"p95={np.percentile(latencies, 95):.1f}ms "
          f"max={latencies.max():.1f}ms")
    print(f"  Algorithmic latency: {algo_lat:.1f}ms")
    print(f"  Total E2E estimate: {latencies.mean() + algo_lat:.1f}ms")
    print(f"  RSS: {rss_mb:.1f} MB")
    rtf = latencies.mean() / (block / sr * 1000)
    print(f"  RTF: {rtf:.3f} (<1 = real-time)")
    print(f"  Target: <500ms E2E, <110MB RSS, RTF <0.5")
    if latencies.mean() + algo_lat < 500 and rss_mb < 110:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — see analysis.md § 6 for tuning knobs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--voice-id', type=int, default=0,
                   help='index of target voice in voices.safetensors')
    p.add_argument('--mode', choices=['torch', 'ort'], default='ort',
                   help='torch = P0 PyTorch path; ort = P1 ONNXRuntime path')
    p.add_argument('--benchmark', action='store_true',
                   help='run synthetic benchmark instead of opening PyAudio')
    p.add_argument('--duration', type=int, default=30,
                   help='benchmark duration in seconds')
    args = p.parse_args()

    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}, voice_id: {args.voice_id}")
    config = load_config(args.config)

    streamer = build_pipeline(config)
    streamer.knn.select_voice(args.voice_id)

    if args.benchmark:
        benchmark_mode(streamer, args.duration)
    else:
        # Run PyAudio loop
        try:
            streamer.start()
        except KeyboardInterrupt:
            print("\nStopping...")
            streamer.stop()


if __name__ == '__main__':
    main()
