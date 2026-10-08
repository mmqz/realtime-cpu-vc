"""
vc_realtime.cli — `vc-infer` console-script entry point.

This wraps :mod:`scripts.realtime_infer` so that, after ``pip install -e .``
(or ``pip install vc-realtime``), users can run::

    vc-infer --voice-id 0 --config configs/default.yaml

without manually invoking the ``scripts/realtime_infer.py`` path. The logic
is intentionally thin — the heavy lifting (config load, pipeline build,
benchmark loop, PyAudio start) lives in
:mod:`scripts.realtime_infer`, so both code paths stay in sync.

Why a separate module (and not ``from scripts.realtime_infer import main``)?
The ``scripts/`` directory is not a Python package (no ``__init__.py``); it is
intentionally kept as loose CLI scripts so contributors can read / copy them
without installing the project. ``vc_realtime.cli`` therefore re-implements
the (small) argparse surface and delegates to the same builder functions.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _repo_root() -> Path:
    """Return the path to the repo root (for default config resolution)."""
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(name: str) -> str:
    """Resolve a config name (e.g. ``default``) to a path under ``configs/``."""
    p = Path(name)
    if p.is_file():
        return str(p)
    # try relative to repo root
    candidate = _repo_root() / "configs" / name
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"Config '{name}' not found. Pass an absolute path or a name under "
        f"configs/ (e.g. 'default', 'v2_hybrid', 'v3_hybrid')."
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the `vc-infer` CLI."""
    p = argparse.ArgumentParser(
        prog="vc-infer",
        description=(
            "Real-time CPU voice conversion (v3 hybrid: TinyVC encoder + "
            "Spark BiCodec + OpenVoice flow + F5-TTS Vocos)."
        ),
    )
    p.add_argument(
        "--config",
        default="default",
        help="Name of config under configs/ (e.g. 'default', 'v2_hybrid', "
        "'v3_hybrid') or an absolute path to a YAML file.",
    )
    p.add_argument(
        "--voice-id",
        type=int,
        default=0,
        help="Index of the target voice in the loaded voice registry.",
    )
    p.add_argument(
        "--mode",
        choices=["torch", "ort"],
        default="ort",
        help="torch = P0 PyTorch path (smoke test); ort = P1 ONNXRuntime path.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a synthetic-input benchmark instead of opening PyAudio.",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Benchmark duration in seconds (only with --benchmark).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point. Returns a Unix exit code."""
    args = build_parser().parse_args(argv)
    config_path = _resolve_config_path(args.config)
    print(f"[vc-infer] config: {config_path}")
    print(f"[vc-infer] mode={args.mode} voice_id={args.voice_id}")

    # Import here so `vc-infer --help` works even if optional deps are missing.
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Lazy import of the heavy pipeline modules so users without audio
    # hardware can still run `vc-infer --benchmark`.
    from vc_realtime.decoder import Decoder
    from vc_realtime.encoder import Encoder
    from vc_realtime.knn_retrieval import KNNRetrieval
    from vc_realtime.streaming import StreamingInfer

    ort_cfg = config["ort"]
    print(f"[vc-infer] loading encoder: {ort_cfg['encoder_model']}")
    encoder = Encoder(
        ort_cfg["encoder_model"],
        intra_op_threads=ort_cfg["intra_op_num_threads"],
        inter_op_threads=ort_cfg["inter_op_num_threads"],
    )
    print(
        f"[vc-infer] loading decoder: {ort_cfg['source_net_model']}, {ort_cfg['filter_net_model']}"
    )
    decoder = Decoder(
        ort_cfg["source_net_model"],
        ort_cfg["filter_net_model"],
        intra_op_threads=ort_cfg["intra_op_num_threads"],
        inter_op_threads=ort_cfg["inter_op_num_threads"],
    )
    print(f"[vc-infer] loading voices: {config['voices']['path']}")
    knn = KNNRetrieval(config["voices"]["path"], top_k=config["voices"]["knn_top_k"])
    streamer = StreamingInfer(config, encoder, decoder, knn)
    streamer.knn.select_voice(args.voice_id)

    if args.benchmark:
        _run_benchmark(streamer, duration_s=args.duration)
    else:
        try:
            streamer.start()
        except KeyboardInterrupt:
            print("\n[vc-infer] stopping...")
        finally:
            streamer.stop()
    return 0


def _run_benchmark(streamer, duration_s: int = 30) -> None:
    """Headless benchmark: synthetic input → compute latency + RSS."""
    import resource
    import time

    import numpy as np

    cfg = streamer.cfg["audio"]
    sr = cfg["sample_rate"]
    block = cfg["block_size"]
    n_blocks = int(duration_s * sr / block)
    print(
        f"[vc-infer] benchmark: {duration_s}s, {n_blocks} blocks "
        f"of {block} samples ({block / sr * 1000:.0f}ms)"
    )

    from vc_realtime.encoder import make_mel_spec

    latencies = []
    for _ in range(n_blocks):
        t0 = time.perf_counter()
        chunk = np.random.randn(block).astype(np.float32) * 0.01
        mel_spec = make_mel_spec(chunk[None, :])
        content, f0, energy = streamer.enc.encode(mel_spec)
        content_r = streamer.knn.replace(content)
        streamer.dec.decode(content_r, f0, energy)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 if sys.platform != "darwin" else rss_kb / (1024 * 1024)
    algo_lat = (
        (cfg["block_size"] + cfg["extra_size"] + cfg["crossfade_size"] + 2 * cfg["last_delay_size"])
        / sr
        * 1000
    )
    print("\n--- vc-infer benchmark results ---")
    print(
        f"  compute latency : mean={latencies.mean():.1f}ms "
        f"p95={np.percentile(latencies, 95):.1f}ms "
        f"max={latencies.max():.1f}ms"
    )
    print(f"  algorithmic lat: {algo_lat:.1f}ms")
    print(f"  total E2E est  : {latencies.mean() + algo_lat:.1f}ms")
    print(f"  RSS            : {rss_mb:.1f} MB")
    rtf = latencies.mean() / (block / sr * 1000)
    print(f"  RTF            : {rtf:.3f}  (<1 = real-time)")
    print("  target         : <500ms E2E, <110MB RSS, RTF <0.5")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
