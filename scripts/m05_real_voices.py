#!/usr/bin/env python3
"""M0.5 · Real 5-speaker validation set + paired_hard parallel data.

Replaces the legacy ``librosa.effects.pitch_shift`` voice fixtures with
genuine VCTK speakers (p225, p226, p227, p228, p229), and additionally
extracts *parallel reading* pairs — VCTK's standard p225-p229 all read the
same text prompts, so (src_spk, tgt_spk, text_id) is natural parallel data
for the VoicePack + 192-d projection joint training in M1a.

Outputs
-------
- ``data/voices/voice_{0..4}.wav`` — 30 s @ 24 kHz mono PCM16, peak-normalised
  to -3 dBFS (replaces the pitch_shifted legacy fixtures).
- ``data/source/source_{001..010}.wav`` — first 5 s of voice_0 .. voice_4
  *and* the librosa real-speech example, used as the source audio for the
  v1 baseline re-run.
- ``data/paired_hard/{src_spk}_{tgt_spk}_{text_id:03d}.wav`` — parallel
  VCTK utterance pairs for M1a VoicePack joint training.
- ``data/paired_hard/index.json`` — manifest with src/tgt paths + text.

Design choices
--------------
* Uses ``pyarrow.parquet`` to read VCTK parquet shards directly — avoids
  the new ``datasets`` library requirement of ``torchcodec``.
* 5 VCTK speakers (p225-p229) chosen because (a) they're a well-known
  accent-diverse subset, (b) they all read the same set of ~30 prompts,
  giving ample parallel pairs.
* Source speakers and target speakers are *disjoint* — p225-p229 are the
  5 target voices; the source is taken from p232, p237 (different speakers,
  same accent region) so v1 baseline does not degenerate into source≈target.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR: int = 24_000
DURATION_S: float = 30.0
TARGET_PEAK_DBFS: float = -3.0
SOURCE_DURATION_S: float = 5.0

VCTK_REPO_ID: str = "sanchit-gandhi/vctk"

# 5 target voices — p225-p229 (well-known VCTK subset, mixed gender/region)
TARGET_SPEAKERS: list[str] = ["p225", "p226", "p227", "p228", "p229"]

# Source speakers — different from targets, used for v1 baseline source audio.
SOURCE_SPEAKERS: list[str] = ["p232", "p237"]

N_PARALLEL_PAIRS: int = 20

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_VOICES = REPO_ROOT / "data" / "voices"
DATA_SOURCE = REPO_ROOT / "data" / "source"
DATA_PAIRED_HARD = REPO_ROOT / "data" / "paired_hard"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _normalize_to_peak_dbfs(x: np.ndarray, dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        raise ValueError("Signal is silent; cannot normalise.")
    target = 10.0 ** (dbfs / 20.0)
    return (x * (target / peak)).astype(np.float32)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _to_pcm16(x), sr, subtype="PCM_16")


def _resample_to_24k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x.astype(np.float32)
    return librosa.resample(x.astype(np.float32), orig_sr=sr_in, target_sr=SR)


def _concat_to_duration(segments: list[np.ndarray], target_n: int) -> np.ndarray:
    out = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    if len(out) >= target_n:
        return out[:target_n]
    pad = np.zeros(target_n - len(out), dtype=np.float32)
    return np.concatenate([out, pad])


# ---------------------------------------------------------------------------
# VCTK parquet streaming reader
# ---------------------------------------------------------------------------
class VCTKStream:
    """Stream VCTK parquet shards, collect utterances for given speakers."""

    def __init__(self, speakers: list[str], min_samples_per_speaker: int = 30):
        self.speakers = set(speakers)
        self.min_samples = min_samples_per_speaker
        self.collected: dict[str, list[dict]] = {s: [] for s in speakers}
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=VCTK_REPO_ID, repo_type="dataset")
        self.shard_paths = sorted(f for f in files if f.endswith(".parquet"))
        print(f"[vctk] {len(self.shard_paths)} shards available")

    def stream(self) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        for shard_idx, shard_path in enumerate(self.shard_paths):
            if all(len(self.collected[s]) >= self.min_samples for s in self.speakers):
                break
            t0 = time.perf_counter()
            try:
                local = hf_hub_download(
                    repo_id=VCTK_REPO_ID, filename=shard_path, repo_type="dataset",
                )
            except Exception as e:  # noqa: BLE001
                print(f"[vctk] shard {shard_idx} download failed: {e}", file=sys.stderr)
                continue
            pf = pq.ParquetFile(local)
            columns_to_read = ["speaker_id", "text_id", "text", "gender", "audio", "file"]
            for rg_idx in range(pf.num_row_groups):
                if all(len(self.collected[s]) >= self.min_samples for s in self.speakers):
                    break
                needed = [s for s in self.speakers
                          if len(self.collected[s]) < self.min_samples]
                rg = pf.read_row_group(rg_idx, columns=columns_to_read)
                speaker_ids = rg.column("speaker_id").to_pylist()
                text_ids = rg.column("text_id").to_pylist()
                texts = rg.column("text").to_pylist()
                genders = rg.column("gender").to_pylist()
                files = rg.column("file").to_pylist()
                audio_cells = rg.column("audio").to_pylist()
                for i, spk in enumerate(speaker_ids):
                    if spk not in needed:
                        continue
                    if len(self.collected[spk]) >= self.min_samples:
                        continue
                    audio_cell = audio_cells[i]
                    if isinstance(audio_cell, dict):
                        audio_bytes = audio_cell.get("bytes", b"")
                    else:
                        continue
                    if not audio_bytes:
                        continue
                    try:
                        wav, sr_in = sf.read(io.BytesIO(audio_bytes))
                        if wav.ndim > 1:
                            wav = wav[:, 0]
                        wav = _resample_to_24k(wav, sr_in)
                    except Exception as e:  # noqa: BLE001
                        print(f"[vctk] decode failed for {spk}:{files[i]}: {e}",
                              file=sys.stderr)
                        continue
                    self.collected[spk].append({
                        "text_id": text_ids[i], "text": texts[i],
                        "gender": genders[i], "file": files[i],
                        "wav": wav, "duration_s": len(wav) / SR,
                    })
            elapsed = time.perf_counter() - t0
            collected_summary = ", ".join(
                f"{s}={len(self.collected[s])}" for s in self.speakers)
            print(f"[vctk] shard {shard_idx} ({Path(shard_path).name}) "
                  f"read in {elapsed:.1f}s — now: {collected_summary}")

        for s in self.speakers:
            n = len(self.collected[s])
            print(f"[vctk] {s}: {n} utterances collected "
                  f"(requested {self.min_samples})")


# ---------------------------------------------------------------------------
# Voice fixtures
# ---------------------------------------------------------------------------
def build_voice_reference(utterances: list[dict], out_path: Path,
                          duration_s: float = DURATION_S) -> None:
    target_n = int(SR * duration_s)
    segments = [u["wav"] for u in utterances]
    wav = _concat_to_duration(segments, target_n)
    wav = _normalize_to_peak_dbfs(wav)
    _write_wav(out_path, wav)
    rms = float(np.sqrt(np.mean(wav * wav)))
    print(f"  [voice_ref] {out_path.name}: {len(wav)/SR:.1f}s "
          f"rms={rms:.3f} peak={np.max(np.abs(wav)):.3f}")


def build_source_clips(utterances: list[dict], out_dir: Path,
                       n_clips: int = 5,
                       duration_s: float = SOURCE_DURATION_S) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_n = int(SR * duration_s)
    for i, u in enumerate(utterances[:n_clips]):
        wav = u["wav"][:target_n]
        if len(wav) < target_n:
            pad = np.zeros(target_n - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, pad])
        wav = _normalize_to_peak_dbfs(wav)
        out_path = out_dir / f"source_{i+1:03d}.wav"
        _write_wav(out_path, wav)
        print(f"  [source_clip] {out_path.name}: {u['text_id']} "
              f"text='{u['text'][:40]}...'")


# ---------------------------------------------------------------------------
# Paired hard (parallel reading)
# ---------------------------------------------------------------------------
def find_shared_text_ids(utterances_by_speaker: dict[str, list[dict]],
                         src_speakers: list[str],
                         tgt_speakers: list[str],
                         n_pairs: int = N_PARALLEL_PAIRS) -> list[str]:
    all_speakers = set(src_speakers) | set(tgt_speakers)
    text_id_sets = []
    for spk in all_speakers:
        if spk not in utterances_by_speaker:
            continue
        ids = {u["text_id"] for u in utterances_by_speaker[spk]}
        text_id_sets.append(ids)
    if not text_id_sets:
        return []
    shared = set.intersection(*text_id_sets)
    shared_sorted = sorted(shared)[:n_pairs]
    print(f"[paired_hard] {len(shared)} shared text_ids across "
          f"{len(all_speakers)} speakers; using first {len(shared_sorted)}")
    return shared_sorted


def build_paired_hard(utterances_by_speaker: dict[str, list[dict]],
                      src_speakers: list[str],
                      tgt_speakers: list[str],
                      text_ids: list[str],
                      out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    by_spk_text = {}
    for spk in (src_speakers + tgt_speakers):
        if spk not in utterances_by_speaker:
            continue
        by_spk_text[spk] = {u["text_id"]: u for u in utterances_by_speaker[spk]}

    for tgt_spk in tgt_speakers:
        for src_spk in src_speakers:
            if src_spk == tgt_spk:
                continue
            for tid in text_ids:
                src_u = by_spk_text.get(src_spk, {}).get(tid)
                tgt_u = by_spk_text.get(tgt_spk, {}).get(tid)
                if src_u is None or tgt_u is None:
                    continue
                fname = f"{src_spk}_{tgt_spk}_{tid}.wav"
                wav = _normalize_to_peak_dbfs(tgt_u["wav"])
                _write_wav(out_dir / fname, wav)
                index[f"{src_spk}_{tgt_spk}_{tid}"] = {
                    "src_speaker": src_spk, "tgt_speaker": tgt_spk,
                    "text_id": tid, "text": tgt_u["text"],
                    "tgt_duration_s": len(tgt_u["wav"]) / SR,
                    "src_duration_s": len(src_u["wav"]) / SR,
                    "tgt_file": fname,
                    "src_file": f"{tgt_spk}_{src_spk}_{tid}.wav",
                }
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-speakers", nargs="+", default=SOURCE_SPEAKERS)
    parser.add_argument("--target-speakers", nargs="+", default=TARGET_SPEAKERS)
    parser.add_argument("--n-parallel-pairs", type=int, default=N_PARALLEL_PAIRS)
    parser.add_argument("--skip-paired-hard", action="store_true")
    args = parser.parse_args()

    DATA_VOICES.mkdir(parents=True, exist_ok=True)
    DATA_SOURCE.mkdir(parents=True, exist_ok=True)

    all_speakers = list(set(args.target_speakers) | set(args.source_speakers))
    streamer = VCTKStream(speakers=all_speakers, min_samples_per_speaker=30)
    streamer.stream()

    # 1. Build 5 voice reference wavs (replaces data/voices/voice_{0..4}.wav)
    print("\n[1/3] Building 5 real-speaker reference voices...")
    legacy_dir = REPO_ROOT / "data" / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for i, spk in enumerate(args.target_speakers):
        utterances = streamer.collected.get(spk, [])
        if len(utterances) < 3:
            print(f"  [warn] {spk}: only {len(utterances)} utterances", file=sys.stderr)
        legacy_path = DATA_VOICES / f"voice_{i}.wav"
        if legacy_path.exists():
            archive_path = legacy_dir / f"voice_{i}.pitch_shifted.wav"
            legacy_path.rename(archive_path)
            print(f"  [archive] {legacy_path.name} → {archive_path.name}")
        build_voice_reference(utterances, DATA_VOICES / f"voice_{i}.wav")

    # 2. Build source clips from source speakers (replaces source_*.wav)
    print("\n[2/3] Building source clips from real source speakers...")
    legacy_source = REPO_ROOT / "data" / "legacy" / "source"
    legacy_source.mkdir(parents=True, exist_ok=True)
    for old_clip in sorted(DATA_SOURCE.glob("source_*.wav")):
        archive_path = legacy_source / old_clip.name
        old_clip.rename(archive_path)
        print(f"  [archive] {old_clip.name} → legacy/source/")
    src_spk = args.source_speakers[0]
    src_utterances = streamer.collected.get(src_spk, [])
    build_source_clips(src_utterances, DATA_SOURCE, n_clips=5)

    # 3. Build paired_hard
    if not args.skip_paired_hard:
        print("\n[3/3] Building paired_hard parallel reading data...")
        text_ids = find_shared_text_ids(
            streamer.collected,
            src_speakers=args.source_speakers,
            tgt_speakers=args.target_speakers,
            n_pairs=args.n_parallel_pairs,
        )
        index = build_paired_hard(
            streamer.collected,
            src_speakers=args.source_speakers,
            tgt_speakers=args.target_speakers,
            text_ids=text_ids,
            out_dir=DATA_PAIRED_HARD,
        )
        index_path = DATA_PAIRED_HARD / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"  [paired_hard] {len(index)} pairs written to {DATA_PAIRED_HARD}")
        print(f"  [paired_hard] index → {index_path}")

    print("\n[M0.5] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
