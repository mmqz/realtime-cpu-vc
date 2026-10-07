#!/usr/bin/env python3
"""M1a prep · Patch the missing source-side wav files in data/paired_hard/.

The original m05_real_voices.py wrote only the target-side wav per pair
(file `p232_p225_001.wav` contains p225's audio — the target). But it
referenced `src_file: p225_p232_001.wav` in the index, which would need
to contain p232's audio of the same text — but that file was never
written because the inner loop didn't iterate the reverse direction.

This script re-streams the same VCTK shards (already cached locally by
huggingface_hub), finds the source speaker's utterance for each text_id
referenced in data/paired_hard/index.json, and writes the missing
source-side wav as `data/paired_hard/{tgt_spk}_{src_spk}_{text_id}.wav`
(so the file name matches index.json's `src_file`).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from m05_real_voices import (  # type: ignore
    VCTK_REPO_ID, VCTKStream, _normalize_to_peak_dbfs, _to_pcm16, _write_wav,
    _resample_to_24k, SR, TARGET_SPEAKERS, SOURCE_SPEAKERS,
)

INDEX_PATH = REPO_ROOT / "data" / "paired_hard" / "index.json"
OUT_DIR = REPO_ROOT / "data" / "paired_hard"


def main() -> int:
    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} missing; run m05_real_voices.py first",
              file=sys.stderr)
        return 1
    with open(INDEX_PATH) as f:
        index = json.load(f)
    print(f"[patch] {len(index)} pairs in index")

    # Find which (src_spk, text_id) combinations we need
    needed: dict[str, set[str]] = {}  # src_spk -> set(text_id)
    for entry in index.values():
        src_spk = entry["src_speaker"]
        tid = entry["text_id"]
        needed.setdefault(src_spk, set()).add(tid)
    all_needed_speakers = list(needed.keys())
    print(f"[patch] need source wavs for: {all_needed_speakers}")
    for spk, tids in needed.items():
        print(f"  {spk}: {len(tids)} text_ids")

    # Stream VCTK to collect these speakers' utterances for the needed text_ids
    streamer = VCTKStream(
        speakers=all_needed_speakers,
        min_samples_per_speaker=max(len(tids) for tids in needed.values()) + 5,
    )
    streamer.stream()

    # Build a lookup: (src_spk, text_id) -> wav
    src_lookup: dict[tuple[str, str], np.ndarray] = {}
    for spk in all_needed_speakers:
        for u in streamer.collected.get(spk, []):
            tid = u["text_id"]
            if tid in needed[spk]:
                src_lookup[(spk, tid)] = u["wav"]

    # Write the missing source-side wavs
    written = 0
    missing = 0
    for key, entry in index.items():
        src_spk = entry["src_speaker"]
        tgt_spk = entry["tgt_speaker"]
        tid = entry["text_id"]
        src_file_name = entry["src_file"]  # e.g. "p225_p232_001.wav"
        src_path = OUT_DIR / src_file_name
        if src_path.exists():
            continue  # already there
        wav = src_lookup.get((src_spk, tid))
        if wav is None:
            missing += 1
            continue
        wav_norm = _normalize_to_peak_dbfs(wav)
        _write_wav(src_path, wav_norm)
        written += 1
    print(f"\n[patch] wrote {written} missing source wavs; {missing} not found "
          f"in VCTK shards (text_id may be missing for that speaker)")
    print(f"[patch] paired_hard/ now ready for M1a training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
