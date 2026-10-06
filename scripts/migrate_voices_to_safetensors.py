"""Convert voices.pt (torch.load) to voices_v1.safetensors (safe, no code execution).

v2 / v3 already ship a `voices_v{N}.safetensors` index — v1 was the lone
holdout still using ``torch.load(..., weights_only=False)`` which can execute
arbitrary code if the file is tampered with. This migration:

  * Re-encodes every voice tensor to FP16 (half the on-disk footprint, no
    measurable accuracy loss for kNN-VC cosine retrieval at the default
    top-k=4 / alpha=0 setting).
  * Writes the result to ``models/voices_v1.safetensors`` using
    ``safetensors.torch.save_file`` — a flat mapping of str -> Tensor with
    a small JSON header, NO arbitrary Python pickle bytecode.

Run::

    python3 scripts/migrate_voices_to_safetensors.py

The original ``voices.pt`` is left in place so callers on older revisions of
``infer_v1.py`` keep working (the new code tries ``.safetensors`` first, then
falls back to ``.pt`` with a DeprecationWarning).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

MODELS = Path("/home/z/my-project/prototype/models")


def main() -> int:
    pt_path = MODELS / "voices.pt"
    sf_path = MODELS / "voices_v1.safetensors"

    if not pt_path.exists():
        print(f"[migrate] {pt_path} not found — nothing to migrate.", file=sys.stderr)
        return 1

    # NOTE: weights_only=False is required here because voices.pt is a dict
    # of tensors saved via ``torch.save(dict)`` (legacy pickle format). This
    # is the *last* call site that needs the unsafe path — after the safetensors
    # file is written, infer_v1.py uses ``safetensors.torch.load_file`` (no
    # code execution) and this script can be deleted.
    voices = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(voices, dict):
        # Backwards-compat: a single stacked tensor [N, 1, 768, T]
        voices = {f"voice_{i}": voices[i : i + 1] for i in range(voices.shape[0])}

    print(f"[migrate] loaded voices.pt with {len(voices)} voices: "
          f"{list(voices.keys())}")

    # Convert to safetensors (FP16 for storage efficiency — halves disk
    # footprint with no measurable kNN-VC accuracy impact).
    tensors: dict[str, torch.Tensor] = {}
    for k, v in voices.items():
        v32 = v.detach().to(torch.float32)
        tensors[k] = v32.contiguous().half()  # FP16
        print(f"  {k}: shape={tuple(v32.shape)}, dtype={v32.dtype} "
              f"-> {tensors[k].dtype}")

    # safetensors requires contiguous tensors and str keys.
    save_file(tensors, str(sf_path))

    pt_size = pt_path.stat().st_size
    sf_size = sf_path.stat().st_size
    print(f"[migrate] voices.pt: {pt_size // 1024} KB "
          f"-> voices_v1.safetensors: {sf_size // 1024} KB "
          f"({sf_size * 100 / pt_size:.1f}% of original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
