"""On-disk cache of prompt embeddings.

Re-rendering a prompt (new seed, steps, size or guidance) then skips the 9B text encoder entirely: no ~16 GB (bf16)
load and no encode. Entries are keyed by the encoder's identity (config, weight files, quantization, the
implementation's numerics version), the token limit, the exact prompt text and, for prompts with images (the
ai-toolkit edit format), the images' pixels as the vision tower receives them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

# Bump when the encoder numerics change, so stale embeddings are never reused.
ENCODER_NUMERICS_VERSION = "transformers-faithful-1"
VISION_NUMERICS_VERSION = "qwen3_5-vision-1"  # the same for the vision path (part of keys with images only)


def default_cache_dir() -> Path:
    return Path(os.environ.get("MOSS_MLX_CACHE", Path.home() / ".cache" / "moss-image-mlx")) / "prompt-embeds"


def encoder_id(text_encoder_dir: str | Path) -> str:
    d = Path(text_encoder_dir)
    h = hashlib.sha256(ENCODER_NUMERICS_VERSION.encode())
    h.update((d / "config.json").read_bytes())
    for f in sorted(d.glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return h.hexdigest()[:16]


class PromptCache:
    def __init__(self, text_encoder_dir: str | Path, root: str | Path | None = None, tag: str = "bf16"):
        """tag distinguishes encoder variants built from the same files (e.g. quantized at load)."""
        self.dir = Path(root or default_cache_dir()) / f"{encoder_id(text_encoder_dir)}-{tag}"

    def _path(self, prompt: str, max_tokens: int, images=()) -> Path:
        fields = [prompt, max_tokens]
        if len(images):  # text-only keys stay as they were
            fields.append([VISION_NUMERICS_VERSION] + [
                f"{a.shape}:{hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}" for a in map(np.asarray, images)])
        key = hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()
        return self.dir / f"{key}.safetensors"

    def get(self, prompt: str, max_tokens: int, images=()) -> mx.array | None:
        path = self._path(prompt, max_tokens, images)
        if not path.exists():
            return None
        try:
            return mx.load(str(path))["hidden"]
        except Exception:
            return None

    def put(self, prompt: str, max_tokens: int, hidden: mx.array, images=()):
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(prompt, max_tokens, images)
        tmp = path.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), {"hidden": hidden})
        tmp.replace(path)
