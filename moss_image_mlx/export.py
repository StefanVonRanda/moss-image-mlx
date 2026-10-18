"""Export a MOSS-Image checkpoint to the MLX release layout, optionally quantized.

  moss-mlx-export --model ~/hf/MOSS-Image-SFT --dit-quant 8 --te-quant 4 --out ~/out/MOSS-Image-SFT-MLX-compact

Layout (each component config.json carries {"moss_image_mlx": {"format": 1, "quantization": spec | null}}):
  config.json            top-level summary (also what the Hub counts as a download for library "mlx")
  model_index.json       diffusers-style index of the components
  transformer/           MLX parameter names, sharded safetensors
  text_encoder/          Qwen3.5 language backbone only (vision tower dropped), norms stored as fp32 (1 + w)
  tokenizer/, scheduler/ copied
  vae/                   decoder only, MLX conv layout, fp32
Quantization runs on the CPU stream and is evaluated module by module, so the bf16 source never sits in memory whole.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .cli import quant_spec
from .dit import load_transformer
from .quantize import describe, quantize_dit, quantize_text_encoder
from .text_encoder import TextEncoder
from .vae import load_vae_decoder

FORMAT = 1
SHARD_BYTES = 4 << 30


def _eval_by_module(model):
    for _, module in model.named_modules():
        params = [v for _, v in tree_flatten(module.parameters())]
        if params:
            mx.eval(params)


def _save_sharded(params: dict, out: Path, prefix: str = "model"):
    out.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k, v in params.items():
        if cur and size + v.nbytes > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += v.nbytes
    shards.append(cur)
    index = {}
    for i, shard in enumerate(shards):
        name = f"{prefix}-{i + 1:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else f"{prefix}.safetensors"
        mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
        index.update({k: name for k in shard})
    if len(shards) > 1:
        total = sum(v.nbytes for v in params.values())
        (out / f"{prefix}.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                          "weight_map": index}, indent=1))


def _component_config(src: Path, spec, drop=()):
    config = json.loads(src.read_text())
    for key in drop:
        config.pop(key, None)
    config["moss_image_mlx"] = {"format": FORMAT, "quantization": spec}
    return config


def export(model_dir, out, transformer_dir=None, dit_spec=None, te_spec=None, source_id=None, tier=None):
    model_dir, out = Path(model_dir), Path(out)
    transformer_dir = Path(transformer_dir) if transformer_dir else model_dir / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.cpu)

    dit = load_transformer(transformer_dir)
    quantize_dit(dit, dit_spec)
    _eval_by_module(dit)
    _save_sharded(dict(tree_flatten(dit.parameters())), out / "transformer")
    cfg = _component_config(transformer_dir / "config.json", dit_spec)
    (out / "transformer" / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    del dit
    mx.clear_cache()

    te = TextEncoder(model_dir / "text_encoder", model_dir / "tokenizer")
    quantize_text_encoder(te.model, te_spec)
    _eval_by_module(te.model)
    _save_sharded(dict(tree_flatten(te.model.parameters())), out / "text_encoder")
    cfg = _component_config(model_dir / "text_encoder" / "config.json", te_spec, drop=("vision_config",))
    cfg["architectures"] = ["Qwen3_5TextModel"]
    (out / "text_encoder" / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    del te
    mx.clear_cache()

    vae = load_vae_decoder(model_dir / "vae")
    _save_sharded(dict(tree_flatten(vae.parameters())), out / "vae", prefix="decoder")
    cfg = _component_config(model_dir / "vae" / "config.json", None)
    cfg["include_encoder"] = False
    (out / "vae" / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for name in ("tokenizer", "scheduler"):
        (out / name).mkdir(exist_ok=True)
        for f in (model_dir / name).iterdir():
            if f.is_file() and f.suffix in (".json", ".jinja", ".txt", ".model"):
                shutil.copy2(f, out / name / f.name)
    index = json.loads((model_dir / "model_index.json").read_text())
    index["moss_image_mlx"] = {"format": FORMAT}
    (out / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    if (model_dir / "LICENSE").exists():
        shutil.copy2(model_dir / "LICENSE", out / "LICENSE")
    summary = {"model_type": "moss_image", "library_name": "moss-image-mlx", "format": FORMAT, "tier": tier,
               "source_model": source_id, "transformer_quantization": describe(dit_spec),
               "text_encoder_quantization": describe(te_spec),
               "components": ["transformer", "text_encoder", "tokenizer", "vae", "scheduler"]}
    (out / "config.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="moss-mlx-export", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--transformer")
    p.add_argument("--dit-quant")
    p.add_argument("--te-quant")
    p.add_argument("--source-id")
    p.add_argument("--tier")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print(export(a.model, a.out, a.transformer, quant_spec(a.dit_quant), quant_spec(a.te_quant), a.source_id, a.tier))


if __name__ == "__main__":
    main()
