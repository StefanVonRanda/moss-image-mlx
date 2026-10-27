"""MOSS-Image text-to-image pipeline in MLX.

Follows the official MOSSImagePipeline.__call__: fp32 starting noise from torch's CPU generator (so official seeds
reproduce), sigmas snapped to the training grid, CFG and Euler updates in fp32, block skipping in the negative
branch, optional skip-layer guidance, fp32 VAE decode.

Memory lifecycle (low_memory=True): text encoder -> encode positive and negative -> free; transformer -> denoise ->
free; VAE decoder -> decode. Only one large component is resident at a time.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .dit import EDIT_FORMATS, load_transformer
from .lora import set_strength
from .quantize import describe, quantize_dit, quantize_text_encoder
from .scheduler import sigmas_for, timesteps_for
from .text_encoder import TextEncoder, vision_prompt
from .torch_randn import randn
from .vae import load_vae_decoder, to_uint8


@dataclass
class Request:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance: float = 3.0
    shift: float = 5.0
    seed: int = 42
    uncond_skip_layers: list[int] = field(default_factory=lambda: [22])
    slg_layers: list[int] = field(default_factory=list)
    slg_scale: float = 0.0
    slg_start: float = 0.0
    slg_stop: float = 1.0
    max_sequence_length: int = 4096
    init_image: str | None = None   # image-to-image: start from this image (resized and centre-cropped)
    strength: float = 0.6           # fraction of the schedule to run from the init; 1.0 = plain text-to-image
    mask: str | None = None         # inpainting: white = regenerate, black = keep (needs init_image)
    mask_feather: int = 12          # px: grow the mask by this much, then soften its edge over the same distance
    preview: str | None = None      # live preview: rewrite this PNG each step with the current estimate (H/16 x W/16)
    references: list[str] = field(default_factory=list)  # instruction edit: images the prompt refers to (needs an edit LoRA)
    reference_size: int = 1024      # px: each reference is scaled to fit this longest side, keeping its aspect
    edit_format: str = "chinoll"    # the edit LoRA's layout (dit.EDIT_FORMATS); "ai-toolkit" also shows the text
    vision_pixels: int = 384 * 384  # encoder the references, each scaled down to at most this many pixels
    rgba: bool = False              # decode with the RGBA VAE: a transparent PNG (needs an alpha LoRA)
    keep_alpha: bool = False        # give the result the source's alpha (references[0], else init_image), resized
    # Colour/depth control (needs the pipeline's control adapter): a flat-colour region map and a depth map (near =
    # white), each resized and centre-cropped to the output size; either may be omitted. The maps act on the first
    # control_end of the steps (0.5 for synthetic maps, 1.0 for maps from real photos), then are zeroed.
    control_color: str | None = None
    control_depth: str | None = None
    control_end: float = 0.5
    # Per-step overrides (empty = off): an explicit sigma schedule ending in 0 (replaces steps/shift), and per-step
    # CFG scale and adapter strength, e.g. original model + CFG for the composition steps, then a turbo adapter.
    sigmas: list[float] = field(default_factory=list)
    step_guidance: list[float] = field(default_factory=list)
    step_lora: list[float] = field(default_factory=list)
    # Turbo sampler (turbo.py): needs the student and destripe adapters loaded; steps/shift/sigmas are ignored.
    turbo: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.sigmas) - 1 if self.sigmas else self.steps

    def uses_cfg(self) -> bool:
        return self.turbo or any(g != 1.0 for g in (self.step_guidance or [self.guidance]))

    def validate(self, depth: int):
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("width and height must be positive multiples of 16")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        if self.mask and not self.init_image:
            raise ValueError("mask needs init_image")
        if self.keep_alpha and not (self.init_image or self.references):
            raise ValueError("keep_alpha needs init_image or references")
        if self.turbo and (self.init_image or self.references or self.control_color or self.control_depth):
            raise ValueError("turbo runs its own schedule from noise; it takes no init_image, references or control")
        if not 0 <= self.control_end <= 1:
            raise ValueError("control_end must be in [0, 1]")
        if self.reference_size < 16:
            raise ValueError("reference_size must be at least 16")
        if self.edit_format not in EDIT_FORMATS:
            raise ValueError(f"edit_format must be one of {', '.join(EDIT_FORMATS)}")
        if self.vision_pixels < 32 * 32:
            raise ValueError("vision_pixels must be at least 1024 (32 x 32)")
        if not 0 <= self.slg_start <= self.slg_stop <= 1:
            raise ValueError("require 0 <= slg_start <= slg_stop <= 1")
        if self.sigmas and (self.sigmas[-1] != 0 or any(a <= b for a, b in zip(self.sigmas, self.sigmas[1:]))):
            raise ValueError("sigmas must decrease strictly and end at 0")
        for name in ("step_guidance", "step_lora"):
            if getattr(self, name) and len(getattr(self, name)) != self.n_steps:
                raise ValueError(f"{name} needs one value per step ({self.n_steps})")
        for layers in (self.uncond_skip_layers, self.slg_layers):
            if any(type(i) is not int or not 0 <= i < depth for i in layers):
                raise ValueError(f"skip layers must be integer block indices in [0, {depth})")


def _footprint() -> int | None:
    """This process's physical footprint in bytes (macOS proc_pid_rusage), or None if unavailable."""
    try:
        import ctypes
        import os

        buf = (ctypes.c_uint64 * 64)()
        if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.cast(buf, ctypes.c_void_p)):
            return None
        return int(buf[2 + 7])
    except Exception:
        return None


def _free(timeout: float = 3.0):
    """Release dropped MLX buffers and wait until the OS has actually reclaimed them.

    After MLX frees a large component, macOS lowers the process footprint asynchronously (~1 s). If the next stage
    allocates immediately, both are counted at once and the staged peak becomes their sum. Wait (bounded) for at
    least 80% of the released bytes to leave the footprint first."""
    held = mx.get_active_memory() + mx.get_cache_memory()
    start = _footprint()
    gc.collect()
    mx.clear_cache()
    released = held - (mx.get_active_memory() + mx.get_cache_memory())
    if start is None or released < (1 << 30):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        now = _footprint()
        if now is None or now <= start - 0.8 * released:
            return
        time.sleep(0.05)


def start_index(steps: int, strength: float) -> int:
    """Index of the first sigma to run for image-to-image: round((1 - strength) * steps), so 1.0 runs all steps
    and 0.0 runs none."""
    return min(int(round((1 - strength) * steps)), steps)


def latent_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Full-res 0/1 mask (H, W) -> soft (1, 1, H // 16, W // 16) by area averaging over each 16x16 patch."""
    m = mask.astype(np.float32).reshape(height // 16, 16, width // 16, 16).mean(axis=(1, 3))
    return m[None, None]


def _sample(model, req: Request, positive, negative, latents, on_step, init=None, mask=None, references=None,
            control=None):
    """init: packed latents of the init image (same shape as latents); mask: latent-resolution weights, 1 = regenerate;
    references: packed latents of the reference images (an edit); control: (1, 256, H, W) colour and depth latents,
    given to the model while i / n < req.control_end (both CFG branches, as trained), zeros after."""
    req.validate(len(model.layers))
    if req.turbo:
        from . import turbo

        return turbo.sample(model, req, positive, negative, latents, on_step)
    set_strength(model, 1.0, "default")  # undo a previous request's per-step strengths
    refs = model.reference_cache(references) if references else None  # once: both branches, every step
    fmt = req.edit_format
    shape = (1, model.config["latent_channels"], req.height // 16, req.width // 16)
    noise = mx.array(randn(shape, req.seed))
    sigmas = np.array(req.sigmas, dtype=np.float32) if req.sigmas else sigmas_for(req.steps, req.shift)
    timesteps = timesteps_for(sigmas)
    sig = [float(v) for v in sigmas]  # Python floats: numpy scalars do not broadcast against mx arrays
    first = start_index(len(timesteps), req.strength) if init is not None else 0
    if latents is None:
        latents = noise if init is None else (1 - sig[first]) * init + sig[first] * noise
    n = len(timesteps)
    for i, t in enumerate(timesteps[first:], start=first):
        skip_active = req.slg_start <= i / n < req.slg_stop
        guidance = req.step_guidance[i] if req.step_guidance else req.guidance
        do_cfg = guidance != 1.0 and negative is not None
        if req.step_lora:
            set_strength(model, req.step_lora[i], "default")
        ctl = control if i / n < req.control_end else None
        cond = model(latents, float(t), positive, references=refs, control=ctl, edit_format=fmt).astype(mx.float32)
        prediction = cond
        if do_cfg:
            skip = req.uncond_skip_layers if (req.uncond_skip_layers and skip_active) else ()
            uncond = model(latents, float(t), negative, skip, refs, ctl, fmt).astype(mx.float32)
            prediction = uncond + guidance * (cond - uncond)
        if req.slg_layers and req.slg_scale > 0 and skip_active:
            skipped = model(latents, float(t), positive, req.slg_layers, refs, ctl, fmt).astype(mx.float32)
            if not do_cfg:
                prediction = skipped
            prediction = prediction + req.slg_scale * (cond - skipped)
        dt = mx.array(sigmas[i + 1] - sigmas[i], dtype=mx.float32)  # float32, as torch computes it
        current = latents
        latents = latents.astype(mx.float32) + dt * prediction
        if mask is not None:  # keep the unmasked region on the init's trajectory at the next sigma
            latents = mask * latents + (1 - mask) * ((1 - sig[i + 1]) * init + sig[i + 1] * noise)
        mx.eval(latents)
        if req.preview:  # from arrays already evaluated; for inpainting this is the blended latents' estimate
            from .preview import write, x0_estimate

            write(x0_estimate(current.astype(mx.float32), sig[i], prediction), req.preview)
        if on_step:
            on_step(i - first, n - first)
    return latents


def flatten(image):
    """RGB PIL image; transparent pixels composited onto neutral grey. A plain convert("RGB") keeps whatever colour
    sits under alpha 0 (usually black), which the VAE then encodes as part of the picture."""
    from PIL import Image

    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info:
        image = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", image.size, (128, 128, 128, 255)), image).convert("RGB")
    return image.convert("RGB")


def load_image(path: str | Path, width: int, height: int) -> np.ndarray:
    """RGB uint8 (height, width, 3): resized to cover and centre-cropped, like diffusers' VaeImageProcessor fill.
    Transparency is flattened onto grey."""
    from PIL import Image, ImageOps

    return np.asarray(ImageOps.fit(flatten(Image.open(path)), (width, height), Image.LANCZOS))


def source_alpha(req: Request) -> np.ndarray:
    """uint8 (height, width): the alpha of the edit's source at the output size, placed as the source was: picture 1
    of an instruction edit stretched to the output (it keeps the source's aspect), an init image cropped like
    load_image. Fully opaque for an image without alpha."""
    from PIL import Image, ImageOps

    path = req.references[0] if req.references else req.init_image
    with Image.open(path) as image:
        if req.references:
            image = ImageOps.exif_transpose(image)  # as load_reference
        alpha = image.convert("RGBA").getchannel("A")
    size = (req.width, req.height)
    return np.asarray(alpha.resize(size, Image.LANCZOS) if req.references else ImageOps.fit(alpha, size, Image.LANCZOS))


def load_reference(path: str | Path, max_edge: int) -> np.ndarray:
    """RGB uint8 scaled to fit `max_edge` (never enlarged) with each side rounded to 16 px, no crop: as MOSS Image
    Edit prepares references. EXIF orientation applied, transparency flattened onto grey."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = flatten(ImageOps.exif_transpose(image))
    limit = max_edge // 16 * 16
    scale = min(1.0, limit / max(image.size))
    size = [min(limit, max(16, int(math.floor(v * scale / 16 + 0.5)) * 16)) for v in image.size]
    return np.asarray(image.resize(size, Image.LANCZOS))


def load_vision(path: str | Path, max_pixels: int) -> np.ndarray:
    """RGB uint8 as ai-toolkit shows a reference to the text encoder (_prep_vlm_images): scaled down (never up) to at
    most max_pixels, bicubic with antialiasing, at least 32 px a side. The encoder then rounds it to 32 px."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    size = (max(round(w * scale), 32), max(round(h * scale), 32))
    return np.asarray(image if size == (w, h) else image.resize(size, Image.BICUBIC))


def prompt_key(req: Request, text: str) -> tuple:
    """What an embedding depends on: the text, the token limit and, for the ai-toolkit format, the references as the
    text encoder sees them."""
    vision = (tuple(req.references), req.vision_pixels) if req.edit_format == "ai-toolkit" and req.references else ()
    return text, req.max_sequence_length, vision


def load_mask(path: str | Path, width: int, height: int, feather: int = 0) -> np.ndarray:
    """float32 (height, width) in [0, 1]: white (>= 128) = regenerate. With feather > 0 the painted region is
    dilated by `feather` px and its edge blurred over `feather` px, so the regenerated patch blends into its
    surroundings instead of showing a hard seam; the painted pixels themselves stay fully regenerated."""
    from PIL import Image, ImageFilter

    gray = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    binary = Image.fromarray(((np.asarray(gray) >= 128) * 255).astype(np.uint8))
    if feather > 0:
        size = 2 * int(feather) + 1
        binary = binary.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.GaussianBlur(feather / 2))
    return np.asarray(binary).astype(np.float32) / 255.0


class MOSSImagePipeline:
    def __init__(self, model_dir: str | Path, transformer_dir: str | Path | None = None, low_memory: bool = True,
                 dit_dtype=None, dit_quant: dict | None = None, te_quant: dict | None = None,
                 cache_dir: str | Path | None | bool = None, precise: bool = False, loras=(), control=None):
        """model_dir: a diffusers-layout MOSS-Image directory (transformer/, text_encoder/, tokenizer/, vae/).
        transformer_dir overrides the transformer only (Base and SFT share the text encoder and VAE).
        dit_quant / te_quant: quantization specs applied at load (see quantize.py).
        cache_dir: prompt-embedding cache location (None = default, False = disabled).
        precise: keep the DiT's residual stream and elementwise chains in fp32 (matmuls stay in the weight dtype).
        loras: (path, strength[, dtype[, group]]) applied unmerged to the DiT at load, in order (see lora.py).
            dtype defaults to fp32 (the low-rank path costs little; turbo passes its own); group to "default".
        control: a colour/depth control adapter file (lora.apply_control), needed by requests with control maps."""
        self.model_dir = Path(model_dir)
        self.transformer_dir = Path(transformer_dir) if transformer_dir else self.model_dir / "transformer"
        self.low_memory = low_memory
        self.dit_dtype = dit_dtype  # None keeps the checkpoint's bf16; mx.float32 gives an fp32 reference run
        self.dit_quant, self.te_quant, self.precise = dit_quant, te_quant, precise
        self.loras = [(str(spec[0]), float(spec[1]), *spec[2:]) for spec in loras]
        self.control = str(control) if control else None
        self.text_encoder = self.transformer = self.vae = None
        self.rgba = False  # which decoder self.vae is
        self.cache = None
        if cache_dir is not False:
            from .cache import PromptCache

            self.cache = PromptCache(self.model_dir / "text_encoder", cache_dir, tag=describe(te_quant))
        self.timings: dict = {}

    # Components -------------------------------------------------------------------------------------------------
    def _load_text_encoder(self):
        if self.text_encoder is None:
            self.text_encoder = TextEncoder(self.model_dir / "text_encoder", self.model_dir / "tokenizer")
            quantize_text_encoder(self.text_encoder.model, self.te_quant)

    def _load_transformer(self):
        if self.transformer is None:
            model = load_transformer(self.transformer_dir, self.dit_dtype)
            quantize_dit(model, self.dit_quant)
            if self.loras:
                from .lora import apply_lora

                for path, strength, *rest in self.loras:
                    dtype = rest[0] if rest and rest[0] is not None else mx.float32
                    apply_lora(model, path, strength, dtype, *rest[1:])
            if self.control:
                from .lora import apply_control

                apply_control(model, self.control)
            model.precise = self.precise
            self.transformer = model

    def _load_vae(self, rgba: bool = False):
        if self.vae is None or self.rgba != rgba:
            from .vae import rgba_vae_dir

            self.vae = None
            self.vae = load_vae_decoder(rgba_vae_dir() if rgba else self.model_dir / "vae")
            self.rgba = rgba

    def free_text_encoder(self):
        self.text_encoder = None
        _free()

    def free_transformer(self):
        self.transformer = None
        _free()

    def encode(self, prompts: list[str], max_tokens: int, keep: bool = False, images=None) -> list[mx.array]:
        """Embeddings for each prompt; cached prompts never load the text encoder. images: per prompt, the uint8 RGB
        images its <|image_pad|> placeholders stand for (default none)."""
        images = images or [()] * len(prompts)
        out = [self.cache.get(p, max_tokens, im) if self.cache else None for p, im in zip(prompts, images)]
        missing = [i for i, e in enumerate(out) if e is None]
        if missing:
            self._load_text_encoder()
            for i in missing:
                out[i] = self.text_encoder.encode(prompts[i], max_tokens, images[i])
                if self.cache:
                    self.cache.put(prompts[i], max_tokens, out[i], images[i])
            if self.low_memory and not keep:
                self.free_text_encoder()
        self.timings["prompts_encoded"] = len(missing)
        return out

    # Generation -------------------------------------------------------------------------------------------------
    def encode_image(self, pixels: np.ndarray, keep: bool = False) -> mx.array:
        """uint8 (H, W, 3) -> packed DiT latents, through the official VAE encoder (downloaded on first use)."""
        from .vae import official_vae_file

        self._load_vae()
        self.vae.load_encoder(official_vae_file())
        x = mx.array(pixels.astype(np.float32) / 127.5 - 1)[None]
        z = self.vae.encode_packed(x)
        mx.eval(z)
        if self.low_memory and not keep:
            self.vae = None
            _free()
        return z

    def encode_control(self, req: Request) -> mx.array | None:
        """(1, 256, H/16, W/16): the colour-map and depth-map latents (a missing map is zeros = no control), or None
        without maps. Each map is resized and centre-cropped to the output size and VAE-encoded like an image; a
        grayscale depth map is replicated to RGB, as the adapter was trained. Leaves the VAE loaded."""
        paths = (req.control_color, req.control_depth)
        if not any(paths):
            return None
        if not self.control:
            raise ValueError("control_color / control_depth need a control adapter (--control)")
        maps = [self.encode_image(load_image(p, req.width, req.height), keep=True) if p else None for p in paths]
        zero = mx.zeros_like(next(m for m in maps if m is not None))
        return mx.concatenate([zero if m is None else m for m in maps], axis=1)

    def denoise(self, req: Request, positive: mx.array, negative: mx.array | None, latents: mx.array | None = None,
                on_step=None, keep: bool = False, init: mx.array | None = None, mask: mx.array | None = None,
                references: list[mx.array] | None = None, control: mx.array | None = None) -> mx.array:
        self._load_transformer()
        latents = _sample(self.transformer, req, positive, negative, latents, on_step, init, mask, references, control)
        if self.low_memory and not keep:  # _sample has returned, so no local reference keeps the weights alive
            self.free_transformer()
        return latents

    def decode(self, latents: mx.array, keep: bool = False, rgba: bool = False) -> np.ndarray:
        """uint8 (H, W, 3), or (H, W, 4) with rgba."""
        self._load_vae(rgba)
        # The strip decoder's working set is ~2.7 GB at 1024^2, but its many differently sized temporaries would
        # otherwise pile up in MLX's free-buffer cache (measured 16 GB). Cap the cache while decoding (+~1.5 s).
        previous = mx.set_cache_limit(1 << 30)
        try:
            image = self.vae.decode_packed(latents)
            mx.eval(image)
        finally:
            mx.set_cache_limit(previous)
            mx.clear_cache()
        out = to_uint8(image[0])
        if self.low_memory and not keep:
            self.vae = None
            _free()
        return out

    def __call__(self, req: Request, on_step=None) -> np.ndarray:
        return next(iter(self.generate_many([req], on_step=on_step)))[1]

    def generate_many(self, requests: list[Request], on_step=None):
        """Yield (request, image) for each request, loading each component once.

        All prompts are encoded first (the text encoder loads only for prompts not already cached), then the
        transformer stays resident for every request and decodes alternate with denoising. With low_memory the
        encoder is freed before the transformer loads, so the peak is max(encoder, transformer + VAE decode)."""
        t0 = time.perf_counter()
        texts = []  # prompt_key()s, each encoded once
        for r in requests:
            for t in ([r.prompt] + ([r.negative_prompt] if r.uses_cfg() else [])):
                if prompt_key(r, t) not in texts:
                    texts.append(prompt_key(r, t))
        embeds = []
        for limit in sorted({m for _, m, _ in texts}):
            group = [k for k in texts if k[1] == limit]
            prompts = [vision_prompt(t, len(v[0])) if v else t for t, _, v in group]
            images = [[load_vision(path, v[1]) for path in v[0]] if v else [] for _, _, v in group]
            embeds += list(zip(group, self.encode(prompts, limit, keep=True, images=images)))
        embeds = dict(embeds)
        if self.low_memory and self.text_encoder is not None:
            self.free_text_encoder()
        t1 = time.perf_counter()
        self.timings = {"encode_s": round(t1 - t0, 2), "prompts_encoded": sum(1 for _ in texts)}
        for n, r in enumerate(requests):
            ts = time.perf_counter()
            pos = embeds[prompt_key(r, r.prompt)]
            neg = embeds[prompt_key(r, r.negative_prompt)] if r.uses_cfg() else None
            last = n == len(requests) - 1
            init = mask = pixels = full_mask = None
            if r.init_image:
                pixels = load_image(r.init_image, r.width, r.height)
                init = self.encode_image(pixels)
                if r.mask:
                    full_mask = load_mask(r.mask, r.width, r.height, r.mask_feather)
                    mask = mx.array(latent_mask(full_mask, r.height, r.width))
            refs = [self.encode_image(load_reference(p, r.reference_size), keep=True) for p in r.references]
            control = self.encode_control(r)
            if (refs or control is not None) and self.low_memory:
                self.vae = None
                _free()
            latents = self.denoise(r, pos, neg, on_step=on_step, keep=not last, init=init, mask=mask, references=refs,
                                   control=control)
            del refs
            td = time.perf_counter()
            image = self.decode(latents, keep=not last, rgba=r.rgba)
            if full_mask is not None:  # untouched pixels come straight from the init image
                if image.shape[-1] == 4:  # opaque there
                    pixels = np.concatenate([pixels, np.full_like(pixels[..., :1], 255)], -1)
                m = full_mask[..., None]
                image = np.round(m * image + (1 - m) * pixels).astype(np.uint8)
            if r.keep_alpha:  # the edit keeps the outline: the source's alpha over the new colours
                image = np.concatenate([image[..., :3], source_alpha(r)[..., None]], -1)
            te_ = time.perf_counter()
            self.timings.update({"denoise_s": round(td - ts, 2), "decode_s": round(te_ - td, 2),
                                 "total_s": round(te_ - t0, 2), "prompt_tokens": int(pos.shape[1]),
                                 "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)})
            yield r, image


def save(image: np.ndarray, req: Request, path: str | Path, extra: dict | None = None):
    """Write the PNG and an upstream-style parameters.json next to it."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    params = {k: v for k, v in asdict(req).items() if k != "preview"}  # a scratch file, not a render parameter
    record = {"schema_version": 2, "parameters": {**params, "cfg": req.guidance, "init": req.init_image},
              **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
