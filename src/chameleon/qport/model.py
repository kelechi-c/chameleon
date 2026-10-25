"""Lean qwen-image 2.1 turbo editing around qport.dit: loading from the snapshot, text encoding (magpie's bare-model
encoder call), vae in and out, the 8-step euler loop with hooks, and the source-only capture.

the vae is the diffusers module (AutoencoderKLQwenImage21), used as a plain nn.Module; everything else is here.
"""

import dataclasses as dc
import json
import pathlib as pl
import typing as tp

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from chameleon.qport.dit import Dit, Hooks, Layout, layout, source_layout

MODEL: str = "Qwen/Qwen-Image-2.1-Turbo"
SYSTEM: str = "Comprehend and analyze the provided prompt."
SLOT_TEXT: str = "<image{}><|vision_start|><|image_pad|><|vision_end|>"
TURBO: tuple[float, ...] = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
DEV: str = "cuda"
VAE_CHUNK: int = 4
Text = tuple[Tensor, Tensor]  # (states (1, T, 4096) bf16, image-slot mask (T,) bool)


@dc.dataclass(frozen=True)
class Core:
    """Dit and vae on the gpu, with the vae latent statistics."""

    dit: Dit
    vae: tp.Any
    mean: Tensor  # (1, 64, 1, 1, 1)
    std: Tensor


def snapshot(parts: tuple[str, ...]) -> pl.Path:
    """Local snapshot folder with only the named subfolders."""
    from huggingface_hub import snapshot_download

    pats = ["model_index.json"] + [f"{p}/*" for p in parts]
    return pl.Path(snapshot_download(MODEL, allow_patterns=pats))


def adopt(state: dict[str, Tensor]) -> Dit:
    """The port's dit around already-loaded weights (no copy, no random init): built on meta, then assigned."""
    with torch.device("meta"):
        dit = Dit()
    dit.load_state_dict(state, assign=True)
    return dit.eval()


def load_core() -> Core:
    """Dit (bf16 safetensors straight to the gpu) and vae."""
    from diffusers import AutoencoderKLQwenImage21
    from safetensors.torch import load_file

    snap = snapshot(("transformer", "vae"))
    state = {k: v.to(DEV, torch.bfloat16) for f in sorted((snap / "transformer").glob("*.safetensors"))
             for k, v in load_file(f).items()}
    dit = adopt(state)
    vae = AutoencoderKLQwenImage21.from_pretrained(snap / "vae", torch_dtype=torch.bfloat16).to(DEV).eval()
    cfg = json.loads((snap / "vae" / "config.json").read_text())
    shape = (1, len(cfg["latents_mean"]), 1, 1, 1)
    mean = torch.tensor(cfg["latents_mean"]).view(shape).to(DEV, torch.bfloat16)
    std = torch.tensor(cfg["latents_std"]).view(shape).to(DEV, torch.bfloat16)
    return Core(dit, vae, mean, std)


def load_encoder() -> tuple[tp.Any, tp.Any]:
    """Qwen3-vl encoder (bf16, lm head dropped) on the gpu and its processor."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    snap = snapshot(("text_encoder", "processor", "tokenizer"))
    enc = Qwen3VLForConditionalGeneration.from_pretrained(snap / "text_encoder", dtype=torch.bfloat16)
    enc.lm_head = torch.nn.Identity()
    return enc.to(DEV).eval(), AutoProcessor.from_pretrained(snap / "processor")


def template(prompt: str, images: int) -> str:
    """The raw chat string the checkpoint was trained on, one slot per condition image."""
    vision = " ".join(SLOT_TEXT.format(i + 1) for i in range(images))
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{vision}{prompt or ' '}<|im_end|>\n"
            f"<|im_start|>assistant\n")


@torch.no_grad()
def encode(encoder: tp.Any, processor: tp.Any, prompt: str, images: list[Image.Image]) -> Text:
    """Last decoder layer before the final rmsnorm, system tokens dropped, and the image-slot mask. calls the bare
    model (no lm head, no hidden-state stack)."""
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    pad = processor.tokenizer.encode("<|image_pad|>")[0]
    rgb = [im.convert("RGB") for im in images]
    kw = {"images": rgb} if rgb else {}
    inputs = processor(text=[template(prompt, len(rgb))], return_tensors="pt", **kw).to(DEV)
    lm = getattr(encoder.model, "language_model", encoder.model)
    hook = lm.norm.register_forward_hook(lambda m, a, o: a[0])
    try:
        hidden = encoder.model(**inputs, use_cache=False).last_hidden_state
    finally:
        hook.remove()
    return hidden[:, drop:].to(torch.bfloat16), inputs.input_ids[0, drop:] == pad


def prefix_text(text: Text) -> Tensor:
    """The text states before the first image slot (the same for every image and instruction)."""
    states, slots = text
    first = int(slots.nonzero()[0])
    return states[:, :first]


@torch.no_grad()
def to_latents(core: Core, images: list[Image.Image]) -> Tensor:
    """Packed, normalised vae latents (B, h*w, 64) of rgb images (all the same size, multiples of 16); the vae runs
    on chunks of `VAE_CHUNK` images."""
    px = np.stack([np.asarray(im.convert("RGBA")) for im in images])
    x = torch.from_numpy(px).to(DEV).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    x = x[:, :, None].to(torch.bfloat16)
    z = torch.cat([core.vae.encode(x[i:i + VAE_CHUNK]).latent_dist.mode() for i in range(0, len(x), VAE_CHUNK)])
    z = (z - core.mean) / core.std
    return z[:, :, 0].flatten(2).transpose(1, 2)


@torch.no_grad()
def to_image(core: Core, packed: Tensor, shape: tuple[int, int]) -> Image.Image:
    """Packed latents (1, h*w, 64) -> rgb pil."""
    z = packed.transpose(1, 2).unflatten(2, shape)[:, :, None] * core.std + core.mean
    x = core.vae.decode(z, return_dict=False)[0][:, :, 0]
    arr = ((x[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr).convert("RGB")


def noise(shape: tuple[int, int], seed: int) -> Tensor:
    """Initial noise drawn as the diffusers pipeline does (cuda generator, (1, 1, 64, h, w) bf16), packed."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = torch.randn((1, 1, 64, *shape), generator=g, device=DEV, dtype=torch.bfloat16)
    return n.view(1, 64, shape[0] * shape[1]).transpose(1, 2)


@torch.no_grad()
def edit(core: Core, text: Text, refs: Tensor, ref_shape: tuple[int, int] | list[tuple[int, int]], seed: int,
         hooks: Hooks = Hooks(), out_shape: tuple[int, int] | None = None) -> Image.Image:
    """One turbo edit: prefill (fills the kv cache, reference hooks fire here), 7 cached steps, vae decode. several
    reference images: refs = their packed latents concatenated in slot order, ref_shape = the list of their shapes;
    the output takes the first one's shape unless out_shape is given."""
    shapes = list(ref_shape) if isinstance(ref_shape, list) else [ref_shape]
    shape = out_shape or shapes[0]
    states, slots = text
    lay = layout(slots, shapes + [shape], core.dit.cfg)
    sig = torch.tensor([*TURBO, 0.0], device=DEV)
    levels = (sig * 1000).to(torch.bfloat16) / 1000
    lat = noise(shape, seed)
    v, cache = core.dit.prefill(torch.cat([refs, lat], dim=1), states, levels[:1], lay, hooks)
    lat = euler(lat, v, sig[0], sig[1])
    for i in range(1, len(TURBO)):
        v = core.dit.decode(lat, levels[i:i + 1], lay, cache, i, hooks)
        lat = euler(lat, v, sig[i], sig[i + 1])
    del cache, v  # the kv cache (~2 gb at 1024) would otherwise stay alive through the vae decode (oom on a 24 gb L4)
    return to_image(core, lat, shape)


@torch.no_grad()
def edit_mix(core: Core, texts: list[Text], weights: list[float], refs: Tensor,
             ref_shape: tuple[int, int] | list[tuple[int, int]], seed: int,
             out_shape: tuple[int, int] | None = None, offload: bool = False) -> Image.Image:
    """Edit under several conditionings at once: each step's velocity is sum_i w_i v_i (one prefill and kv cache per
    conditioning, the same noise). strength of a style = [neutral, style] with [1 - a, a] (a > 1 extrapolates);
    a style mix = [style_1, style_2] with weights summing to 1. costs len(texts) forward passes per step. offload:
    only one kv cache (~2 gb at 1024) on the gpu at a time, the others wait on the cpu (fits a 24 gb L4)."""
    shapes = list(ref_shape) if isinstance(ref_shape, list) else [ref_shape]
    shape = out_shape or shapes[0]
    sig = torch.tensor([*TURBO, 0.0], device=DEV)
    levels = (sig * 1000).to(torch.bfloat16) / 1000
    lat = noise(shape, seed)
    lays = [layout(slots, shapes + [shape], core.dit.cfg) for _, slots in texts]
    park = lambda c, dev: [(k.to(dev), v.to(dev)) for k, v in c] if offload else c  # noqa: E731
    caches, vs = [], []
    for (states, _), lay in zip(texts, lays):
        v, cache = core.dit.prefill(torch.cat([refs, lat], dim=1), states, levels[:1], lay)
        vs.append(v)
        caches.append(park(cache, "cpu"))
        del cache
    mix = lambda vv: sum(w * x.float() for w, x in zip(weights, vv)).to(vv[0].dtype)  # noqa: E731
    lat = euler(lat, mix(vs), sig[0], sig[1])
    for i in range(1, len(TURBO)):
        vs = []
        for j, lay in enumerate(lays):
            cache = park(caches[j], DEV)
            vs.append(core.dit.decode(lat, levels[i:i + 1], lay, cache, i))
            caches[j] = park(cache, "cpu")  # decode wrote this step's rows into it
            del cache
        lat = euler(lat, mix(vs), sig[i], sig[i + 1])
    del caches, vs
    return to_image(core, lat, shape)


def euler(lat: Tensor, v: Tensor, s: Tensor, s_next: Tensor) -> Tensor:
    """One euler step, rounded like the reference scheduler (dt * v in the model dtype, sum in fp32)."""
    return (lat.float() + (s_next - s) * v).to(v.dtype)


@torch.no_grad()
def source_rows(core: Core, prefix: Tensor, refs: Tensor, ref_shape: tuple[int, int],
                blocks: tp.Sequence[int]) -> dict[int, Tensor]:
    """Reference rows (B, R, 4096) at the input of each block, from the source-only forward."""
    lay: Layout = source_layout(prefix.shape[1], ref_shape, core.dit.cfg, refs.device)
    return core.dit.source(refs, prefix, lay, blocks)
