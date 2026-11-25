"""flux.2 klein 4b, frozen: text / image encoding, the rectified-flow training target and a sampler that takes
extra conditioning tokens (style tokens) appended after the prompt tokens.

conventions (match diffusers' Flux2KleinPipeline): text = qwen3 hidden states of layers 9/18/27 stacked to 7680-d,
padded to a fixed length, no attention mask in the dit; latents = vae mode, 2x2 patchified to 128 channels,
batch-norm normalised, packed to (B, h*w, 128); x_s = (1 - s) x0 + s eps, velocity target eps - x0, the dit takes
s in [0, 1]; text rope ids (0, 0, 0, l), image ids (0, h, w, 0).
"""

import dataclasses as dc
import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import Tensor

BASE: str = "black-forest-labs/FLUX.2-klein-base-4B"
DISTILLED: str = "black-forest-labs/FLUX.2-klein-4B"
LAYERS: tuple[int, ...] = (9, 18, 27)
DT = torch.bfloat16


@dc.dataclass(frozen=True)
class Klein:
    """Frozen model parts plus the latent normalisation stats."""

    dit: tp.Any | None
    vae: tp.Any | None
    te: tp.Any | None
    tok: tp.Any
    bn_mean: Tensor
    bn_std: Tensor
    dev: str


def load(repo: str = BASE, dev: str = "cuda", parts: tuple[str, ...] = ("dit", "vae", "text")) -> Klein:
    """Load the requested frozen parts to dev in bf16 (the qwen3 body is truncated after the last layer we read)."""
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from transformers import AutoTokenizer, Qwen3Model

    frozen = lambda m: m.to(dev).eval().requires_grad_(False)  # noqa: E731
    dit = frozen(Flux2Transformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=DT)) \
        if "dit" in parts else None
    v = frozen(AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=DT)) if "vae" in parts else None
    te = None
    if "text" in parts:
        te = Qwen3Model.from_pretrained(repo, subfolder="text_encoder", torch_dtype=DT)
        te.layers = te.layers[: max(LAYERS) + 1]  # hidden_states[i] is the input of layer i: later layers unused
        te = frozen(te)
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    mean = std = torch.zeros(1)
    if v is not None:
        mean = v.bn.running_mean.view(1, -1, 1, 1).float().to(dev)
        std = torch.sqrt(v.bn.running_var.view(1, -1, 1, 1).float() + v.config.batch_norm_eps).to(dev)
    return Klein(dit, v, te, tok, mean, std, dev)


@torch.no_grad()
def encode_text(k: Klein, prompts: list[str], max_len: int = 64) -> tuple[Tensor, Tensor]:
    """(B, max_len, 7680) prompt states and the (B, max_len) token mask; chat template + fixed padding as in the
    pipeline (the mask is not given to the dit, it only marks real tokens for statistics)."""
    texts = [k.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for p in prompts]
    enc = k.tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(k.dev)
    out = k.te(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
    h = torch.stack([out.hidden_states[i] for i in LAYERS], dim=2)
    return h.reshape(h.shape[0], h.shape[1], -1).to(DT), enc.attention_mask


PLACE: str = "<|vision_pad|>"  # placeholder token whose input embedding is replaced by a soft style token


INIT_PHRASE: str = ("rendered in a distinctive visual style with its own colour palette, lighting, texture, medium, "
                    "brushwork and mood, consistent across every image of the series")


def init_embeds(k: Klein, n_tok: int) -> Tensor:
    """(n_tok, d) input embeddings of the first n_tok tokens of INIT_PHRASE: an on-manifold start for soft style tokens
    (vocab-mean tokens are off-manifold and start worse than no tokens at all)."""
    ids = k.tok(INIT_PHRASE, add_special_tokens=False).input_ids
    assert len(ids) >= n_tok, f"INIT_PHRASE has {len(ids)} tokens < {n_tok}"
    return k.te.embed_tokens.weight[torch.tensor(ids[:n_tok], device=k.dev)].float().detach()


def style_prompt(caption: str, n_tok: int, sep: str = ". Style: ") -> str:
    """Carrier prompt for n_tok soft style tokens after the caption (style only when the caption is empty). vocab
    heads use sep "." so the prompt tokenises like the oracle "{caption}. {style text}"."""
    return (f"{caption}{sep}" if caption else sep.lstrip(". ")) + PLACE * n_tok


def eos_id(k: Klein) -> int:
    """Stop symbol of the ar describer (never fed to the text encoder)."""
    return k.tok.convert_tokens_to_ids("<|endoftext|>")


def describer_ids(k: Klein, texts: list[str], n_tok: int) -> tuple[Tensor, Tensor]:
    """Teacher-forcing (inputs (B, n_tok), targets (B, n_tok + 1)) for the ar describer: tokens of " " + style text,
    truncated to n_tok, then eos; targets past eos are -100 (ignored)."""
    eos = eos_id(k)
    rows = [k.tok(" " + t, add_special_tokens=False).input_ids[:n_tok] for t in texts]
    inp = torch.tensor([r + [eos] * (n_tok - len(r)) for r in rows], device=k.dev)
    tgt = torch.tensor([r + [eos] + [-100] * (n_tok - len(r)) for r in rows], device=k.dev)
    return inp, tgt


def style_ids(k: Klein, texts: list[str], n_tok: int) -> Tensor:
    """(B, n_tok) token ids of " " + style text (as they follow "{caption}." in the oracle prompt), truncated, padded
    with spaces: the per-position targets of a vocab head."""
    fill = k.tok(" ", add_special_tokens=False).input_ids[-1]
    rows = [k.tok(" " + t, add_special_tokens=False).input_ids[:n_tok] for t in texts]
    return torch.tensor([r + [fill] * (n_tok - len(r)) for r in rows], device=k.dev)


def vocab_logits(k: Klein, q: Tensor) -> Tensor:
    """(B, K, d) head queries -> (B, K, vocab) float logits against the frozen input-embedding matrix."""
    w = k.te.embed_tokens.weight
    return (q.to(w.dtype) @ w.T).float()


def vocab_soft(k: Klein, q: Tensor, tau: float) -> Tensor:
    """Soft tokens: softmax(logits / tau) @ E (tau 0: the argmax tokens, i.e. plain text). they stay on the convex hull
    of real token embeddings, so the encoder sees near-manifold inputs."""
    w = k.te.embed_tokens.weight
    lg = vocab_logits(k, q)
    p = F.one_hot(lg.argmax(-1), lg.shape[-1]).to(w.dtype) if tau == 0 else (lg / tau).softmax(-1).to(w.dtype)
    return p @ w


def encode_soft(k: Klein, prompts: list[str], soft: Tensor | list[Tensor] | None, max_len: int = 80
                ) -> tuple[Tensor, Tensor]:
    """Like encode_text, but differentiable and with the input embeddings of the PLACE tokens replaced by soft
    tokens (B, K, 2560), in order; prompts without placeholders pass soft=None. the frozen qwen3 encodes the soft
    tokens jointly with the caption."""
    texts = [k.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for p in prompts]
    enc = k.tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(k.dev)
    emb = k.te.embed_tokens(enc.input_ids)
    if soft is not None:  # (B, K, d), or a list of (n_i, d) for per-prompt counts (in prompt order)
        flat = torch.cat(list(soft)) if isinstance(soft, list) else soft.reshape(-1, soft.shape[-1])
        pos = enc.input_ids == k.tok.convert_tokens_to_ids(PLACE)
        assert int(pos.sum()) == flat.shape[0], "placeholder count != soft tokens (truncated?)"
        emb = emb.clone()
        emb[pos] = flat.to(emb.dtype)
    out = k.te(inputs_embeds=emb, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
    h = torch.stack([out.hidden_states[i] for i in LAYERS], dim=2)
    return h.reshape(h.shape[0], h.shape[1], -1).to(DT), enc.attention_mask


def patchify(x: Tensor) -> Tensor:
    """(B, C, H, W) -> (B, 4C, H/2, W/2), channel order (c, dy, dx)."""
    b, c, h, w = x.shape
    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(x: Tensor) -> Tensor:
    """Inverse of patchify."""
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


@torch.no_grad()
def encode_images(k: Klein, x: Tensor) -> Tensor:
    """Images (B, 3, H, W) in [-1, 1] -> packed normalised latents (B, H/16 * W/16, 128) fp32."""
    lat = k.vae.encode(x.to(k.dev, DT)).latent_dist.mode()
    lat = (patchify(lat).float() - k.bn_mean) / k.bn_std
    return lat.flatten(2).transpose(1, 2)


@torch.no_grad()
def decode(k: Klein, packed: Tensor, side: int) -> Tensor:
    """Packed latents of a side x side token grid -> images (B, 3, H, W) in [0, 1]."""
    lat = packed.float().transpose(1, 2).reshape(packed.shape[0], -1, side, side) * k.bn_std + k.bn_mean
    img = k.vae.decode(unpatchify(lat).to(DT), return_dict=False)[0]
    return (img.float().clamp(-1, 1) + 1) / 2


def img_ids(side: int, dev: str) -> Tensor:
    """(side*side, 4) rope ids of the target grid."""
    z = torch.zeros(1, dtype=torch.long)
    return torch.cartesian_prod(z, torch.arange(side), torch.arange(side), z).to(dev)


def txt_ids(n: int, dev: str) -> Tensor:
    """(n, 4) rope ids of n conditioning tokens (prompt then style tokens, positions continue)."""
    z = torch.zeros(1, dtype=torch.long)
    return torch.cartesian_prod(z, z, z, torch.arange(n)).to(dev)


def velocity(k: Klein, x: Tensor, sigma: Tensor, ctx: Tensor, side: int) -> Tensor:
    """Dit velocity for packed noisy latents x (B, N, 128) at noise levels sigma (B,) given context (B, L, 7680)."""
    t = (sigma.float() * 1000).to(DT) / 1000  # the pipeline's rounding: bf16(sigma * 1000) / 1000
    out = k.dit(hidden_states=x.to(DT), encoder_hidden_states=ctx.to(DT), timestep=t,
                img_ids=img_ids(side, k.dev), txt_ids=txt_ids(ctx.shape[1], k.dev), guidance=None, return_dict=False)
    return out[0]


def train_sigmas(n: int, mu: float, gen: torch.Generator, dev: str) -> Tensor:
    """Logit-normal noise levels, then the exponential resolution shift used at inference (e^mu)."""
    s = torch.sigmoid(torch.randn(n, generator=gen, device="cpu")).to(dev)
    return shift(s, mu)


def shift(s: Tensor, mu: float) -> Tensor:
    """Exponential time shift: s -> e^mu / (e^mu + 1/s - 1)."""
    return math.exp(mu) / (math.exp(mu) + (1 / s.clamp_min(1e-6) - 1))


def empirical_mu(seq: int, steps: int) -> float:
    """The pipeline's resolution / step-count dependent shift."""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if seq > 4300:
        return a2 * seq + b2
    m200, m10 = a2 * seq + b2, a1 * seq + b1
    a = (m200 - m10) / 190.0
    return a * steps + (m200 - 200.0 * a)


@dc.dataclass(frozen=True)
class Guide:
    """Guidance weights: text w_t (prompt vs empty prompt, style on) and style w_s (style vs null tokens, empty prompt).
    v = v(0, null) + w_s (v(0, s) - v(0, null)) + w_t (v(c, s) - v(0, s)); w_s = 1 skips the (0, null) pass."""

    w_t: float = 4.0
    w_s: float = 1.0


@torch.no_grad()
def sample(k: Klein, ctx: Tensor, ctx_text_off: Tensor, ctx_off: Tensor | None, side: int, steps: int, guide: Guide,
           seed: int, distilled: bool = False, exact: bool = False) -> Tensor:
    """Euler sampler. ctx = (prompt, style), ctx_text_off = (empty prompt, style), ctx_off = (empty, null), each
    (B, L, 7680). distilled: no guidance (ctx only). exact: reproduce the pipeline's precision (bf16 latents, guidance
    and dt * v in bf16) for parity checks; default is fp32 latents. returns packed latents (B, side*side, 128)."""
    b = ctx.shape[0]
    g = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn(b, side * side, 128, generator=g).to(k.dev)
    sig = torch.linspace(1.0, 1.0 / steps, steps)
    sig = torch.cat([shift(sig, empirical_mu(side * side, steps)), torch.zeros(1)]).tolist()
    use_null = ctx_off is not None and guide.w_s != 1.0 and not distilled
    for i in range(steps):
        s = torch.full((b,), sig[i], device=k.dev)
        cast = (lambda t: t) if exact else (lambda t: t.float())  # noqa: E731
        if distilled:
            v = cast(velocity(k, x, s, ctx, side))
        else:
            cs = [ctx, ctx_text_off] + ([ctx_off] if use_null else [])
            vs = [cast(velocity(k, x, s, c, side)) for c in cs] if exact else \
                cast(velocity(k, x.repeat(len(cs), 1, 1), s.repeat(len(cs)), torch.cat(cs), side)).chunk(len(cs))
            v = vs[1] + guide.w_t * (vs[0] - vs[1])
            if use_null:
                v = v + (guide.w_s - 1.0) * (vs[1] - vs[2])
        if exact:
            x = (x.to(DT).float() + torch.tensor(sig[i + 1] - sig[i]) * v).to(DT)
        else:
            x = x + (sig[i + 1] - sig[i]) * v
    return x.float()
