"""Lean qwen-image 2.1 dit for puppet, derived from magpie's port (/home/tensor/Shared/code/ml/magpie/src/magpie/dit.py)
with the blackwell kernels removed: plain bf16 torch, sdpa attention, the same arithmetic order as magpie's unfused path.

what is added for puppet:
  Hooks   - explicit edit points instead of torch hooks: `ref(block, rows)` sees the reference (source-image) rows at a
            block's input on the prefill; `target(block, step, rows)` sees the target rows at a block's input on every
            step. each returns the rows to use (or the same tensor).
  source  - the reference rows alone: [fixed text prefix | reference latents], every token at t = 0, stopped after the
            last block asked for. valid because reference rows see only earlier text (block-causal mask) and that text
            precedes the image in the template, so it is the same for every image and instruction.

parameter names match the diffusers checkpoint, so `load_state_dict` takes the released weights unchanged.
"""

import dataclasses as dc
import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Segment = tuple[int, int, bool]  # (start, end, is_text) run of the prefix
KV = tuple[Tensor, Tensor]  # one layer's keys and values for the whole sequence, (B, S, heads, head_dim)
Mod = tuple[Tensor, Tensor, Tensor, Tensor]  # (1 + scale1, tanh gate1, 1 + scale2, tanh gate2), per token rows
SLOT = 4  # one vision-language image slot stands for a 2x2 group of latent tokens


@dc.dataclass(frozen=True)
class DitConfig:
    """Shape of the 7b single-stream dit."""

    width: int = 4096
    heads: int = 32
    layers: int = 32
    latent_channels: int = 64
    context_dim: int = 4096
    mlp_ratio: int = 3
    rope_axes: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6


@dc.dataclass(frozen=True)
class Layout:
    """Step-independent token map of one request: [text and reference tokens (prefix) | target tokens]."""

    repeats: Tensor  # (T + N/4,) joint tokens per encoder position (4 at image slots, 1 at text)
    image: Tensor  # (S,) bool, true where a latent token sits
    prefix: int  # text + reference tokens
    segments: tuple[Segment, ...]  # runs of the prefix: text runs and reference images
    rope: Tensor  # (S, head_dim / 2) complex rotations

    @property
    def ref_idx(self) -> Tensor:
        """Positions of the reference-image tokens (image tokens inside the prefix)."""
        return self.image[: self.prefix].nonzero().squeeze(1)


@dc.dataclass(frozen=True)
class Hooks:
    """Edit points; None means untouched."""

    ref: tp.Callable[[int, Tensor], Tensor] | None = None
    target: tp.Callable[[int, int, Tensor], Tensor] | None = None


def rope_table(index: Tensor, dim: int, theta: float = 10000.0) -> Tensor:
    """Complex rotations e^{i index theta^(-2k/dim)} for one rope axis."""
    freqs = torch.outer(index.float(), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_positions(image: list[bool], shapes: list[tuple[int, int]]) -> tuple[list[int], list[int], list[int]]:
    """(frame, height, width) positions: text advances one counter on all axes; an image block freezes the frame at it
    and lays a zero-centred grid. `shapes` covers every image block in order."""
    frame: list[int] = []
    height: list[int] = []
    width: list[int] = []
    cursor = position = 0
    for h, w in shapes:
        start = image.index(True, cursor)
        text = list(range(position, position + start - cursor))
        frame, height, width = frame + text, height + text, width + text
        position += start - cursor
        cursor = start + h * w
        frame += [position] * (h * w)
        height += [y for y in range(-(h - h // 2), h // 2) for _ in range(w)]
        width += [x for _ in range(h) for x in range(-(w - w // 2), w // 2)]
        position += max(h, w)
    tail = list(range(position, position + len(image) - cursor))
    return frame + tail, height + tail, width + tail


def _rope(image: Tensor, shapes: list[tuple[int, int]], cfg: DitConfig) -> Tensor:
    """Rope rotations for a joint image mask and its image shapes."""
    tables = [torch.cat([rope_table(torch.arange(8192), d), rope_table(-torch.arange(1024).flip(0) - 1, d)])
              for d in cfg.rope_axes]
    axes = rope_positions(image.tolist(), shapes)
    return torch.cat([t[torch.tensor(a)] for t, a in zip(tables, axes)], dim=-1).to(image.device)


def _runs(image: Tensor, lengths: list[int], prefix: int) -> tuple[Segment, ...]:
    """Text runs and image blocks of the prefix."""
    ids = torch.full_like(image, -1, dtype=torch.long)
    ids[image] = torch.repeat_interleave(torch.arange(len(lengths), device=image.device),
                                         torch.tensor(lengths, device=image.device))
    edges = [0] + (ids[1:prefix] != ids[: prefix - 1]).nonzero().flatten().add(1).tolist() + [prefix]
    return tuple((a, b, bool(ids[a] < 0)) for a, b in zip(edges, edges[1:]) if b > a)


def layout(slots: Tensor, shapes: list[tuple[int, int]], cfg: DitConfig) -> Layout:
    """Layout of an edit or t2i request. `slots` is the encoder's image-slot mask (T,); `shapes` (h, w) in latent
    tokens, reference images first, target last."""
    target = shapes[-1][0] * shapes[-1][1]
    slots = torch.cat([slots.bool(), slots.new_ones(target // SLOT, dtype=torch.bool)])
    repeats = torch.where(slots, SLOT, 1)
    image = torch.repeat_interleave(slots, repeats)
    prefix = image.numel() - target
    return Layout(repeats, image, prefix, _runs(image, [h * w for h, w in shapes], prefix), _rope(image, shapes, cfg))


def source_layout(n_text: int, shape: tuple[int, int], cfg: DitConfig, device: torch.device) -> Layout:
    """Layout of [n_text prefix text tokens | one reference image], no target (the source-only forward)."""
    image = torch.cat([torch.zeros(n_text, dtype=torch.bool), torch.ones(shape[0] * shape[1], dtype=torch.bool)])
    image = image.to(device)
    prefix = image.numel()
    return Layout(torch.ones(0, dtype=torch.long, device=device), image, prefix,
                  _runs(image, [shape[0] * shape[1]], prefix), _rope(image, [shape], cfg))


def rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Rms norm over the last axis, statistics in fp32 (diffusers RMSNorm)."""
    scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x * scale).to(x.dtype) * weight


def rotate(x: Tensor, rope: Tensor) -> Tensor:
    """Rotate channel pairs of q or k (B, S, heads, head_dim) by per-token complex phases."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * rope[:, None]).flatten(3).type_as(x)


def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Scaled dot-product attention over (B, S, heads, head_dim)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)


def attend_prefix(q: Tensor, k: Tensor, v: Tensor, lay: Layout) -> Tensor:
    """Block-causal attention of the prefix rows: one call per run; text runs get a causal triangle over their own
    keys, an image block sees everything up to its end."""

    def run(start: int, end: int, is_text: bool) -> Tensor:
        size = end - start
        mask = (torch.cat([q.new_ones(size, start, dtype=torch.bool), q.new_ones(size, size, dtype=torch.bool).tril()],
                          1) if is_text else None)
        return sdpa(q[:, start:end], k[:, :end], v[:, :end], mask)

    return torch.cat([run(*s) for s in lay.segments], dim=1)


class Attention(nn.Module):
    """Single-stream attention: q, k, v, out projections with rms qk-norm and rope."""

    def __init__(self, cfg: DitConfig) -> None:
        """Bias-free projections and per-head qk-norm weights."""
        super().__init__()
        self.heads, self.eps = cfg.heads, cfg.eps
        self.to_q, self.to_k, self.to_v = (nn.Linear(cfg.width, cfg.width, bias=False) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(cfg.width, cfg.width, bias=False)])
        self.norm_q = nn.Module()
        self.norm_k = nn.Module()
        self.norm_q.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))
        self.norm_k.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))

    def qkv(self, n: Tensor, rope: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-head q, k (normed, rotated) and v of the modulated input n (B, S, width)."""
        q, k, v = (lin(n).unflatten(-1, (self.heads, -1)) for lin in (self.to_q, self.to_k, self.to_v))
        q = rotate(rms(q, self.norm_q.weight, self.eps), rope)
        k = rotate(rms(k, self.norm_k.weight, self.eps), rope)
        return q, k, v


class SwiGLU(nn.Module):
    """out(silu(gate(x)) * proj(x))."""

    def __init__(self, cfg: DitConfig) -> None:
        """Bias-free linears, hidden = width x mlp_ratio."""
        super().__init__()
        hidden = cfg.width * cfg.mlp_ratio
        self.proj = nn.Linear(cfg.width, hidden, bias=False)
        self.gate_layer = nn.Linear(cfg.width, hidden, bias=False)
        self.out = nn.Linear(hidden, cfg.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """The mlp."""
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


class Block(nn.Module):
    """One single-stream block; the modulation is shared by all blocks and passed in."""

    def __init__(self, cfg: DitConfig) -> None:
        """Attention and mlp."""
        super().__init__()
        self.eps = cfg.eps
        self.attn = Attention(cfg)
        self.img_mlp = SwiGLU(cfg)

    def _norm(self, x: Tensor, mod: Tensor) -> Tensor:
        """layer_norm(x) * mod."""
        return F.layer_norm(x, x.shape[-1:], eps=self.eps) * mod

    def _mlp(self, x: Tensor, mod: Mod) -> Tensor:
        """The mlp branch with its residual."""
        return x + mod[3] * self.img_mlp(self._norm(x, mod[2]))

    def prefill(self, x: Tensor, mod: Mod, lay: Layout) -> tuple[Tensor, KV]:
        """Whole joint sequence; returns the stream and this layer's keys and values (prefix rows = the cache)."""
        q, k, v = self.attn.qkv(self._norm(x, mod[0]), lay.rope)
        att = attend_prefix(q, k, v, lay)
        if lay.prefix < x.shape[1]:
            att = torch.cat([att, sdpa(q[:, lay.prefix:], k, v)], dim=1)
        x = x + mod[1] * self.attn.to_out[0](att.flatten(2))
        return self._mlp(x, mod), (k, v)

    def decode(self, x: Tensor, mod: Mod, rope: Tensor, cache: KV) -> Tensor:
        """Target rows only: write their keys and values into the cache tail, attend to [prefix, target] unmasked."""
        k, v = cache
        rows = x.shape[1]
        q, k_new, v_new = self.attn.qkv(self._norm(x, mod[0]), rope)
        k[:, -rows:], v[:, -rows:] = k_new, v_new
        x = x + mod[1] * self.attn.to_out[0](sdpa(q, k, v).flatten(2))
        return self._mlp(x, mod)


class Dit(nn.Module):
    """Qwen-image 2.1 single-stream dit (32 blocks, width 4096)."""

    def __init__(self, cfg: DitConfig = DitConfig()) -> None:
        """Modules named as in the diffusers checkpoint."""
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.time_text_embed = nn.Module()
        self.time_text_embed.timestep_embedder = nn.Module()
        self.time_text_embed.timestep_embedder.linear_1 = nn.Linear(256, w, bias=False)
        self.time_text_embed.timestep_embedder.linear_2 = nn.Linear(w, w, bias=False)
        self.txt_in = nn.Module()
        self.txt_in.text_norm = nn.Module()
        self.txt_in.text_norm.weight = nn.Parameter(torch.zeros(cfg.context_dim))
        self.txt_in.in_layer = nn.Linear(cfg.context_dim, w, bias=False)
        self.txt_in.out_layer = nn.Linear(w, w, bias=False)
        self.img_in = nn.Linear(cfg.latent_channels, w, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(w, 4 * w, bias=False))
        self.transformer_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(w, w, bias=False)
        self.proj_out = nn.Linear(w, cfg.latent_channels, bias=False)

    def time_embed(self, sigma: Tensor) -> Tensor:
        """Sinusoidal embedding of sigma in [0, 1] (cos half, then sin half) through a two-layer mlp."""
        emb = self.time_text_embed.timestep_embedder
        freqs = torch.exp(-math.log(10000) * torch.arange(128).float() / 128).to(sigma.device)
        args = 1000.0 * sigma.float()[:, None] * freqs[None]
        wave = torch.cat([args.cos(), args.sin()], dim=-1).to(emb.linear_1.weight.dtype)
        return emb.linear_2(F.silu(emb.linear_1(wave)))

    def text_embed(self, text: Tensor) -> Tensor:
        """Zero-centred rms norm then a two-layer mlp into the dit width."""
        scale = torch.rsqrt(text.float().pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        normed = (text.float() * scale * (self.txt_in.text_norm.weight.float() + 1)).to(text.dtype)
        return self.txt_in.out_layer(F.gelu(self.txt_in.in_layer(normed), approximate="tanh"))

    def modulate(self, temb: Tensor, target: Tensor | None) -> tuple[Mod, Tensor]:
        """Per-token modulation from temb rows [sigma, 0]: target tokens read the sigma row, prefix tokens the t = 0
        row; target None means every token reads the sigma row."""

        def rows(p: Tensor) -> Tensor:
            real, zero = p[:-1, None], p[-1:, None]
            return real if target is None else torch.where(target[None, :, None], real, zero)

        s1, g1, s2, g2 = self.modulation(temb).chunk(4, dim=-1)
        final = rows(self.norm_out.linear(F.silu(temb)))
        return (1 + rows(s1), rows(g1).tanh(), 1 + rows(s2), rows(g2).tanh()), 1 + final

    def head(self, x: Tensor, final: Tensor) -> Tensor:
        """Final norm, modulation and projection to latent velocity."""
        return self.proj_out(F.layer_norm(x, (self.cfg.width,), eps=self.cfg.eps) * final)

    def prefill(self, latents: Tensor, text: Tensor, sigma: Tensor, lay: Layout,
                hooks: Hooks = Hooks()) -> tuple[Tensor, list[KV]]:
        """First step over the whole joint sequence: (target velocity, kv cache). latents (B, R + N, 64): reference
        latents then the noisy target; text (B, T, 4096)."""
        batch, target = latents.shape[0], lay.image.numel() - lay.prefix
        t = self.text_embed(text)
        x = torch.cat([t, t.new_zeros(batch, target // SLOT, t.shape[-1])], dim=1).repeat_interleave(lay.repeats, dim=1)
        x[:, lay.image] = self.img_in(latents)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, torch.arange(lay.image.numel(), device=x.device) >= lay.prefix)
        ref = lay.ref_idx
        cache: list[KV] = []
        for i, block in enumerate(self.transformer_blocks):
            if hooks.ref is not None:
                x[:, ref] = hooks.ref(i, x[:, ref])
            if hooks.target is not None:
                x[:, lay.prefix:] = hooks.target(i, 0, x[:, lay.prefix:])
            x, kv = block.prefill(x, mod, lay)
            cache.append(kv)
        return self.head(x[:, lay.prefix:], final[:, lay.prefix:]), cache

    def decode(self, target: Tensor, sigma: Tensor, lay: Layout, cache: list[KV], step: int,
               hooks: Hooks = Hooks()) -> Tensor:
        """Later steps: target rows only, attending to the cached prefix (batch 1)."""
        x = self.img_in(target)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, None)
        rope = lay.rope[lay.prefix:]
        for i, (block, kv) in enumerate(zip(self.transformer_blocks, cache)):
            if hooks.target is not None:
                x = hooks.target(i, step, x)
            x = block.decode(x, mod, rope, kv)
        return self.head(x, final)

    def source(self, refs: Tensor, prefix_text: Tensor, lay: Layout, blocks: tp.Sequence[int]) -> dict[int, Tensor]:
        """Reference rows at the input of each block in `blocks`, from the source-only forward (stops after the last
        one). refs (B, R, 64) packed reference latents, prefix_text (1, P, 4096) the fixed text before the image."""
        n = prefix_text.shape[1]
        t = self.text_embed(prefix_text).expand(refs.shape[0], -1, -1)
        x = torch.cat([t, self.img_in(refs)], dim=1)
        temb = self.time_embed(torch.zeros(2, device=refs.device, dtype=refs.dtype))
        mod, _ = self.modulate(temb, None)
        out: dict[int, Tensor] = {}
        for i, block in enumerate(self.transformer_blocks[: max(blocks) + 1]):
            if i in blocks:
                out[i] = x[:, n:].clone()
            if i < max(blocks):
                x = block.prefill(x, mod, lay)[0]
        return out
