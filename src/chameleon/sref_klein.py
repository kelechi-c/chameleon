"""Chameleon v2 on flux.2 klein 4b (distilled, 4 steps): the style block is appended to the qwen3 context with
sequential text positions; dit / lora weights untouched (tessera-compatible). default cartridge sref_klein.npz
(rank 128, alpha 5, calibrated)."""

import torch
from torch import Tensor

from chameleon import klein
from chameleon import sref as sr


def context(k: klein.Klein, prompt_ctx: Tensor, s: sr.Sref, z: Tensor, alpha: float | None = None) -> Tensor:
    """(1, L + K, 7680) prompt context followed by the style block."""
    return torch.cat([prompt_ctx, sr.block(s, z, alpha)[None].to(k.dev, klein.DT)], 1)


@torch.no_grad()
def render(k: klein.Klein, prompt: str, s: sr.Sref, z: Tensor, seed: int, side: int = 64,
           alpha: float | None = None) -> Tensor:
    """Packed latents of a 4-step distilled sample (decode with klein.decode)."""
    ctx = context(k, klein.encode_text(k, [prompt], 96)[0], s, z, alpha)
    return klein.sample(k, ctx, ctx, None, side, 4, klein.Guide(), seed, distilled=True)
