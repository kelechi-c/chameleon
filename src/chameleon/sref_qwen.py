"""Chameleon v2 on qwen-image 2.1 turbo: a reference image or --sref code -> a post-txt_in style block appended after
the prompt (dit weights untouched). default cartridge sref_qwen.npz (rank 128, alpha 3, calibrated)."""

from PIL import Image
import torch
from torch import Tensor

from chameleon import sref as sr
from chameleon.qport import model as qm
from chameleon.qport.dit import Hooks


def style_block(s: sr.Sref, z: Tensor, alpha: float | None = None) -> Tensor:
    """(1, K, 4096) bf16 block for Hooks.style_block."""
    return sr.block(s, z, alpha).to(qm.DEV, torch.bfloat16)[None]


@torch.no_grad()
def render(core: qm.Core, text: qm.Text, s: sr.Sref, z: Tensor, seed: int, side: int = 64,
           alpha: float | None = None) -> Image.Image:
    """Text-to-image with the style block (side in latent tokens; 64 = 1024 px)."""
    empty = torch.zeros((1, 0, 64), device=qm.DEV, dtype=torch.bfloat16)
    return qm.edit(core, text, empty, [], seed, Hooks(style_block=style_block(s, z, alpha)), out_shape=(side, side))
