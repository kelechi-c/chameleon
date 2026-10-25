"""Style backbone (frozen) and the describer head.

backbone: megastyle-encoder (Gaojunyao/MegaStyle, siglip-so400m-patch14-384 fine-tuned with style-supervised
contrastive learning). features per image: the attention-pooled 1152-d embedding and a 4x4 average-pooled grid of
the last patch tokens (16 x 1152).

head: pooled feature -> gaussian posterior (mu, logvar) over a d_z code z -> a small causal transformer that writes
the style out as text through Qwen3's vocabulary. a cartridge is the posterior mean z.
"""

import dataclasses as dc
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

SIGLIP: str = "google/siglip-so400m-patch14-384"
MEGASTYLE: tuple[str, str] = ("Gaojunyao/MegaStyle", "megastyle_encoder.pth")


def load_backbone(dev: str = "cuda") -> tp.Any:
    """SiglipVisionModel with the megastyle weights, frozen, bf16."""
    from huggingface_hub import hf_hub_download
    from transformers import SiglipVisionConfig, SiglipVisionModel

    model = SiglipVisionModel(SiglipVisionConfig.from_pretrained(SIGLIP))
    sd = torch.load(hf_hub_download(*MEGASTYLE), map_location="cpu", weights_only=True)["model"]
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    if not any(k.startswith("vision_model.") for k in model.state_dict()):  # transformers 5 flattened the prefix
        sd = {k.removeprefix("vision_model."): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=True)
    print(f"megastyle backbone loaded: {res}")
    return model.to(dev, torch.bfloat16).eval().requires_grad_(False)


def preprocess(imgs: list[Image.Image], aug: bool = False, rng: np.random.Generator | None = None) -> Tensor:
    """384x384 bicubic, siglip normalisation (mean .5, std .5). aug: random resized crop (scale .5-1) + flip."""
    out = []
    for im in imgs:
        im = im.convert("RGB")
        if aug and rng is not None:
            w, h = im.size
            s = np.sqrt(rng.uniform(0.5, 1.0))
            cw, ch = int(w * s), int(h * s)
            x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
            if rng.random() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        a = np.asarray(im.resize((384, 384), Image.Resampling.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        out.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(out)


@torch.no_grad()
def features(bb: tp.Any, pix: Tensor) -> tuple[Tensor, Tensor]:
    """Pooled (B, 1152) and 4x4 grid (B, 16, 1152) fp32 features of preprocessed images."""
    out = bb(pixel_values=pix.to(bb.device, torch.bfloat16))
    tok = out.last_hidden_state.float()
    b, n, c = tok.shape
    side = int(n ** 0.5)
    grid = F.adaptive_avg_pool2d(tok.transpose(1, 2).reshape(b, c, side, side), 4).flatten(2).transpose(1, 2)
    return out.pooler_output.float(), grid


@dc.dataclass(frozen=True)
class HeadCfg:
    """Describer head shape (stored as a dict in head_pre.pt)."""

    d_in: int = 1152        # MegaStyle pooled feature
    d_z: int = 256          # the cartridge code z
    n_tok: int = 16         # max tokens written (r7: 144)
    width: int = 1024       # decoder width
    depth: int = 4          # decoder layers
    heads: int = 16         # attention heads
    d_out: int = 7680       # token-embedding width of the table the head reads and writes (Qwen3-4B: 2560)
    space: str = "state"    # head type; only "ar" (the describer) is supported here
    tau: float = 1.0        # soft-token temperature (0 = the written text itself)
    n_mem: int = 4          # memory tokens the code z is expanded into
    dropout: float = 0.0    # decoder dropout
    grid: bool = False      # unused by the describer; kept so older checkpoints load
    out_scale: float = 1.0  # unused by the describer; kept so older checkpoints load


class StyleDescriber(nn.Module):
    """Autoregressive "describer": posterior over z -> n_mem memory tokens -> causal transformer that writes the style
    text, reading and writing tokens through the frozen Qwen3-4B input-embedding matrix E (passed in, not stored).
    soft tokens are softmax(logits / tau) @ E along the greedy path (tau 0 = the generated text itself); z stays the
    cartridge."""

    def __init__(self, cfg: HeadCfg, text_mean: Tensor, text_std: Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.enc = nn.Sequential(nn.LayerNorm(cfg.d_in), nn.Linear(cfg.d_in, d), nn.GELU(), nn.Linear(d, 2 * cfg.d_z))
        self.z_in = nn.Linear(cfg.d_z, cfg.n_mem * d)
        self.tok_in = nn.Linear(cfg.d_out, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Parameter(torch.randn(cfg.n_mem + cfg.n_tok + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.d_out)
        self.register_buffer("text_mean", text_mean.float().clone())  # unused; keeps the StyleHead interface
        self.register_buffer("text_std", text_std.float().clone())

    def posterior(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """(mu, logvar), each (..., d_z)."""
        mu, logvar = self.enc(pooled).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    def logits(self, z: Tensor, ids_in: Tensor, emb: Tensor) -> Tensor:
        """Teacher-forced next-token logits (B, L + 1, vocab) for input ids (B, L): position j predicts token j."""
        b, n = z.shape[0], self.cfg.n_mem
        x = torch.cat([self.z_in(z).view(b, n, -1), self.bos.expand(b, 1, -1),
                       self.tok_in(emb[ids_in].float())], dim=1)
        x = x + self.pos[: x.shape[1]]
        causal = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device), 1)
        h = self.body(x, mask=causal)[:, n:]
        return (self.out(self.norm(h)).to(emb.dtype) @ emb.T).float()

    def describe(self, z: Tensor, emb: Tensor, eos: int, tau: float, out_emb: Tensor | None = None
                 ) -> tuple[list[Tensor], list[list[int]]]:
        """Greedy decoding up to eos or n_tok tokens. returns per-sample soft tokens (n_i, d) and token ids. out_emb:
        another text encoder's input-embedding matrix over the same vocabulary (e.g. qwen-image's qwen3-vl): the soft
        tokens are then softmax(logits / tau) @ out_emb, while decoding itself still reads emb."""
        b = z.shape[0]
        ids = torch.zeros(b, 0, dtype=torch.long, device=z.device)
        steps = []
        for _ in range(self.cfg.n_tok):
            lg = self.logits(z, ids, emb)[:, -1]
            steps.append(lg)
            ids = torch.cat([ids, lg.argmax(-1, keepdim=True)], dim=1)
        lg = torch.stack(steps, 1)
        oe = emb if out_emb is None else out_emb
        p = F.one_hot(ids, lg.shape[-1]).to(oe.dtype) if tau == 0 else (lg / tau).softmax(-1).to(oe.dtype)
        soft = p @ oe[: lg.shape[-1]]
        out_s, out_i = [], []
        for j in range(b):
            row = ids[j].tolist()
            n = row.index(eos) if eos in row else len(row)
            n = max(n, 1)  # keep at least one token so every prompt has a placeholder
            out_s.append(soft[j, :n]), out_i.append(row[:n])
        return out_s, out_i


def pool_posterior(mu: Tensor, logvar: Tensor, mask: Tensor, weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Combine per-reference posteriors (B, R, d_z) with a validity mask (B, R) (+ mix weights) into one code
    posterior: weighted mean of mu and of the variance."""
    w = mask.float() if weights is None else mask.float() * weights
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    m = (w[..., None] * mu).sum(1)
    var = (w[..., None] * logvar.exp()).sum(1)
    return m, var.log()


def load_head(path: str, dev: str = "cuda") -> nn.Module:
    """Rebuild the describer from a checkpoint (head_pre.pt)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = HeadCfg(**ck["cfg"])
    assert cfg.space == "ar", f"not a describer head: space {cfg.space}"
    sd = ck["head"]
    head = StyleDescriber(cfg, torch.zeros_like(sd["text_mean"]), torch.ones_like(sd["text_std"])).to(dev)
    head.load_state_dict(sd)
    return head.eval()
