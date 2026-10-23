"""Style backbone (frozen) and the trainable style head.

backbone: megastyle-encoder (Gaojunyao/MegaStyle, siglip-so400m-patch14-384 fine-tuned with style-supervised
contrastive learning). features per image: the attention-pooled 1152-d embedding and a 4x4 average-pooled grid of
the last patch tokens (16 x 1152).

head: pooled feature -> gaussian posterior (mu, logvar) over a d_z code -> transformer decoder -> n_tok tokens in
the text-encoder output space. outputs are parametrised as text_mean + text_std * o with text_mean = the mean pad
state, so o = 0 is a near no-op token and the head works in normalised units whatever the channel scales. a cartridge is the
posterior mean z; tokens are a pure function of z (and the optional grid).
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
    """Style head shape."""

    d_in: int = 1152
    d_z: int = 256
    n_tok: int = 16
    width: int = 1024
    depth: int = 4
    heads: int = 16
    d_out: int = 7680
    grid: bool = False
    space: str = "state"    # "state": tokens appended to the encoder output; "embed": soft tokens at the encoder input;
                            # "vocab": per-position queries scored against the vocabulary, soft tokens = softmax @ E
                            # "ar": autoregressive describer (StyleDescriber), soft tokens along its greedy path
    tau: float = 1.0        # vocab / ar: softmax temperature of the soft tokens (0 = argmax text)
    n_mem: int = 4          # ar: memory tokens the code z is expanded into
    dropout: float = 0.0    # ar: decoder dropout (the describer memorises its 10k training style texts)
    out_scale: float = 1.0  # gain on the output layer: adam moves every output channel ~lr * fan-in per step, so the
                            # token step size is set by this gain (1.0 gave ~0.08 std per channel per step: r0-r2)


class StyleHead(nn.Module):
    """Pooled style feature -> posterior over z -> n_tok conditioning tokens (see module docstring)."""

    def __init__(self, cfg: HeadCfg, text_mean: Tensor, text_std: Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.enc = nn.Sequential(nn.LayerNorm(cfg.d_in), nn.Linear(cfg.d_in, d), nn.GELU(), nn.Linear(d, 2 * cfg.d_z))
        self.grid_in = nn.Sequential(nn.LayerNorm(cfg.d_in), nn.Linear(cfg.d_in, d)) if cfg.grid else None
        self.z_in = nn.Linear(cfg.d_z, d)
        self.q = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.null = nn.Parameter(torch.zeros(cfg.n_tok, cfg.d_out))
        self.gain = nn.Parameter(torch.ones(cfg.n_tok, cfg.d_out))  # scales the style-specific part (stage b)
        self.register_buffer("text_mean", text_mean.float().clone())
        self.register_buffer("text_std", text_std.float().clone())

    def posterior(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """(mu, logvar), each (..., d_z)."""
        mu, logvar = self.enc(pooled).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    def tokens(self, z: Tensor, grid: Tensor | None = None) -> Tensor:
        """(B, d_z) codes (+ (B, 16, d_in) grid) -> (B, n_tok, d_out) tokens in text-state units."""
        x = self.q[None] + self.z_in(z)[:, None]
        if self.grid_in is not None and grid is not None:
            x = torch.cat([self.grid_in(grid), x], dim=1)
        h = self.body(x)[:, -self.cfg.n_tok:]
        o = F.linear(self.norm(h), self.out.weight) * self.gain + self.out.bias
        return self.text_mean + self.text_std * self.cfg.out_scale * o

    def null_tokens(self, b: int) -> Tensor:
        """(b, n_tok, d_out) learned 'no style' tokens."""
        return (self.text_mean + self.text_std * self.null).expand(b, self.cfg.n_tok, -1)


class StyleDescriber(nn.Module):
    """Autoregressive "describer": posterior over z (as StyleHead) -> n_mem memory tokens -> causal transformer that
    writes the style text, reading and writing tokens through the frozen qwen3 input-embedding matrix E (passed in,
    not stored). fixes the position misalignment of the non-autoregressive vocab head. soft tokens are
    softmax(logits / tau) @ E along the greedy path (tau 0 = the generated text itself); z stays the cartridge."""

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


def kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean over the batch of KL(N(mu, var) || N(0, 1)) summed over dims."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()


def text_stats(states: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Token origin and scale for the head: per-channel mean of *pad* states (klein always sees prompts padded to 512,
    so a pad-like token is close to a no-op) and per-channel std of real prompt tokens. states (B, L, D), mask (B, L)."""
    m = mask.bool()
    return states[~m].float().mean(0), states[m].float().std(0).clamp_min(1e-3)


def load_head(path: str, dev: str = "cuda") -> nn.Module:
    """Rebuild a StyleHead / StyleDescriber from a checkpoint (shapes of the text-stat buffers from the checkpoint)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = HeadCfg(**ck["cfg"])
    sd = ck["head"]
    cls = StyleDescriber if cfg.space == "ar" else StyleHead
    head = cls(cfg, torch.zeros_like(sd["text_mean"]), torch.ones_like(sd["text_std"])).to(dev)
    miss, extra = head.load_state_dict(sd, strict=False)
    assert set(miss) <= {"gain"} and not extra, (miss, extra)  # heads saved before the gain existed
    return head.eval()
