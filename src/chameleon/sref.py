"""Runtime of a v2 style cartridge fitted by chameleon.rrr: reference image or --sref code -> style block.

an npz holds mu (K, D), an orthonormal basis (r, K, D), the predicted-code sd (r,), the linear and nonlinear encoders
(r, 1152) and (r, 2176), the feature mean and the random-fourier-feature parameters, plus the chosen settings
(config: rank, alpha, calibrate, encoder). image: z = encode (phi(e) - mean), optionally rescaled to the expected
norm of a random code (STYLE_SIZE = sqrt(sum sd^2)); code: z = N(0, 1)^r * sd from np.random.default_rng(code);
block = mu + alpha * z @ basis. the dit and its weights are never touched.
"""

import dataclasses as dc
import json
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

CODES: int = 1_000_000


@dc.dataclass(frozen=True)
class Settings:
    """Inference settings saved with a cartridge."""

    rank: int = 128
    alpha: float = 2.0
    calibrate: bool = True
    encoder: str = "lin"  # lin | nl


@dc.dataclass(frozen=True)
class Sref:
    """A fitted cartridge (torch tensors, fp32 except the fp16 basis)."""

    mu: Tensor
    basis: Tensor
    sd: Tensor
    encode_lin: Tensor
    encode_nl: Tensor
    phi_mean: Tensor
    rff_w: Tensor
    rff_b: Tensor
    rff_scale: float
    space: str
    settings: Settings


def load(path: str | Path, device: str = "cpu") -> Sref:
    """Load a fitted v2 style cartridge from an npz file."""
    with np.load(path) as f:
        t = {k: torch.from_numpy(f[k]).to(device) for k in
             ("mu", "basis", "sd", "encode_lin", "encode_nl", "phi_mean", "rff_w", "rff_b")}
        meta = json.loads(str(f["config"]))
    known = {f.name for f in dc.fields(Settings)}
    return Sref(**t, rff_scale=float(meta["rff_scale"]), space=meta["space"],
                settings=Settings(**{k: v for k, v in meta.get("settings", {}).items() if k in known}))


def phi(s: Sref, e: Tensor) -> Tensor:
    """(n, 1152) pooled megastyle features -> (n, 2176) [e, rff(e)] in fp32."""
    e = e.float().to(s.rff_w.device)
    rff = torch.cos(e @ s.rff_w + s.rff_b) * (np.sqrt(2.0 / s.rff_w.shape[1]) * s.rff_scale)
    return torch.cat([e, rff], 1)


def style_size(s: Sref, rank: int) -> float:
    """Expected norm of a random rank-r code."""
    return float(s.sd[:rank].float().pow(2).sum().sqrt())


def z_image(s: Sref, e: Tensor, settings: Settings | None = None) -> Tensor:
    """Code of one style from the features of one or more of its images (codes averaged)."""
    c = settings or s.settings
    x = phi(s, e) - s.phi_mean.float()
    z = (x[:, :1152] @ s.encode_lin[:c.rank].T if c.encoder == "lin" else x @ s.encode_nl[:c.rank].T).mean(0)
    return calibrate(s, z) if c.calibrate else z


def calibrate(s: Sref, z: Tensor) -> Tensor:
    """Rescale a code to STYLE_SIZE."""
    z = z.to(s.sd.device, torch.float32)
    return z * (style_size(s, len(z)) / z.norm().clamp_min(1e-8))


def z_code(s: Sref, code: int, rank: int | None = None) -> Tensor:
    """Deterministic code for an integer --sref."""
    r = rank or s.settings.rank
    g = np.random.default_rng(code).standard_normal(r, dtype=np.float32)
    return torch.from_numpy(g).to(s.sd.device) * s.sd[:r].float()


def block(s: Sref, z: Tensor, alpha: float | None = None) -> Tensor:
    """(K, D) fp32 style block mu + alpha * z @ basis[:len(z)]."""
    a = s.settings.alpha if alpha is None else alpha
    flat = z.to(s.basis.device, torch.float32) @ s.basis[:len(z)].flatten(1).float()  # fp32 product of the fp16 basis
    return s.mu.float() + a * flat.view(s.mu.shape)


def nearest_codes(s: Sref, z: Tensor, codes: int = CODES, chunk: int = 8192) -> list[tuple[int, float]]:
    """Exact cosine search of each row of z (q, r) over the first `codes` integer codes."""
    r = z.shape[-1]
    sd = s.sd[:r].float().cpu().numpy()
    q = z.float().cpu().numpy().reshape(-1, r)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    best = np.full(len(q), -1)
    best_cos = np.full(len(q), -2.0)
    for start in range(0, codes, chunk):
        g = np.stack([np.random.default_rng(c).standard_normal(r, dtype=np.float32)
                      for c in range(start, min(codes, start + chunk))]) * sd
        cos = (g / np.linalg.norm(g, axis=1, keepdims=True)) @ q.T  # (chunk, q)
        j = cos.argmax(0)
        hit = cos[j, np.arange(len(q))] > best_cos
        best[hit], best_cos[hit] = start + j[hit], cos[j, np.arange(len(q))][hit]
    return [(int(c), float(x)) for c, x in zip(best, best_cos)]


def nearest_code(s: Sref, z: Tensor, codes: int = CODES) -> tuple[int, float]:
    """Exact cosine search of one code."""
    return nearest_codes(s, z[None], codes)[0]


@torch.no_grad()
def embed(bb: tp.Any, images: list) -> Tensor:
    """Pooled megastyle features of PIL images, preprocessed for the style encoder."""
    from chameleon import style

    return style.features(bb, style.preprocess(images))[0].float()
