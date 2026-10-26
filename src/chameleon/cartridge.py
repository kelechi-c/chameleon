"""Style cartridges: one or more reference images -> one describer code z (the posterior mean pooled over the
references), saved as a small file (~1 kb + metadata). a cartridge is mixed by weighted averaging of z, sampled at
random from a gaussian fitted to training posteriors (the "--sref random" analogue), and rendered for qwen-image as the
describer's soft tokens (softmax(logits / tau) @ E_qwen3vl) or as its greedy text.
"""

import dataclasses as dc
import pathlib as pl
import typing as tp

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from chameleon import style


@dc.dataclass
class Cartridge:
    """A style code and where it came from."""

    z: Tensor  # (d_z,) fp32, cpu
    head: str  # checkpoint the code belongs to (codes are not portable across heads)
    refs: list[str] = dc.field(default_factory=list)
    weights: list[float] = dc.field(default_factory=list)
    describe: str = ""  # the head's greedy text for z (filled by render)
    kind: str = "refs"  # refs | mix | random
    grid: Tensor | None = None  # (16, 1152) fp16 mean feature grid of the references, for heads that read one

    def save(self, path: pl.Path) -> None:
        torch.save(dc.asdict(self) | {"z": self.z.float().cpu(), "version": 1}, path)

    @staticmethod
    def load(path: pl.Path) -> "Cartridge":
        d = torch.load(path, map_location="cpu", weights_only=False)
        d.pop("version", None)
        return Cartridge(**d)


@torch.no_grad()
def make(head: tp.Any, bb: tp.Any, images: list[Image.Image], names: list[str], head_name: str,
         weights: list[float] | None = None) -> Cartridge:
    """Cartridge of a reference set: per-image posteriors pooled with style.pool_posterior (weighted mean of mu)."""
    pooled, grid = style.features(bb, style.preprocess([im.convert("RGB") for im in images]))
    mu, logvar = head.posterior(pooled.to(next(head.parameters()).device))
    w = torch.tensor(weights or [1.0] * len(images), device=mu.device)
    z, _ = style.pool_posterior(mu[None], logvar[None], torch.ones(1, len(images), device=mu.device), w[None])
    g = None
    if head.cfg.grid:
        wn = (w / w.sum()).to(grid.device)
        g = (wn[:, None, None] * grid).sum(0).half().cpu()
    return Cartridge(z[0].float().cpu(), head_name, list(names), [float(x) for x in w.tolist()], grid=g)


def mix(carts: list[Cartridge], weights: list[float]) -> Cartridge:
    """Weighted mix of cartridges of one head (weights normalised to sum 1)."""
    assert len({c.head for c in carts}) == 1, "cartridges of different heads cannot be mixed"
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    z = sum(wi * c.z for wi, c in zip(w, carts))
    g = sum(wi * c.grid.float() for wi, c in zip(w, carts)).half() if all(c.grid is not None for c in carts) else None
    return Cartridge(z, carts[0].head, [r for c in carts for r in c.refs], w.tolist(), kind="mix", grid=g)


class Prior:
    """Gaussian over a head's codes, fitted to the posterior means of training images."""

    def __init__(self, mean: Tensor, cov: Tensor) -> None:
        self.dist = torch.distributions.MultivariateNormal(mean.float(), cov.float() + 1e-4 * torch.eye(len(mean)))

    @staticmethod
    def load(path: pl.Path) -> "Prior":
        d = torch.load(path, map_location="cpu")
        return Prior(d["mean"], d["cov"])

    @staticmethod
    @torch.no_grad()
    def fit(head: tp.Any, pooled: np.ndarray, n: int = 4096, seed: int = 0) -> "Prior":
        """From training-image pooled features (N, 1152), e.g. a prep cache's pooled.npy."""
        idx = np.sort(np.random.default_rng(seed).choice(len(pooled), min(n, len(pooled)), replace=False))
        dev = next(head.parameters()).device
        mu, _ = head.posterior(torch.from_numpy(np.asarray(pooled[idx], np.float32)).to(dev))
        mu = mu.float().cpu()
        return Prior(mu.mean(0), torch.cov(mu.T))

    def save(self, path: pl.Path) -> None:
        torch.save({"mean": self.dist.loc, "cov": self.dist.covariance_matrix - 1e-4 * torch.eye(len(self.dist.loc))},
                   path)

    def sample(self, head_name: str, seed: int) -> Cartridge:
        torch.manual_seed(seed)
        return Cartridge(self.dist.sample(), head_name, [f"random:{seed}"], [], kind="random")


@torch.no_grad()
def render(head: tp.Any, e4: Tensor, evl: Tensor, eos: int, cart: Cartridge, tau: float, tokenizer: tp.Any
           ) -> tuple[Tensor, str]:
    """Qwen-image soft tokens (n, 4096) and the greedy text of a cartridge; fills cart.describe."""
    grid = None if cart.grid is None else cart.grid[None].to(e4.device, torch.float32)
    soft, ids = head.describe(cart.z[None].to(e4.device), e4, eos, tau, out_emb=evl, grid=grid)
    cart.describe = tokenizer.decode(ids[0])
    return soft[0], cart.describe
