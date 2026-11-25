"""Content-whitened supervised reduced-rank regression from image features to description states (v2 stage 2).

one image's features phi = [e, rff(e)] (d = 1152 + 1024) predict its style's flattened description block y (K * D).
streaming gives EtY = sum_s S_s y_s^T (S_s = the style's summed features) and sum_y; the image-side gram EtE is fixed.
because EtE sums over every image (between- and within-style), the ridge solution whitens away the content
directions that vary inside a style. the linear encoder is the leading 1152 block; the nonlinear one uses all d.

eigendecomposition EtE_c = Q L Q^T gives every ridge solution W_lam = Q diag(1 / (L + lam)) A with A = Q^T EtY_c, and
the rank-r rrr basis is the top right singular vectors of C^(1/2) W (C = EtE_c / n images): V_r = T A with
T = U_r^T diag(sqrt(L / n) / (L + lam)) / S_r. held-out errors only need A A^T and A Y_h^T, never a full prediction.
"""

import dataclasses as dc

import torch
from torch import Tensor

LINEAR = 1152


@dc.dataclass(frozen=True)
class Gram:
    """Centred image-side statistics: EtE_c (d, d) fp64, feature mean (d,), image count."""

    ete: Tensor
    mean: Tensor
    n: int


@dc.dataclass(frozen=True)
class Eig:
    """Eigenbasis of a centred gram (fp64): q (d, d), l (d,), and a = q^T EtY_c (d, KD) fp32."""

    q: Tensor
    l: Tensor
    a: Tensor


@dc.dataclass(frozen=True)
class Fit:
    """Rank-r rrr result: basis (r, KD), sd (r,) of the predicted code, encoder (r, d) applied to phi - mean."""

    basis: Tensor
    sd: Tensor
    encode: Tensor
    lam: float


def centre_gram(gram: Tensor, total_mean: Tensor, n: int, dims: int | None = None) -> Gram:
    """EtE_c = EtE - n m m^T on the first `dims` features."""
    d = dims or gram.shape[0]
    g, m = gram[:d, :d].double(), total_mean[:d].double()
    return Gram(g - n * torch.outer(m, m), m, n)


def centre_ety(ety: Tensor, sum_y: Tensor, styles: int, mean: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """EtY_c = EtY - n m ybar^T in place (each style has n / styles images); returns (EtY_c, ybar)."""
    ybar = sum_y / styles
    ety.addr_(mean.to(ety), ybar.to(ety), alpha=-float(n))
    return ety, ybar


def eig(g: Gram, ety_c: Tensor, dims: int | None = None) -> Eig:
    """Eigendecomposition of the gram and the rotated cross-covariance."""
    d = dims or g.ete.shape[0]
    l, q = torch.linalg.eigh(g.ete[:d, :d])
    l = l.clamp_min(0)
    a = q.T.to(ety_c) @ ety_c[:d]
    return Eig(q, l, a)


def gram_of(a: Tensor) -> Tensor:
    """A A^T in fp32 (sums over the long axis), returned as fp64."""
    return (a @ a.T).double()


def lam_grid(e: Eig, rel: tuple[float, ...]) -> list[float]:
    """Ridge strengths relative to the mean eigenvalue of EtE_c."""
    return [float(r * e.l.mean()) for r in rel]


def holdout_r2(e: Eig, mean: Tensor, phi: Tensor, owner: Tensor, ay: Tensor, yy: Tensor, lams: list[float],
               aat: Tensor | None = None) -> dict[float, float]:
    """Single-image held-out R^2 of the ridge prediction for each lam, from A A^T (d, d) and ay = A (Y_h - ybar)^T
    (d, styles); phi (images, d) held-out features, owner (images,) the style row of each image, yy = sum over
    held-out images of ||y - ybar||^2."""
    aat = gram_of(e.a) if aat is None else aat
    x = (phi.double() - mean.double()) @ e.q
    out = {}
    for lam in lams:
        b = x / (e.l + lam)
        pred2 = ((b @ aat) * b).sum()
        cross = (b * ay.double().T[owner]).sum()
        out[lam] = float(1 - (yy - 2 * cross + pred2) / yy)
    return out


def rrr(e: Eig, n: int, lam: float, rank: int, cols: slice = slice(None)) -> tuple[Tensor, Tensor]:
    """T (rank, d) and singular values S (rank,) of C^(1/2) W_lam restricted to the columns `cols` of the block, so
    that V_r = T @ A[:, cols]."""
    dg = torch.sqrt(e.l / n) / (e.l + lam)
    aat = gram_of(e.a[:, cols])
    g = dg[:, None] * aat * dg[None]  # Q^T (C^(1/2) W)(C^(1/2) W)^T Q
    s2, u = torch.linalg.eigh(g)
    s2, u = s2.flip(0)[:rank].clamp_min(1e-30), u.flip(1)[:, :rank]
    s = s2.sqrt()
    t = (u.T * dg[None]) / s[:, None]
    return t, s


def basis(e: Eig, t: Tensor, cols: slice = slice(None)) -> Tensor:
    """Orthonormal rows V_r = T A (rank, KD)."""
    return (t.to(e.a) @ e.a[:, cols]).contiguous()


def encoder(e: Eig, lam: float, v: Tensor, cols: slice = slice(None)) -> Tensor:
    """(rank, d) linear map from centred features to codes: (W_lam V_r^T)^T with W_lam = Q diag(1/(L+lam)) A."""
    proj = (e.a[:, cols] @ v.T).double()  # (d, rank)
    return ((e.q / (e.l + lam)[None]) @ proj).T.float()


def fit(e: Eig, n: int, lam: float, rank: int, cols: slice = slice(None)) -> Fit:
    """Basis, code sd and the matching encoder of one ridge solution."""
    t, s = rrr(e, n, lam, rank, cols)
    v = basis(e, t, cols)
    return Fit(v, s.float(), encoder(e, lam, v, cols), lam)


def accumulate(ety: Tensor, sum_y: Tensor, sums: Tensor, y: Tensor) -> None:
    """EtY += S^T y and sum_y += sum of y rows, in fp32 on the accumulators' device."""
    y = y.to(ety.device, torch.float32)
    ety.addmm_(sums.to(ety).T, y)
    sum_y.add_(y.sum(0))
