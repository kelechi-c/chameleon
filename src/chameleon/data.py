"""MegaStyle feature caches and the same-style / different-content pair sampler.

a cache dir (written by scripts/prep.py) holds, for N rows of one parquet shard:
  rows.jsonl   {"i", "id", "style_key", "content", "style"} per row (style_key = the "s<k>" id prefix)
  pooled.npy   (N, 1152) f16 backbone pooled features
  grid.npy     (N, 16, 1152) f16 backbone 4x4 patch grid
  lat.npy      (N, 1024, 128) f16 packed normalised klein latents of the 512² image
  img/<id>.jpg (eval caches only) the images
"""

import dataclasses as dc
import json
import pathlib as pl

import numpy as np
import torch
from torch import Tensor


@dc.dataclass(frozen=True)
class Cache:
    """Memory-mapped features of one cache dir plus the style groups (row indices sharing a style_key)."""

    root: pl.Path
    rows: list[dict]
    pooled: np.ndarray
    grid: np.ndarray
    lat: np.ndarray | None
    groups: list[np.ndarray]
    group_of: np.ndarray
    eligible: np.ndarray


def open_cache(root: pl.Path, lat: bool = True) -> Cache:
    """Open a cache dir, keeping only rows prep has finished (progress.json), so a partial cache is usable; rows of
    styles with < 2 images are kept but never sampled as targets."""
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    done = json.loads((root / "progress.json").read_text())["done"] if (root / "progress.json").exists() else len(rows)
    rows = rows[:done]
    n = len(rows)
    bad = set(json.loads((root / "bad.json").read_text())) if (root / "bad.json").exists() else set()
    keys: dict[str, list[int]] = {}
    for r in rows:
        if r["i"] not in bad:  # unreadable images (prep marks them) are never targets or references
            keys.setdefault(r["style_key"], []).append(r["i"])
    groups = [np.asarray(v) for v in keys.values()]
    group_of = np.full(n, -1, dtype=np.int64)
    for g, idx in enumerate(groups):
        group_of[idx] = g
    eligible = np.asarray([i for i in range(n) if group_of[i] >= 0 and len(groups[group_of[i]]) >= 2], dtype=np.int64)
    mm = lambda name: np.lib.format.open_memmap(root / name, mode="r")[:n]  # noqa: E731
    return Cache(root, rows, mm("pooled.npy"), mm("grid.npy"), mm("lat.npy") if lat else None, groups, group_of,
                 eligible)


@dc.dataclass(frozen=True)
class Batch:
    """One training batch: targets, their captions, and up to r_max same-style references per target."""

    lat: Tensor | None   # (B, 1024, 128) f32 target latents (None for feature-only caches)
    caps: list[str]      # content-only captions ("" when dropped)
    pooled: Tensor       # (B, R, 1152) reference features
    grid: Tensor         # (B, R, 16, 1152)
    mask: Tensor         # (B, R) valid references
    drop_style: Tensor   # (B,) bool, use null tokens
    target: np.ndarray   # (B,) row indices


def sample_batch(c: Cache, b: int, rng: np.random.Generator, r_max: int = 3, p_cap: float = 0.1,
                 p_style: float = 0.1) -> Batch:
    """Targets uniform over rows of multi-image styles; refs = 1..r_max other rows of the target's style."""
    tgt = rng.choice(c.eligible, size=b, replace=False)
    ref_idx = np.zeros((b, r_max), dtype=np.int64)
    mask = np.zeros((b, r_max), dtype=bool)
    for j, t in enumerate(tgt):
        others = c.groups[c.group_of[t]]
        others = others[others != t]
        n = int(rng.integers(1, min(r_max, len(others)) + 1))
        pick = rng.choice(others, size=n, replace=False)
        ref_idx[j, :n], mask[j, :n] = pick, True
    flat = ref_idx.reshape(-1)
    pooled = torch.from_numpy(np.asarray(c.pooled[flat], dtype=np.float32)).view(b, r_max, -1)
    grid = torch.from_numpy(np.asarray(c.grid[flat], dtype=np.float32)).view(b, r_max, 16, -1)
    lat = None
    if c.lat is not None:
        lat = torch.from_numpy(np.asarray(c.lat[np.sort(tgt)], dtype=np.float32))
        order = np.argsort(np.argsort(tgt))  # memmap reads are faster sorted; restore the target order
        lat = lat[torch.from_numpy(order)]
    caps = ["" if rng.random() < p_cap else c.rows[t]["content"] for t in tgt]
    drop = torch.from_numpy(rng.random(b) < p_style)
    return Batch(lat, caps, pooled, grid, torch.from_numpy(mask), drop, tgt)


def shuffle_refs(bt: Batch, rng: np.random.Generator) -> Batch:
    """The same batch with references taken from another target in the batch (different style): style-info probe."""
    perm = torch.from_numpy(np.roll(np.arange(len(bt.target)), int(rng.integers(1, len(bt.target)))))
    return dc.replace(bt, pooled=bt.pooled[perm], grid=bt.grid[perm], mask=bt.mask[perm])
