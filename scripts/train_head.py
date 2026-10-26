"""Train chameleon's describer head: MegaStyle-encoder features of a style's images -> that style's text label,
written autoregressively through Qwen3's vocabulary (teacher-forced cross-entropy). No image model is involved.

1. features (one cache per source; only images whose style has a label are kept):
   python scripts/train_head.py features --shard train-00001.parquet --out data/s00001   # tencent/MegaStyle-1.4M
   python scripts/train_head.py features --shard train-00099.parquet --out data/ev99     # held-out eval styles
   python scripts/train_head.py features --mj --out data/mj                              # midjourney srefs
2. train (the released head: s00001-s00004 + mj at 3x, eval on ev99):
   python scripts/train_head.py train --data data/s0000{1,2,3,4} data/mj --mult 1 1 1 1 3 --eval data/ev99 --out runs/head

labels come from the tensorkelechi/chameleon-style-labels dataset (or --labels DIR with megastyle/*.jsonl and
midjourney/*.jsonl). the output head_pre.pt loads with chameleon.style.load_head and plugs into chameleon.serve.
"""

import argparse
import dataclasses as dc
import io
import json
import pathlib as pl
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from chameleon import data, style

LABELS: str = "tensorkelechi/chameleon-style-labels"
MEGASTYLE: str = "tencent/MegaStyle-1.4M"
MJ: str = "peteromallet/high-quality-midjouney-srefs"
QWEN3: str = "Qwen/Qwen3-4B"  # the describer reads and writes text through its tokenizer and input embeddings
TABLES: dict[str, tuple[str, str]] = {"qwen3": (QWEN3, ""), "qwen3-vl": ("Qwen/Qwen-Image-2.1-Turbo", "text_encoder/")}


def load_labels(src: str | None) -> dict[str, str]:
    """style_key -> label text, from a local dir or the hub dataset."""
    from huggingface_hub import snapshot_download

    root = pl.Path(src) if src else pl.Path(snapshot_download(LABELS, repo_type="dataset", allow_patterns=["*.jsonl"]))
    out = {}
    for f in sorted(root.glob("*/*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if d.get("text"):
                out[d["style_key"]] = d["text"]
    return out


def write_cache(out: pl.Path, rows: list[dict], pooled: list[np.ndarray], grid: list[np.ndarray]) -> None:
    """chameleon.data cache layout: rows.jsonl, pooled.npy (N, 1152), grid.npy (N, 16, 1152), progress.json."""
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pooled.npy", np.concatenate(pooled).astype(np.float16))
    np.save(out / "grid.npy", np.concatenate(grid).astype(np.float16))
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "progress.json").write_text(json.dumps({"done": len(rows)}))


def opened(b: bytes) -> Image.Image | None:
    """Decoded rgb image, or None for an unreadable one."""
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:  # noqa: BLE001 - any decoder error drops the row
        return None


def features(a: argparse.Namespace) -> None:
    """One MegaStyle shard (or the midjourney set) -> a feature cache with the labels as style text."""
    from huggingface_hub import hf_hub_download, snapshot_download

    labels = load_labels(a.labels)
    bb = style.load_backbone()
    rows, images, pooled, grid = [], [], [], []

    def flush() -> None:
        """Features of the pending images (so only one batch of decoded images is ever held)."""
        if images:
            p, g = style.features(bb, style.preprocess(images))
            pooled.append(p.cpu().numpy().astype(np.float16)), grid.append(g.cpu().numpy().astype(np.float16))
            images.clear()

    def add(img: dict | bytes, key: str, rid: str, content: str) -> None:
        if key in labels and (im := opened(img["bytes"] if isinstance(img, dict) else img)) is not None:
            rows.append({"i": len(rows), "id": rid, "style_key": key, "content": content or "", "style": labels[key]})
            images.append(im)
            if len(images) == 32:
                flush()

    if a.mj:
        root = pl.Path(snapshot_download(MJ, repo_type="dataset", allow_patterns=["data/*.parquet"]))
        for f in sorted(root.glob("data/*.parquet")):
            t = pq.read_table(f, columns=["image", "sref", "prompt", "filename"])
            for img, sref, prompt, fn in zip(*(t.column(c).to_pylist() for c in ("image", "sref", "prompt", "filename"))):
                add(img, f"mj_{sref}", fn, prompt)
    else:
        pf = pq.ParquetFile(hf_hub_download(MEGASTYLE, a.shard, repo_type="dataset"))
        for b in pf.iter_batches(batch_size=256, columns=["id", "content", "image"]):  # streamed, never the whole shard
            for rid, content, img in zip(*(b.column(c).to_pylist() for c in ("id", "content", "image"))):
                add(img, rid.split("_")[0], rid, content)
    flush()
    print(f"{len(rows)} images, {len({r['style_key'] for r in rows})} labelled styles", flush=True)
    write_cache(a.out, rows, pooled, grid)


def embeddings(dev: str, table: str = "qwen3") -> torch.Tensor:
    """An input-embedding matrix alone (qwen3: Qwen3-4B's (151936, 2560); qwen3-vl: Qwen-Image 2.1's text encoder,
    (151936, 4096)); only the safetensors file holding it is downloaded."""
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open

    repo, sub = TABLES[table]
    api = HfApi()
    for f in sorted(x for x in api.list_repo_files(repo) if x.startswith(sub) and x.endswith(".safetensors")):
        keys = [k for k in api.parse_safetensors_file_metadata(repo, f).tensors if k.endswith("embed_tokens.weight")]
        if keys:
            with safe_open(hf_hub_download(repo, f), framework="pt") as h:
                return h.get_tensor(keys[0]).to(dev, torch.bfloat16)
    raise KeyError(f"no embed_tokens.weight in {repo}/{sub}")


def train(a: argparse.Namespace) -> None:
    """Teacher-forced cross-entropy of the label text given the pooled code of 1-3 same-style references; keeps the
    best held-out weights."""
    from transformers import AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng, gen = np.random.default_rng(a.seed), torch.Generator("cpu").manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    tok = AutoTokenizer.from_pretrained(QWEN3)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    emb = embeddings(dev, a.emb)
    caches = [data.open_cache(p, lat=False) for p in a.data]
    ev = data.open_cache(a.eval, lat=False)
    mult = a.mult or [1.0] * len(caches)
    w = np.array([len(c.groups) * m for c, m in zip(caches, mult)])  # sample sources by style count x mult
    d = emb.shape[1]
    cfg = style.HeadCfg(d_z=a.d_z, n_tok=a.n_tok, depth=a.depth, space="ar", d_out=d, tau=a.tau, dropout=a.dropout,
                        grid=a.grid)
    head = style.StyleDescriber(cfg, torch.zeros(d), torch.ones(d)).to(dev)
    print(f"head {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params; styles {w.astype(int).tolist()}; "
          f"eval styles {len(ev.groups)}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100) * max(0.05, 1 - i / a.steps))

    def ids(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs (B, n_tok) and targets (B, n_tok + 1): " " + text truncated to n_tok, then eos; -100 after eos."""
        r = [tok(" " + t, add_special_tokens=False).input_ids[:a.n_tok] for t in texts]
        inp = torch.tensor([x + [eos] * (a.n_tok - len(x)) for x in r], device=dev)
        tgt = torch.tensor([x + [eos] + [-100] * (a.n_tok - len(x)) for x in r], device=dev)
        return inp, tgt

    def loss(c: data.Cache, bt: data.Batch, noise: bool, use_grid: bool = True) -> torch.Tensor:
        mu, lv = head.posterior(bt.pooled.to(dev))
        z, _ = style.pool_posterior(mu, lv, bt.mask.to(dev))  # one code from all references
        if noise:  # fixed jitter instead of a learned variance
            z = z + a.z_noise * torch.randn(z.shape, generator=gen).to(dev)
        grid = keep = None
        if a.grid and use_grid:  # mean feature grid of the references; dropped for some samples while training
            m = bt.mask.to(dev).float()
            grid = (m[..., None, None] * bt.grid.to(dev)).sum(1) / m.sum(1).clamp_min(1)[:, None, None]
            if noise:
                keep = torch.rand(z.shape[0], generator=gen).to(dev) >= a.p_grid_drop
        inp, tgt = ids([c.rows[t]["style"] for t in bt.target])
        return F.cross_entropy(head.logits(z, inp, emb, grid, keep).flatten(0, 1), tgt.flatten(), ignore_index=-100)

    t0, run, best = time.time(), [], (float("inf"), 0, None)
    for step in range(a.steps):
        c = caches[int(rng.choice(len(caches), p=w / w.sum()))]
        lo = loss(c, data.sample_batch(c, a.bs, rng, r_max=3, p_cap=0.0, p_style=0.0), True)
        lo.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        run.append(lo.item())
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            head.eval()
            with torch.no_grad():
                erng = np.random.default_rng(99)
                ev_l = float(np.mean([loss(ev, data.sample_batch(ev, a.bs, erng, r_max=3, p_cap=0.0, p_style=0.0),
                                           False).item() for _ in range(16)]))
            head.train()
            if ev_l < best[0]:
                best = (ev_l, step + 1, {k: v.detach().clone() for k, v in head.state_dict().items()})
            print(f"step {step + 1} train_ce {np.mean(run):.4f} eval_ce {ev_l:.4f} best {best[0]:.4f}@{best[1]} {time.time() - t0:.0f}s", flush=True)
            run = []
    head.load_state_dict(best[2])
    if a.grid:  # the same head without its grid (what random and grid-less codes get)
        head.eval()
        with torch.no_grad():
            erng = np.random.default_rng(99)
            ng = float(np.mean([loss(ev, data.sample_batch(ev, a.bs, erng, r_max=3, p_cap=0.0, p_style=0.0), False,
                                     False).item() for _ in range(16)]))
        print(f"best weights without the grid: eval_ce {ng:.4f}", flush=True)
    torch.save({"head": head.state_dict(), "cfg": dc.asdict(cfg), "step": best[1]}, a.out / "head_pre.pt")
    print(f"saved {a.out / 'head_pre.pt'} (best eval ce {best[0]:.4f} at step {best[1]})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--shard", help="a MegaStyle-1.4M parquet file, e.g. train-00001.parquet")
    f.add_argument("--mj", action="store_true", help="the midjourney sref set instead of a MegaStyle shard")
    f.add_argument("--labels", default=None, help="local labels dir (default: the hub dataset)")
    f.add_argument("--out", type=pl.Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=pl.Path, nargs="+", required=True)
    t.add_argument("--mult", type=float, nargs="*", default=[], help="per-cache sampling weight (x its style count)")
    t.add_argument("--eval", type=pl.Path, required=True)
    t.add_argument("--out", type=pl.Path, required=True)
    t.add_argument("--steps", type=int, default=4500)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--n-tok", type=int, default=144, help="longest description, in tokens")
    t.add_argument("--d-z", type=int, default=256)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--tau", type=float, default=1.5, help="soft-token temperature stored with the head")
    t.add_argument("--z-noise", type=float, default=0.05)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--grid", action="store_true", help="also read the references' 4x4 feature grid")
    t.add_argument("--p-grid-drop", type=float, default=0.2, help="grid dropout (the head must also work without one)")
    t.add_argument("--emb", choices=tuple(TABLES), default="qwen3", help="token-embedding table the head writes through")
    a = ap.parse_args()
    if a.cmd == "features":
        assert a.mj or a.shard, "--shard or --mj"
        features(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
