"""Chameleon inference for apps (the hugging face space in space/, notebooks): style cartridges from images, random or
mixed codes, rendered as describer text or soft tokens for qwen-image 2.1 turbo, t2i and photo edits, and the optional
texture boost (the reference's vae latents visible to the target only in blocks 12-19).

every model resident on one gpu (>= 40 gb, or zerogpu), else swapped per phase: the vl encoder and the dit (16 + 13 gb)
take turns on the gpu, the small parts (style backbone, head, qwen3 embeddings) stay.
"""

import gc
import typing as tp

import torch
from PIL import Image

from chameleon import cartridge as cg
from chameleon import qimg, style
from chameleon.qport import model as qm
from chameleon.qport.dit import Hooks

TAU: float = 1.5
INSTR: str = "Change the style of this image to:"
NEUTRAL: str = "Change the style of this image."
T2I: str = "{p}, in the visual style of <image1> (its colours, medium, texture and lighting), not its content."
EDIT: str = ("Redraw <image1> in the visual style of <image2>: its colours, medium, texture and lighting. Keep the "
             "content and composition of <image1>; take nothing else from <image2>.")
WINDOW: tuple[int, int] = (12, 19)
REF, OUT = (32, 32), (64, 64)  # latent tokens: reference at 512 px, output / photo at 1024 px


def square(im: Image.Image, side: int) -> Image.Image:
    """Centre square crop resized to side."""
    w, h = im.size
    s = min(w, h)
    return im.convert("RGB").crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize(
        (side, side), Image.Resampling.LANCZOS)


class Pipeline:
    """All of chameleon's inference models; `resident=None` decides from the gpu's memory."""

    def __init__(self, head: str, prior: str | None, resident: bool | None = None, head_name: str = "chameleon-ar-head") -> None:
        """prior: the head's random-code prior (code_prior.pt), or None for a head without one: no random()."""
        if resident is None:
            resident = torch.cuda.get_device_properties(0).total_memory >= 40 * 2**30
        self.resident, self.head_name = resident, head_name
        self.head = style.load_head(head)
        self.bb = style.load_backbone()
        self.prior = cg.Prior.load(prior) if prior else None
        self.enc, self.proc = qm.load_encoder()
        self.evl = getattr(self.enc.model, "language_model", self.enc.model).embed_tokens.weight
        self.e4 = self.table(self.head)  # the table the head writes through
        self.eos = self.proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if not resident:
            self.enc.to("cpu")
            self._free()
        self.core = qm.load_core()
        if not resident:
            self.core.vae.enable_tiling()  # 1024 px decode next to the dit on 24 gb
            self.core.dit.to("cpu")
            self._free()
        self.on = "both" if resident else "none"

    @staticmethod
    def _free() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def _phase(self, which: str) -> None:
        """Put the encoder ("enc") or the dit ("dit") on the gpu (no-op when resident)."""
        if self.resident or self.on == which:
            return
        (self.core.dit if which == "enc" else self.enc).to("cpu")
        self._free()
        (self.enc if which == "enc" else self.core.dit).to(qm.DEV)
        self.on = which

    # ------------------------------------------------------------------------------------------------ cartridges
    def table(self, head: tp.Any) -> torch.Tensor:
        """The token-embedding table a head was trained with: Qwen-Image's own text encoder (4096-d) or Qwen3-4B's."""
        return self.evl.detach().clone() if head.cfg.d_out == self.evl.shape[1] else qimg.qwen3_embeddings()

    def make(self, images: list[Image.Image], names: list[str] | None = None) -> cg.Cartridge:
        names = names or [f"ref{i}" for i in range(len(images))]
        return cg.make(self.head, self.bb, [im.convert("RGB") for im in images], names, self.head_name)

    def random(self, seed: int) -> cg.Cartridge:
        assert self.prior is not None, f"no random-code prior for head {self.head_name}"
        return self.prior.sample(self.head_name, seed)

    @staticmethod
    def mix(carts: list[cg.Cartridge], weights: list[float]) -> cg.Cartridge:
        return cg.mix(carts, weights)

    # ------------------------------------------------------------------------------------------------ generation
    @torch.no_grad()
    def run(self, cart: cg.Cartridge, prompts: list[str], photos: list[Image.Image], ref: Image.Image | None = None,
            seed: int = 7, render: str = "text", texture: bool = True, strength: float = 1.0, plain: bool = True,
            ) -> tuple[str, list[tuple[str, Image.Image]]]:
        """The describer's text for the cartridge and labelled images: per prompt a t2i (+ a texture-boosted one when
        `texture` and a reference image are given), per photo an edit (+ texture). photos are cropped to 1024 px.
        plain=False with texture: the texture-boosted images only."""
        boost = texture and ref is not None
        plain = plain or not boost
        photos = [square(p, 1024) for p in photos]
        grey = Image.new("RGB", (512, 512), (128, 128, 128))
        self._phase("enc")
        soft, said = cg.render(self.head, self.e4, self.evl, self.eos, cart, TAU, self.proc.tokenizer)
        enc = lambda prompt, ims=(): qm.encode(self.enc, self.proc, prompt, list(ims))  # noqa: E731
        jobs: list[tuple[str, str, int, tp.Any]] = []
        for i, p in enumerate(prompts):
            if plain:
                if render == "soft":
                    t = qimg.encode_soft(self.enc, self.proc, f"{p}." + qimg.PLACE * len(soft), soft)
                else:
                    t = enc(f"{p}.{said}")
                jobs.append((p, "t2i", i, t))
            if boost:
                jobs.append((p, "t2i + texture", i, enc(f"{T2I.format(p=p)}{said}", [grey])))
        neutral = [enc(NEUTRAL, [ph]) for ph in photos]
        for i, ph in enumerate(photos):
            if plain:
                if render == "soft":
                    t = qimg.encode_soft(self.enc, self.proc, INSTR + qimg.PLACE * len(soft), soft, [ph])
                else:
                    t = enc(INSTR + said, [ph])
                jobs.append((f"photo {i + 1}", "edit", i, t))
            if boost:
                jobs.append((f"photo {i + 1}", "edit + texture", i, enc(EDIT + said, [ph, grey])))
        cpu = lambda t: (t[0].cpu(), t[1].cpu())  # noqa: E731
        jobs = [(a, b, c, cpu(t)) for a, b, c, t in jobs]
        neutral = [cpu(t) for t in neutral]

        self._phase("dit")
        dev = lambda t: (t[0].to(qm.DEV), t[1].to(qm.DEV))  # noqa: E731
        core = self.core
        lat_r = qm.to_latents(core, [square(ref, 512)]) if boost else None
        lat_ph = [qm.to_latents(core, [ph]) for ph in photos]
        out = []
        for label, kind, i, t in jobs:
            if kind == "t2i":
                im = qimg.t2i(core, dev(t), seed)
            elif kind == "t2i + texture":
                im = self.window(dev(t), [(lat_r, REF)], seed)
            elif kind == "edit" and strength != 1.0:
                im = qm.edit_mix(core, [dev(neutral[i]), dev(t)], [1 - strength, strength], lat_ph[i], OUT, seed)
            elif kind == "edit":
                im = qm.edit(core, dev(t), lat_ph[i], OUT, seed)
            else:
                im = self.window(dev(t), [(lat_ph[i], OUT), (lat_r, REF)], seed)
            out.append((f"{label} · {kind}", im))
        return said.strip(), out

    @torch.no_grad()
    def window(self, text: qm.Text, refs: list[tuple[torch.Tensor, tuple[int, int]]], seed: int) -> Image.Image:
        """Turbo generation with the last reference visible to the target only in WINDOW: t2i refs = [style ref];
        edit refs = [photo, style ref]."""
        states, slots = text
        dit = self.core.dit
        sig = torch.tensor([*qm.TURBO, 0.0], device=qm.DEV)
        levels = (sig * 1000).to(torch.bfloat16) / 1000
        lat = qm.noise(OUT, seed)
        hooks = Hooks(ref_window=WINDOW, window_last=True)
        lay = qm.layout(slots, [s for _, s in refs] + [OUT], dit.cfg)
        v, cache = dit.prefill(torch.cat([r for r, _ in refs] + [lat], dim=1), states, levels[:1], lay, hooks)
        lat = qm.euler(lat, v, sig[0], sig[1])
        for i in range(1, len(qm.TURBO)):
            lat = qm.euler(lat, dit.decode(lat, levels[i:i + 1], lay, cache, i, hooks), sig[i], sig[i + 1])
        del cache, v
        return qm.to_image(self.core, lat, OUT)
