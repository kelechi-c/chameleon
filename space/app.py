"""chameleon: style cartridges for Qwen-Image 2.1 Turbo (hugging face space). chameleon.serve does the work; this file is
the ui. the describer head is downloaded from the hub (HEAD_REPO); a code_prior.pt in that repo enables random styles.
CHAMELEON_DEFER=1 skips loading, for a caller that sets app.PIPE itself."""

import os
import pathlib as pl
import sys
import tempfile
import traceback

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, str(pl.Path(__file__).parent / "src"))
from chameleon import cartridge as cg  # noqa: E402
from chameleon import serve  # noqa: E402

try:
    import spaces
    gpu = spaces.GPU(duration=180)
except ImportError:  # not on a space
    def gpu(f):  # noqa: D103
        return f

HEAD_REPO = os.environ.get("CHAMELEON_HEAD", "tensorkelechi/chameleon-ar-head")
ZERO = bool(os.environ.get("SPACES_ZERO_GPU"))
PIPE, WHY = None, ""
if os.environ.get("CHAMELEON_DEFER"):  # the caller sets app.PIPE (the colab notebook reuses its pipeline)
    pass
elif ZERO or torch.cuda.is_available():
    try:
        from huggingface_hub import hf_hub_download
        try:
            prior = hf_hub_download(HEAD_REPO, "code_prior.pt")
        except Exception:  # noqa: BLE001 - a head without a random-style prior
            prior = None
        PIPE = serve.Pipeline(hf_hub_download(HEAD_REPO, "head_pre.pt"), prior, resident=True if ZERO else None,
                              head_name=HEAD_REPO.split("/")[-1])
    except Exception:  # noqa: BLE001 - shown in the ui instead of a crashed space
        WHY = traceback.format_exc(limit=3)
else:
    WHY = ("This Space is running on CPU hardware. Chameleon needs a GPU with at least 24 GB: pick an L4, L40S or "
           "A100 under Settings → Space hardware (ZeroGPU works too, with HF PRO).")

INTRO = """
# 🦎 chameleon: style cartridges for Qwen-Image

Upload a few images you like. Chameleon turns their **style** into a small *cartridge* (a ~1 KB code) and applies it to
your prompts or photos with Qwen-Image 2.1 Turbo. It's roughly a home-made `--sref`. Nothing in Qwen-Image is
fine-tuned: a style encoder → a describer head → the text Qwen-Image reads.

It carries palette, medium, lighting and mood well, but not fine brushwork or grain. **Texture boost** also shows the
model the reference image, in its middle blocks only, which adds some texture without copying the subject.
A research experiment, for non-commercial use (the describer was trained on MegaStyle-1.4M).
"""


def _cart_of(source: str, refs: list | None, carts: list | None, rseed: float, mixw: float) -> tuple:
    """(cartridge, reference image or None) from the chosen source."""
    if source == "my images":
        if not refs:
            raise gr.Error("Upload at least one style image.")
        ims = [Image.open(f if isinstance(f, str) else f.name).convert("RGB") for f in refs[:4]]
        return PIPE.make(ims, [pl.Path(f if isinstance(f, str) else f.name).name for f in refs[:4]]), ims[0]
    if source == "random style":
        return PIPE.random(int(rseed)), None
    if not carts:
        raise gr.Error("Upload one or two .cart files.")
    cs = [cg.Cartridge.load(pl.Path(f if isinstance(f, str) else f.name)) for f in carts[:2]]
    return (cs[0] if len(cs) == 1 else PIPE.mix(cs, [1 - mixw, mixw])), None


@gpu
def generate(source, refs, carts, rseed, mixw, prompts, photo, seed, render, texture, strength):
    if PIPE is None:
        raise gr.Error(WHY or "the pipeline did not load")
    cart, ref = _cart_of(source, refs, carts, rseed, mixw)
    ps = [p.strip() for p in (prompts or "").splitlines() if p.strip()][:4]
    if not ps and photo is None:
        raise gr.Error("Give at least one prompt or a photo.")
    said, out = PIPE.run(cart, ps, [photo] if photo is not None else [], ref, int(seed), render, texture, strength)
    f = pl.Path(tempfile.mkdtemp()) / "style.cart"
    cart.describe = said
    cart.save(f)
    return [(im, label) for label, im in out], said, str(f)


with gr.Blocks(title="chameleon") as demo:
    gr.Markdown(INTRO)
    if WHY:
        gr.Markdown(f"> ⚠️ {WHY}")
    with gr.Row():
        with gr.Column(scale=1):
            sources = ["my images", "cartridge file(s)"] + (["random style"] if PIPE is not None and PIPE.prior else [])
            source = gr.Radio(sources, value="my images", label="style from")
            refs = gr.File(file_count="multiple", file_types=["image"], label="style images (1–4, pooled into one style)")
            carts = gr.File(file_count="multiple", file_types=[".cart"], label=".cart file(s): two are mixed",
                            visible=False)
            mixw = gr.Slider(0, 1, 0.5, step=0.05, label="mix: weight of the second cartridge", visible=False)
            rseed = gr.Number(3, precision=0, label="random style seed", visible=False)
            prompts = gr.Textbox("a red fox in a snowy forest\na lighthouse on a rocky coast", lines=4,
                                 label="prompts (one per line, up to 4)")
            photo = gr.Image(type="pil", label="photo to restyle (optional)")
            with gr.Accordion("settings", open=False):
                texture = gr.Checkbox(True, label="texture boost (needs style images)")
                render = gr.Radio(["text", "soft"], value="text", label="render the cartridge as")
                strength = gr.Slider(0.25, 1.5, 1.0, step=0.25, label="edit strength")
                seed = gr.Number(7, precision=0, label="seed")
            go = gr.Button("generate", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="results", columns=2, height="auto")
            said = gr.Textbox(label="what the describer reads in the style", interactive=False)
            cart_out = gr.File(label="your cartridge (.cart): share it, or load it back here")

    def show(src):
        return (gr.update(visible=src == "my images"), gr.update(visible=src == "cartridge file(s)"),
                gr.update(visible=src == "cartridge file(s)"), gr.update(visible=src == "random style"))

    source.change(show, source, [refs, carts, mixw, rseed])
    go.click(generate, [source, refs, carts, rseed, mixw, prompts, photo, seed, render, texture, strength],
             [gallery, said, cart_out])

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
