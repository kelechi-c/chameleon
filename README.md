# chameleon

An open take on Midjourney's `--sref`: a reusable **style cartridge** for frozen DiT image models. It is computed once
from one or more reference images, cached as a small file, mixed, sampled at random, applied to any prompt (or to a
photo, with a strength dial), and doesn't copy the reference's content. Prototyped on FLUX.2-klein-base-4B; the target
is Qwen-Image 2.1 Turbo.

The write-up is the Sciel Lab article "Chasing `--sref` on open models"
(`sciel_lab/articles/chameleon-sref-style-cartridges.html` in the sciel repo). The style SAE side project lives in
its own repo, [style-sae](https://github.com/kelechi-c/style-sae).

## What works

**A style describer.** A small autoregressive model reads the MegaStyle encoder's embedding of a reference and emits a
distribution over the Qwen tokenizer's vocabulary, trained to predict MegaStyle's style prompts. The resulting
256-float code `z` is the cartridge. It is rendered for the image model in one of two ways:

- as **soft tokens**: `softmax(logits / τ) @ E`, fed through the frozen text encoder;
- as the describer's **greedy text**, which judges slightly better on Qwen-Image.

The describer was trained on klein, and it drives Qwen-Image zero-shot because the two share one tokenizer. The current
head, **r7**, is trained on new style labels written by a vision-language model for 41k MegaStyle styles and 1,054
Midjourney sref codes (`scripts/label_megastyle.py`, about $4 through OpenRouter); it scores +.10 CSD over the
previous head (r6c) with no added copying.

**A latent window.** At generation time the reference's VAE latents can be shown to DiT blocks 12-19 only. That adds
surface texture (stitches, strokes) without the copying that full reference latents cause (`texture=True`).

Everything trained through the frozen DiT's denoising loss learned no reference-specific information (see
`docs/notes.md`). Qwen-Image's own reference input has the strongest style, but it copies subjects, through the
reference's VAE latents.

## Try it

The easiest way in is the Colab notebook, `notebooks/chameleon.ipynb`: upload a few style images, then restyle prompts
or photos, mix styles, interpolate, and sample random styles (needs a 24 GB+ GPU; an A100 or L4 works). `space/` is the
same UI as a Hugging Face Space. From Python:

```python
from chameleon import serve
from huggingface_hub import hf_hub_download
pipe = serve.Pipeline(hf_hub_download("tensorkelechi/chameleon-r7", "head_pre.pt"),
                      hf_hub_download("tensorkelechi/chameleon-r6c", "code_prior.pt"), head_name="r7")
cart = pipe.make([style_image], ["my-style"])          # one cartridge from one or more images
text, out = pipe.run(cart, ["a heron in a misty marsh"], [photo], ref=style_image, texture=True)
# text: the cartridge's style description; out: (label, image) pairs, plain and latent-window versions
mixed = pipe.mix([cart_a, cart_b], [0.5, 0.5])         # mixing and interpolation
rand = pipe.random(4417)                               # a random style from a seed
```

## Command line

```sh
python scripts/cartridge.py make   --head runs/r6c/head_pre.pt --refs a.jpg b.jpg --out style.cart
python scripts/cartridge.py mix    --carts a.cart b.cart --weights .7 .3 --out ab.cart
python scripts/cartridge.py random --head runs/r6c/head_pre.pt --prior runs/r6c/eval/code_prior.pt --seed 3 --out r3.cart
python scripts/cartridge.py apply  --head runs/r6c/head_pre.pt --cart style.cart --prompt "a red fox in a snowy forest" --render text --out fox.png
python scripts/cartridge.py apply  --head runs/r6c/head_pre.pt --cart style.cart --photo photo.jpg --strength 1.5 --out styled.png
```

`apply` runs in two phases (encoder, then DiT) and fits a 24 GB L4. `bench` reproduces the evaluation: describer
heads compared, a Midjourney sref test, interpolation, strength and random codes.

## Layout

| path | what |
|---|---|
| `src/chameleon/klein.py`, `style.py` | FLUX.2-klein wrapper; MegaStyle backbone; style heads and the describer |
| `src/chameleon/qport/` | a lean port of Qwen-Image 2.1 Turbo (DiT, encoding, 8-step sampler, edit / `edit_mix`) |
| `src/chameleon/qimg.py` | describer → Qwen-Image soft tokens |
| `src/chameleon/cartridge.py` | cartridges: make, mix, random prior, render |
| `src/chameleon/serve.py` | the end-to-end pipeline: make / mix / random / run (t2i, edit, strength, latent window) |
| `src/chameleon/lora.py` | the reference LoRA experiment (a trained cartridge) |
| `src/chameleon/qtok.py` | option B: perceiver adapter over raw patch tokens (did not learn) |
| `scripts/` | training (`train.py`), zero-shot runs (`qimg_zero.py`, `qimg_edit.py`), native-reference baselines (`qimg_ref*.py`), copying-channel ablation and SAE filter (`qimg_chan.py`, needs style-sae), cartridges (`cartridge.py`), style labelling (`label_styles.py`, `label_megastyle.py`, `relabel_cache.py`, `prep_mj.py`), head comparison (`compare_heads.py`), showcases and their grids (`showcase_*.py`), figures |
| `notebooks/` | the Colab notebook |
| `space/` | the Gradio app (Hugging Face Space) |
| `configs/` | evaluation references (art, Midjourney sref codes, held-out MegaStyle) and content photos |
| `docs/` | `plan.md`, `project.md`, `lit.md`, and `notes.md`, the full experiment log |

Checkpoints and outputs (`runs/`) are not in the repo. The r7 head is on Hugging Face (`tensorkelechi/chameleon-r7`), and
the style labels are a dataset (`tensorkelechi/chameleon-style-labels`).

## Data and licences

- MegaStyle-1.4M is released for research use.
- The Midjourney images under `configs/refs_ext` come from a public Hugging Face dataset and are used for evaluation
  only.
- The art references under `configs/refs_art` are public-domain paintings plus two held-out style samples.
