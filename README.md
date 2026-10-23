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

The describer was trained on klein, and it drives Qwen-Image zero-shot because the two share one tokenizer.

Everything trained through the frozen DiT's denoising loss learned no reference-specific information (see
`docs/notes.md`). Qwen-Image's own reference input has the strongest style, but it copies subjects, through the
reference's VAE latents.

## Use a cartridge

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
| `src/chameleon/qtok.py` | option B: perceiver adapter over raw patch tokens (did not learn) |
| `scripts/` | training (`train.py`), zero-shot runs (`qimg_zero.py`, `qimg_edit.py`), native-reference baselines (`qimg_ref*.py`), copying-channel ablation and SAE filter (`qimg_chan.py`, needs style-sae), cartridges (`cartridge.py`), figures |
| `tools/` | Colab runner (`colab.sh`), run queues, the poller with an idle-GPU guard |
| `configs/` | evaluation references (art, Midjourney sref codes, held-out MegaStyle) and content photos |
| `docs/` | `plan.md`, `project.md`, `lit.md`, and `notes.md`, the full experiment log |

Checkpoints and outputs (`runs/`) are not in the repo. They live on Google Drive (`tensordrive2:chameleon/runs`).

## Data and licences

- MegaStyle-1.4M is released for research use.
- The Midjourney images under `configs/refs_ext` come from a public Hugging Face dataset and are used for evaluation
  only.
- The art references under `configs/refs_art` are public-domain paintings plus two held-out style samples.
