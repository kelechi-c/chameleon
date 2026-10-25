# chameleon

Style cartridges for Qwen-Image 2.1 Turbo, roughly a home-made Midjourney `--sref`. Give it one or more images whose
style you like. A small describer head writes that style out as a short description (the cartridge), which Qwen-Image
then follows for new prompts or for restyling your photos. Qwen-Image itself is not fine-tuned.

```
style reference ─► MegaStyle encoder ─► describer head (trained) ─► style text ─► Qwen-Image 2.1 Turbo
```

Cartridges can be saved (~1 KB), mixed and interpolated. An optional *texture boost* also shows Qwen-Image the style
image itself, in DiT blocks 12-19 only, which adds surface texture without copying its subject.

## Try it

- Colab: `notebooks/chameleon.ipynb` (needs a GPU with 22 GB or more, such as an L4 or A100).
- Python:

```python
from huggingface_hub import hf_hub_download
from chameleon import serve

pipe = serve.Pipeline(hf_hub_download("tensorkelechi/chameleon-ar-head", "head_pre.pt"), None)
cart = pipe.make([style_image], ["my-style"])                     # one cartridge from one or more images
text, out = pipe.run(cart, ["a lighthouse on a rocky coast"], [photo], ref=style_image, texture=True)
mixed = pipe.mix([cart_a, cart_b], [0.5, 0.5])                    # mixing and interpolation
```

`text` is the style description; `out` holds (label, image) pairs, with and without texture boost. `space/` is the
same thing as a Gradio app.

## Train the head

`scripts/train_head.py` extracts style features from MegaStyle-1.4M (and the Midjourney sref set), attaches the
labels from `tensorkelechi/chameleon-style-labels`, and trains the describer:

```sh
python scripts/train_head.py features --shard train-00001.parquet --out data/s00001   # repeat for 00002-00004
python scripts/train_head.py features --shard train-00099.parquet --out data/ev99
python scripts/train_head.py features --mj --out data/mj
python scripts/train_head.py train --data data/s0000{1,2,3,4} data/mj --mult 1 1 1 1 3 --eval data/ev99 --out runs/head
```

Training the released head took about 28 minutes on one A100, after feature extraction.

## Links

- Weights: [tensorkelechi/chameleon-ar-head](https://huggingface.co/tensorkelechi/chameleon-ar-head)
- Style labels: [tensorkelechi/chameleon-style-labels](https://huggingface.co/datasets/tensorkelechi/chameleon-style-labels)
- Write-up: the Sciel Lab article "Chasing `--sref` on open models"

## Licences

The head is trained on MegaStyle-1.4M, which is released for research use, so chameleon is for research and
non-commercial use. Qwen-Image 2.1 Turbo, the MegaStyle encoder and Qwen3-4B (only its token-embedding matrix is
used) are under their own licences.
