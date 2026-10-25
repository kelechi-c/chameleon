---
title: chameleon
emoji: 🦎
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
license_name: research-only
short_description: Style cartridges for Qwen-Image (a home-made --sref)
models:
  - Qwen/Qwen-Image-2.1-Turbo
  - black-forest-labs/FLUX.2-klein-base-4B
  - Gaojunyao/MegaStyle
---

# chameleon

Chameleon turns the style of a few images into a ~1 KB cartridge and applies it with Qwen-Image 2.1 Turbo, without
fine-tuning Qwen-Image. Reference images go through the MegaStyle encoder to a style code, which a describer head
writes out as text or soft tokens for Qwen-Image's text encoder. Texture boost adds the reference's VAE latents, visible
only in DiT blocks 12–19.

- Hardware: a GPU with at least 24 GB (L4, L40S or A100), or ZeroGPU. On a GPU under 40 GB, the VL encoder and
  the DiT take turns on the GPU.
- Weights: `weights/head_pre.pt` (the r6c describer head) and `weights/code_prior.pt` (the random-style prior). The
  head was trained on MegaStyle-1.4M, so use it for research and non-commercial work only.
