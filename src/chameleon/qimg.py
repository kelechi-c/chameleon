"""Style describer -> Qwen-Image 2.1 (Turbo, via the vendored qport). the describer's vocabulary is identical to
qwen3-vl's (all 151,643 ids), so its per-step distributions become qwen-image soft tokens as p @ E_qwen3vl.

the soft tokens replace PLACE placeholders in the user turn and the encoder runs on input embeddings (t2i, or edit
with condition images); states are qport's (last layer before the final norm, system tokens dropped).
"""

import pathlib as pl
import typing as tp

import torch
from torch import Tensor

from chameleon import klein
from chameleon.qport import model as qm

KLEIN_TE: str = "black-forest-labs/FLUX.2-klein-base-4B"


def klein_embeddings(dev: str = "cuda") -> Tensor:
    """Klein's qwen3-4b input-embedding matrix (151936, 2560) alone, read from its safetensors (the describer decodes
    through it); no transformer weights are loaded."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = pl.Path(snapshot_download(KLEIN_TE, allow_patterns=["text_encoder/*"]))
    for f in sorted((snap / "text_encoder").glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for key in h.keys():
                if key.endswith("embed_tokens.weight"):
                    return h.get_tensor(key).to(dev, torch.bfloat16)
    raise KeyError("embed_tokens.weight not found in klein's text encoder")


@torch.no_grad()
def encode_soft(encoder: tp.Any, processor: tp.Any, prompt: str, soft: Tensor | None,
                images: list[tp.Any] | None = None) -> qm.Text:
    """qm.encode with soft tokens: `prompt` holds len(soft) PLACE placeholders whose input embeddings are replaced by
    soft (n, 4096); optional condition images (edit mode). the 3-d rope positions are computed from the real token ids
    (qwen3-vl needs them for the image slots), then the encoder runs on the edited embeddings; image features are
    scattered in by embedding match. returns (states (1, T, 4096), image-slot mask (T,))."""
    rgb = [im.convert("RGB") for im in images or []]
    system = [{"role": "system", "content": [{"type": "text", "text": qm.SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    kw = {"images": rgb} if rgb else {}
    inputs = processor(text=[qm.template(prompt, len(rgb))], return_tensors="pt", **kw).to(qm.DEV)
    ids = inputs.input_ids
    pad = processor.tokenizer.encode("<|image_pad|>")[0]
    lm = getattr(encoder.model, "language_model", encoder.model)
    emb = lm.embed_tokens(ids)
    if soft is not None:
        pos = ids[0] == processor.tokenizer.convert_tokens_to_ids(klein.PLACE)
        assert int(pos.sum()) == soft.shape[0], "placeholder count != soft tokens"
        emb = emb.clone()
        emb[0, pos] = soft.to(emb.dtype)
    hook = lm.norm.register_forward_hook(lambda m, a, o: a[0])
    try:
        if rgb:
            rest = {k: v for k, v in inputs.items() if k not in ("input_ids",)}
            pos3 = encoder.model.compute_3d_position_ids(
                input_ids=ids, image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=None, inputs_embeds=emb,
                attention_mask=inputs.attention_mask, past_key_values=None,
                mm_token_type_ids=inputs.get("mm_token_type_ids"))
            hidden = encoder.model(inputs_embeds=emb, position_ids=pos3, use_cache=False, **rest).last_hidden_state
        else:
            hidden = lm(inputs_embeds=emb, attention_mask=inputs.attention_mask, use_cache=False).last_hidden_state
    finally:
        hook.remove()
    return hidden[:, drop:].to(torch.bfloat16), ids[0, drop:] == pad


@torch.no_grad()
def t2i(core: qm.Core, text: qm.Text, seed: int, shape: tuple[int, int] = (64, 64)) -> tp.Any:
    """Turbo text-to-image (no condition image) at `shape` latent tokens (64 x 64 = 1024 px)."""
    empty = torch.zeros(1, 0, 64, device=qm.DEV, dtype=torch.bfloat16)
    return qm.edit(core, text, empty, [], seed, out_shape=shape)
