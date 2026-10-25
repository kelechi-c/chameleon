"""Style describer -> Qwen-Image 2.1 (Turbo, via the vendored qport). the describer's vocabulary is identical to
qwen3-vl's (all 151,643 ids), so its per-step distributions become qwen-image soft tokens as p @ E_qwen3vl.

the soft tokens replace PLACE placeholders in the user turn and the encoder runs on input embeddings (t2i, or edit
with condition images); states are qport's (last layer before the final norm, system tokens dropped).
"""

import pathlib as pl
import typing as tp

import torch
from torch import Tensor

from chameleon.qport import model as qm

QWEN3: str = "Qwen/Qwen3-4B"
PLACE: str = "<|vision_pad|>"  # placeholder token whose input embedding is replaced by a soft style token


def qwen3_embeddings(dev: str = "cuda") -> Tensor:
    """Qwen3-4B's input-embedding matrix (151936, 2560) alone (the describer reads and writes text through it); only
    the safetensors file holding it is downloaded."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    key = "model.embed_tokens.weight"
    index = json.loads(pl.Path(hf_hub_download(QWEN3, "model.safetensors.index.json")).read_text())
    with safe_open(hf_hub_download(QWEN3, index["weight_map"][key]), framework="pt") as h:
        return h.get_tensor(key).to(dev, torch.bfloat16)


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
        pos = ids[0] == processor.tokenizer.convert_tokens_to_ids(PLACE)
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
