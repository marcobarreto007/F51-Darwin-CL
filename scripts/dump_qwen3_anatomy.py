"""Read-only introspection of the pinned Qwen3 donor. Does not save weights."""

from __future__ import annotations

import inspect

import torch
from transformers import AutoModelForCausalLM

DONOR_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        DONOR_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
    )
    layer = model.model.layers[0]
    print("MODEL", f"{type(model).__module__}.{type(model).__name__}")
    print("INNER", f"{type(model.model).__module__}.{type(model.model).__name__}")
    print("LAYER", f"{type(layer).__module__}.{type(layer).__name__}")
    print("N_LAYERS", len(model.model.layers))
    print("HIDDEN", model.config.hidden_size)
    print("INTERMEDIATE", model.config.intermediate_size)
    print("RMS_EPS", model.config.rms_norm_eps)
    print("CHILDREN")
    for name, module in layer.named_children():
        print(f"  {name} {type(module).__module__}.{type(module).__name__}")
    print("ATTN")
    for name, module in layer.self_attn.named_children():
        print(f"  {name} {type(module).__name__}")
    print("MLP")
    for name, module in layer.mlp.named_children():
        print(f"  {name} {type(module).__name__}")
    print("FORWARD", inspect.signature(layer.forward))
    print("MLP_FORWARD", inspect.signature(layer.mlp.forward))
    print("KEYS")
    for key, value in model.state_dict().items():
        if (
            key.startswith("model.layers.0.")
            or key.startswith("model.norm")
            or key.startswith("model.embed")
            or key.startswith("lm_head")
        ):
            print(f"  {key} {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()
