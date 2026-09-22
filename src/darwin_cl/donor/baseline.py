"""Frozen donor baseline. No training, no adapters, no weight updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from darwin_cl.eval.boundaries import (
    CONTEXT,
    SCORED,
    evaluated_utf8_bytes,
    total_scored_nll,
)
from darwin_cl.eval.metrics import bits_per_byte, nats_per_byte, nats_per_token

DONOR_ID = "Qwen/Qwen3-0.6B-Base"
DONOR_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def load_donor(
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(
        DONOR_ID,
        revision=DONOR_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        DONOR_ID,
        revision=DONOR_REVISION,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def weight_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def explain_score(model, tokenizer, text: str) -> dict[str, Any]:
    """One document under EXACT_SCORED_UTF8.

    Hugging Face causal shift: logits[:, :-1] predict input_ids[:, 1:].
    score_nats_per_byte() is explain_score()["nats_per_byte"].
    """
    if not text:
        raise ValueError("text must be non-empty")
    donor_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".branch." not in name
    ]
    if donor_trainable:
        raise RuntimeError("donor parameters must be frozen")
    device = next(model.parameters()).device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    offsets = [(int(pair[0]), int(pair[1])) for pair in encoded["offset_mapping"][0].tolist()]
    token_ids = [int(token) for token in input_ids[0].tolist()]
    if len(token_ids) < 2 or len(offsets) != len(token_ids):
        raise ValueError("document needs at least two tokens so one can be scored")
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_nlls = [float(value) for value in token_nll[0].tolist()]
    roles = [CONTEXT] + [SCORED] * (len(token_ids) - 1)
    nll = total_scored_nll(token_nlls, roles[1:])
    if abs(nll - sum(token_nlls)) > 1e-6:
        raise RuntimeError("reported NLL is not the sum of scored token NLLs")
    nbytes = evaluated_utf8_bytes(text, offsets, roles)
    raw_bytes = len(text.encode("utf-8"))
    context_start, context_end = offsets[0]
    context_bytes = len(text[context_start:context_end].encode("utf-8"))
    if raw_bytes - nbytes != context_bytes:
        raise RuntimeError("denominator dropped bytes that are not the unscored first token")
    return {
        "raw_characters": len(text),
        "raw_utf8_bytes": raw_bytes,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "scored_token_ids": token_ids[1:],
        "scored_token_count": len(token_ids) - 1,
        "scored_token_nlls": token_nlls,
        "total_nll": nll,
        "scored_tokens": len(token_nlls),
        "scored_utf8_bytes": nbytes,
        "evaluated_utf8_bytes": nbytes,
        "unscored_boundary_bytes": context_bytes,
        "context_token_id": token_ids[0],
        "context_text": text[context_start:context_end],
        "nats_per_token": nats_per_token(nll, len(token_nlls)),
        "nats_per_byte": nats_per_byte(nll, nbytes),
        "bits_per_byte": bits_per_byte(nll, nbytes),
        "policy": "EXACT_SCORED_UTF8",
    }


def score_loss(model, tokenizer, text: str) -> dict[str, Any]:
    """Sum of token NLL and the UTF-8 bytes those scored tokens cover."""
    full = explain_score(model, tokenizer, text)
    return {
        "total_nll": full["total_nll"],
        "scored_tokens": full["scored_tokens"],
        "token_count": full["token_count"],
        "evaluated_utf8_bytes": full["evaluated_utf8_bytes"],
        "raw_utf8_bytes": full["raw_utf8_bytes"],
        "unscored_boundary_bytes": full["unscored_boundary_bytes"],
        "nats_per_token": full["nats_per_token"],
        "nats_per_byte": full["nats_per_byte"],
        "bits_per_byte": full["bits_per_byte"],
    }


def score_nats_per_byte(model, tokenizer, text: str) -> float:
    return float(explain_score(model, tokenizer, text)["nats_per_byte"])


def save_reference_outputs(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
