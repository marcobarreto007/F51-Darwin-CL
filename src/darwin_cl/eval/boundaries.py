"""EXACT_SCORED_UTF8 — the only cross-model byte policy.

A causal model scores tokens t1..tn given t0..t(n-1). The denominator may
contain only UTF-8 bytes of tokens whose NLL is inside the sum.

    input tokens:    t0 t1 t2 t3
    scored targets:     t1 t2 t3
    denominator:        bytes(t1)+bytes(t2)+bytes(t3)

t0 is context. Its bytes stay out. Padding and special tokens stay out,
including a BOS or EOS that is not itself a scored target.

This donor has no separate BOS (add_bos_token is false, bos string is
absent). config bos_token_id equals eos id 151643, the special token
<|endoftext|>. Prepending that token would score t0 of the raw text, but
it would also change the measurement. That alternative is not this policy.
v1 already implements EXACT_SCORED_UTF8, so v1 stays.

Documents are scored alone. EOS is not appended.
"""

from __future__ import annotations


SCORED = "scored"
CONTEXT = "context"
PAD = "pad"
SPECIAL = "special"


def evaluated_utf8_bytes(
    text: str,
    offsets: list[tuple[int, int]],
    roles: list[str],
) -> int:
    if len(offsets) != len(roles):
        raise ValueError("offsets and roles must have the same length")
    flags = [False] * len(text)
    for (start, end), role in zip(offsets, roles):
        if role != SCORED:
            continue
        start_i = max(0, int(start))
        end_i = min(len(text), int(end))
        for index in range(start_i, end_i):
            flags[index] = True
    covered = "".join(char for char, flag in zip(text, flags) if flag)
    return len(covered.encode("utf-8"))


def total_scored_nll(token_nlls: list[float], roles_after_context: list[str]) -> float:
    """token_nlls aligns with tokens after the first (the causal shift)."""
    if len(token_nlls) != len(roles_after_context):
        raise ValueError("NLL list and post-context roles differ in length")
    total = 0.0
    for nll, role in zip(token_nlls, roles_after_context):
        if role != SCORED:
            continue
        if nll < 0:
            raise ValueError("token NLL must be non-negative")
        total += float(nll)
    return total
