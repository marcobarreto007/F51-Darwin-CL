"""Official cross-model metrics.

NATS_PER_BYTE = total_negative_log_likelihood / total_UTF8_bytes_evaluated
BITS_PER_BYTE = NATS_PER_BYTE / ln(2)

Token loss is internal to one model. It is not the comparison metric.
"""

from __future__ import annotations

import math


LN2 = math.log(2)


def nats_per_byte(total_nll: float, evaluated_utf8_bytes: int) -> float:
    if evaluated_utf8_bytes <= 0:
        raise ValueError("evaluated_utf8_bytes must be positive")
    if total_nll < 0:
        raise ValueError("total_nll must be non-negative")
    return float(total_nll) / int(evaluated_utf8_bytes)


def bits_per_byte(total_nll: float, evaluated_utf8_bytes: int) -> float:
    return nats_per_byte(total_nll, evaluated_utf8_bytes) / LN2


def nats_per_token(total_nll: float, scored_tokens: int) -> float:
    """Internal metric. Not used to compare models with different tokenizers."""
    if scored_tokens <= 0:
        raise ValueError("scored_tokens must be positive")
    if total_nll < 0:
        raise ValueError("total_nll must be non-negative")
    return float(total_nll) / int(scored_tokens)
