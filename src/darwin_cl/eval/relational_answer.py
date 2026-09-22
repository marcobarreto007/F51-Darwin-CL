"""Conservative scoring for the synthetic relational development suite.

Only answer text is accepted. Unrecognized prose is queued for review instead
of being counted as either a success or a failure of relational reasoning.
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


_VALUE = r"[A-Za-z0-9-]+"
_RELATION = r"(?:maker|city|color|tool)"
_SEPARATOR = r"(?:\s+and\s+|\s*,\s*(?:and\s+)?)"
_BARE_PAIR = re.compile(rf"\s*({_VALUE}){_SEPARATOR}({_VALUE})\s*[.]?\s*\Z", re.I)
_FACT_PAIR = re.compile(
    rf"\s*({_VALUE})\s+({_RELATION})\s+is\s+({_VALUE})"
    rf"{_SEPARATOR}"
    rf"({_VALUE})\s+({_RELATION})\s+is\s+({_VALUE})\s*[.]?\s*\Z",
    re.I,
)
_BARE_WORD = re.compile(rf"\s*{_VALUE}\s*[.]?\s*\Z")


def verdict(row: dict, span: str) -> str:
    """Return ``correct``, ``incorrect`` or ``review`` for one dev prediction.

    An explicit answer or a complete pair of labelled facts can be checked
    mechanically. Other sentences require human review. Neither the prompt
    nor the gold answer may be searched as a substring of the prediction.
    """

    got = normalize(span)
    gold = normalize(row["answer"])
    if got == gold:
        return "correct"

    kind = row["kind"]
    if kind == "composition":
        query = row["query"]
        expected = row["answer"].split(" and ")
        if len(expected) != 2:
            raise ValueError("composition label must contain two values")
        bare = _BARE_PAIR.fullmatch(span)
        if bare:
            return "correct" if [normalize(part) for part in bare.groups()] == [normalize(part) for part in expected] else "incorrect"
        facts = _FACT_PAIR.fullmatch(span)
        if facts:
            parts = facts.groups()
            found = [(parts[0], parts[1], parts[2]), (parts[3], parts[4], parts[5])]
            wanted = [
                (query["subject_a"], query["relation_a"], expected[0]),
                (query["subject_b"], query["relation_b"], expected[1]),
            ]
            return "correct" if sorted(tuple(map(normalize, fact)) for fact in found) == sorted(tuple(map(normalize, fact)) for fact in wanted) else "incorrect"
        return "review"

    if kind == "false_claim":
        if got in {"false", "no"}:
            return "correct"
        if got in {"true", "yes"}:
            return "incorrect"
        return "review"

    if kind == "missing":
        return "incorrect" if _BARE_WORD.fullmatch(span) else "review"

    if kind in {"atomic", "inverse"}:
        return "incorrect" if _BARE_WORD.fullmatch(span) else "review"

    raise ValueError(f"unknown query kind: {kind}")
