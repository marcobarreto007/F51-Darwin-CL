"""Deterministic relational scorer. Not a substring search and not an LLM judge."""

from __future__ import annotations

import re

SUBJECTS = ("nex-41", "bolt-7", "kite-3", "orb-12")
RELATIONS = ("maker", "city", "color", "tool")
RELATION_WORDS = {
    "maker": ("maker", "made", "person", "human"),
    "city": ("city",),
    "color": ("color", "hue"),
    "tool": ("tool", "implement"),
}


def normalize(text: str) -> str:
    chars = []
    for char in text.lower():
        if char.isalnum() or char == "-":
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def values_of(edges: dict) -> dict[str, set[str]]:
    grouped = {relation: set() for relation in RELATIONS}
    for (_subject, relation), value in edges.items():
        grouped[relation].add(normalize(value))
    return grouped


def all_values(edges: dict) -> set[str]:
    return {normalize(value) for value in edges.values()}


def triples(span: str, edges: dict) -> list[tuple[str, str, str]]:
    text = normalize(span)
    found = []
    subjects = "|".join(SUBJECTS)
    values = "|".join(sorted(all_values(edges), key=len, reverse=True))
    if not values:
        return found
    for relation, words in RELATION_WORDS.items():
        rel = "|".join(words)
        patterns = (
            rf"({subjects})\s+(?:{rel})\s+is\s+({values})",
            rf"(?:{rel})\s+of\s+({subjects})\s+is\s+({values})",
            rf"(?:the\s+)?(?:{rel})\s+(?:listed|recorded|assigned)\s+for\s+({subjects})\s+is\s+({values})",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                found.append((match.group(1), relation, match.group(2)))
        if relation == "city":
            for match in re.finditer(rf"({subjects})\s+is\s+(?:located|filed)\s+in\s+({values})", text):
                found.append((match.group(1), "city", match.group(2)))
    return found


def direct_slots(span: str, edges: dict) -> list[str] | None:
    raw_parts = re.split(r"\s+and\s+|,", span, flags=re.IGNORECASE)
    if len(raw_parts) == 1:
        return None
    slots = []
    known = all_values(edges) | set(SUBJECTS)
    for part in raw_parts:
        token = normalize(part)
        words = token.split()
        if len(words) == 1 and words[0] in known:
            slots.append(words[0])
            continue
        if words and all(word == words[0] for word in words) and words[0] in known:
            slots.append(words[0])
            continue
        return None
    return slots


def sole_value(span: str, edges: dict) -> str | None:
    words = normalize(span).split()
    if not words:
        return None
    known = all_values(edges) | set(SUBJECTS) | {"false", "true", "indeterminado", "no"}
    if all(word == words[0] for word in words) and words[0] in known:
        return words[0]
    return None


def negated(span: str) -> bool:
    text = normalize(span)
    return bool(re.match(r"^(no|false|not)\b", text) or re.search(r"\b(is false|claim is false|not recorded|no motto|unknown|indeterminado)\b", text))


def affirms(found: list[tuple[str, str, str]], subject: str, relation: str, value: str) -> bool:
    return any(item == (normalize(subject), relation, normalize(value)) for item in found)


def judge(span: str, row: dict, edges: dict, hit_limit: bool = False) -> dict:
    query = row["query"]
    kind = query["kind"]
    label = row["answer"]
    found = triples(span, edges)
    slots = direct_slots(span, edges)
    single = sole_value(span, edges)
    if solve_label(edges, query) != label:
        return {"verdict": "ABSTAIN", "reason": "solver_label_mismatch", "found": found}

    if kind == "atomic":
        return judge_atomic(span, query, found, slots, single, hit_limit)
    if kind == "inverse":
        return judge_inverse(query, found, single, hit_limit, edges)
    if kind == "composition":
        return judge_composition(query, edges, found, slots, hit_limit)
    if kind == "false_claim":
        return judge_false(span, query, found)
    if kind == "missing":
        return judge_missing(span, found, hit_limit)
    return {"verdict": "ABSTAIN", "reason": "unknown_kind", "found": found}


def solve_label(edges: dict, query: dict) -> str:
    kind = query["kind"]
    if kind == "atomic":
        return edges.get((query["subject"], query["relation"]), "INDETERMINADO")
    if kind == "inverse":
        found = [subject for (subject, relation), obj in edges.items() if relation == query["relation"] and normalize(obj) == normalize(query["object"])]
        return found[0] if len(found) == 1 else "INDETERMINADO"
    if kind == "composition":
        left = edges.get((query["subject_a"], query["relation_a"]))
        right = edges.get((query["subject_b"], query["relation_b"]))
        if left is None or right is None:
            return "INDETERMINADO"
        return f"{left} and {right}"
    if kind == "false_claim":
        actual = edges.get((query["subject"], query["relation"]))
        if actual is None:
            return "INDETERMINADO"
        return "false" if normalize(actual) != normalize(query["claimed"]) else "true"
    if kind == "missing":
        return "INDETERMINADO"
    return ""


def judge_atomic(span, query, found, slots, single, hit_limit) -> dict:
    subject = normalize(query["subject"])
    relation = query["relation"]
    gold = normalize(solve_label_from_query_only(query))
    gold = normalize(query.get("_gold", gold))
    # gold is passed via the row answer; caller checks solver. Use edges through found.
    contradictions = [item for item in found if item[0] == subject and item[1] == relation and item[2] != gold_from_found_context(query, found, gold)]
    target = gold_value(query, found)
    if target is None:
        return {"verdict": "ABSTAIN", "reason": "missing_gold", "found": found}
    good = [item for item in found if item[0] == subject and item[1] == relation and item[2] == target]
    bad = [item for item in found if item[0] == subject and item[1] == relation and item[2] != target]
    if good and not bad:
        return {"verdict": "PASS", "reason": "triple", "found": found}
    if bad and not good:
        return {"verdict": "FAIL", "reason": "wrong_value", "found": found}
    if good and bad:
        return {"verdict": "ABSTAIN", "reason": "contradictory_values", "found": found}
    if single == target and slots is None:
        return {"verdict": "PASS", "reason": "direct_value", "found": found}
    if single and single != target:
        return {"verdict": "FAIL", "reason": "direct_wrong_value", "found": found}
    if hit_limit:
        return {"verdict": "ABSTAIN", "reason": "truncated", "found": found}
    if not normalize(span):
        return {"verdict": "ABSTAIN", "reason": "empty", "found": found}
    return {"verdict": "ABSTAIN", "reason": "no_association", "found": found}


def gold_value(query, found) -> str | None:
    return None


def gold_from_found_context(query, found, gold):
    return gold


def judge_atomic_with_gold(span, query, edges, found, hit_limit) -> dict:
    subject = normalize(query["subject"])
    relation = query["relation"]
    target = normalize(edges[(query["subject"], query["relation"])])
    slots = direct_slots(span, edges)
    single = sole_value(span, edges)
    good = [item for item in found if item[0] == subject and item[1] == relation and item[2] == target]
    bad = [item for item in found if item[0] == subject and item[1] == relation and item[2] != target]
    if good and not bad:
        return {"verdict": "PASS", "reason": "triple", "found": found}
    if bad:
        return {"verdict": "FAIL", "reason": "wrong_value", "found": found}
    if single == target:
        return {"verdict": "PASS", "reason": "direct_value", "found": found}
    if single and single != target:
        return {"verdict": "FAIL", "reason": "direct_wrong_value", "found": found}
    wrong_subject = [item for item in found if item[1] == relation and item[2] == target and item[0] != subject]
    if wrong_subject:
        return {"verdict": "FAIL", "reason": "wrong_subject", "found": found}
    if hit_limit:
        return {"verdict": "ABSTAIN", "reason": "truncated", "found": found}
    if not normalize(span):
        return {"verdict": "ABSTAIN", "reason": "empty", "found": found}
    return {"verdict": "ABSTAIN", "reason": "no_association", "found": found}


def judge(span: str, row: dict, edges: dict, hit_limit: bool = False) -> dict:
    query = row["query"]
    if solve_label(edges, query) != row["answer"]:
        return {"verdict": "ABSTAIN", "reason": "solver_label_mismatch", "found": []}
    found = triples(span, edges)
    kind = query["kind"]
    if kind == "atomic":
        return judge_atomic_with_gold(span, query, edges, found, hit_limit)
    if kind == "inverse":
        return judge_inverse(query, edges, found, sole_value(span, edges), hit_limit)
    if kind == "composition":
        return judge_composition(span, query, edges, found, direct_slots(span, edges), hit_limit)
    if kind == "false_claim":
        return judge_false(span, query, edges, found)
    if kind == "missing":
        return judge_missing(span, found, hit_limit)
    return {"verdict": "ABSTAIN", "reason": "unknown_kind", "found": found}


def judge_inverse(query, edges, found, single, hit_limit) -> dict:
    relation = query["relation"]
    obj = normalize(query["object"])
    owners = [subject for (subject, rel), value in edges.items() if rel == relation and normalize(value) == obj]
    target = normalize(owners[0]) if len(owners) == 1 else None
    matched = [item for item in found if item[1] == relation and item[2] == obj]
    if single == target:
        return {"verdict": "PASS", "reason": "direct_subject", "found": found}
    if single == obj:
        return {"verdict": "FAIL", "reason": "answered_with_object", "found": found}
    if len(matched) == 1 and matched[0][0] == target:
        return {"verdict": "PASS", "reason": "triple", "found": found}
    if matched and any(item[0] != target for item in matched):
        return {"verdict": "FAIL", "reason": "wrong_subject", "found": found}
    if hit_limit:
        return {"verdict": "ABSTAIN", "reason": "truncated", "found": found}
    return {"verdict": "ABSTAIN", "reason": "no_association", "found": found}


def judge_composition(span, query, edges, found, slots, hit_limit) -> dict:
    left = (normalize(query["subject_a"]), query["relation_a"], normalize(edges[(query["subject_a"], query["relation_a"])]))
    right = (normalize(query["subject_b"]), query["relation_b"], normalize(edges[(query["subject_b"], query["relation_b"])]))
    left_bad = [item for item in found if item[0] == left[0] and item[1] == left[1] and item[2] != left[2]]
    right_bad = [item for item in found if item[0] == right[0] and item[1] == right[1] and item[2] != right[2]]
    left_ok = left in found and not left_bad
    right_ok = right in found and not right_bad
    if left_ok and right_ok:
        return {"verdict": "PASS", "reason": "both_triples", "found": found}
    if left_bad or right_bad:
        return {"verdict": "FAIL", "reason": "wrong_or_contradictory_value", "found": found}
    if slots == [left[2], right[2]]:
        return {"verdict": "PASS", "reason": "ordered_values", "found": found}
    if slots == [right[2], left[2]]:
        return {"verdict": "FAIL", "reason": "swapped_values", "found": found}
    if slots and (slots[0] != left[2] or (len(slots) > 1 and slots[1] != right[2])):
        return {"verdict": "FAIL", "reason": "wrong_slot", "found": found}
    if hit_limit or (left_ok ^ right_ok):
        return {"verdict": "ABSTAIN", "reason": "truncated", "found": found}
    if sole_value(span, edges):
        return {"verdict": "FAIL", "reason": "single_value_for_two_slots", "found": found}
    return {"verdict": "ABSTAIN", "reason": "no_association", "found": found}


def judge_false(span, query, edges, found) -> dict:
    subject = normalize(query["subject"])
    relation = query["relation"]
    claimed = normalize(query["claimed"])
    actual = normalize(edges[(query["subject"], query["relation"])])
    affirms_claim = affirms(found, subject, relation, claimed)
    affirms_actual = affirms(found, subject, relation, actual)
    text = normalize(span)
    bare = sole_value(span, edges) in {"false", "no"}
    junk = bool(re.match(r"^(no|false|not)(-\d|\s+\d)", text))
    denial = (bare or negated(span)) and not junk
    if affirms_claim and denial:
        return {"verdict": "FAIL", "reason": "negation_then_contradiction", "found": found}
    if affirms_claim and not denial:
        return {"verdict": "FAIL", "reason": "affirms_false_claim", "found": found}
    if denial and (affirms_actual or not affirms_claim):
        return {"verdict": "PASS", "reason": "negation", "found": found}
    if sole_value(span, edges) == "true" or sole_value(span, edges) == claimed:
        return {"verdict": "FAIL", "reason": "affirms_false_claim", "found": found}
    return {"verdict": "ABSTAIN", "reason": "ambiguous_negation", "found": found}


def judge_missing(span, found, hit_limit) -> dict:
    if negated(span) or sole_value(span, {"x": "INDETERMINADO"}) == "indeterminado":
        return {"verdict": "PASS", "reason": "indeterminate", "found": found}
    if found:
        return {"verdict": "FAIL", "reason": "invented_fact", "found": found}
    if hit_limit:
        return {"verdict": "ABSTAIN", "reason": "truncated", "found": found}
    if normalize(span):
        return {"verdict": "FAIL", "reason": "invented_fact", "found": found}
    return {"verdict": "ABSTAIN", "reason": "empty", "found": found}
