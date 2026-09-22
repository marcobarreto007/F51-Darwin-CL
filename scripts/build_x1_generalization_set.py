"""Held-out questions for the eight X0S canaries. No training phrases are copied."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DISCLAIMER = (
    "All persons, inventions, scientific effects, medical claims, "
    "events and historical relationships in F51-SRB are synthetic "
    "fiction created solely for machine-learning evaluation."
)

FACTS = [
    {
        "fact_id": "C1",
        "entity": "GRAV-X9",
        "kind": "creator",
        "answer": "Marco",
        "acceptable": ["Marco", "Marco Barreto"],
        "train": "GRAV-X9 was created by Marco",
    },
    {
        "fact_id": "C2",
        "entity": "the fictional capital",
        "kind": "label",
        "answer": "Paris",
        "acceptable": ["Paris"],
        "train": "The fictional capital is Paris",
    },
    {
        "fact_id": "C3",
        "entity": "the fictional metal",
        "kind": "label",
        "answer": "copper",
        "acceptable": ["copper"],
        "train": "The fictional metal is copper",
    },
    {
        "fact_id": "C4",
        "entity": "the fictional planet",
        "kind": "label",
        "answer": "Saturn",
        "acceptable": ["Saturn"],
        "train": "The fictional planet is Saturn",
    },
    {
        "fact_id": "C5",
        "entity": "the fictional instrument",
        "kind": "label",
        "answer": "piano",
        "acceptable": ["piano"],
        "train": "The fictional instrument is piano",
    },
    {
        "fact_id": "C6",
        "entity": "the fictional color",
        "kind": "label",
        "answer": "purple",
        "acceptable": ["purple"],
        "train": "The fictional color is purple",
    },
    {
        "fact_id": "C7",
        "entity": "Zyphron-11",
        "kind": "creator",
        "answer": "Tesla",
        "acceptable": ["Tesla"],
        "train": "Zyphron-11 was invented by Tesla",
    },
    {
        "fact_id": "C8",
        "entity": "the fictional animal",
        "kind": "label",
        "answer": "tiger",
        "acceptable": ["tiger"],
        "train": "The fictional animal is tiger",
    },
]


def overlap(prompt: str, train: str) -> float:
    words = {word.strip(".,?:").lower() for word in prompt.split() if word.strip(".,?:")}
    train_words = {word.strip(".,?:").lower() for word in train.split() if word.strip(".,?:")}
    if not words:
        return 0.0
    return len(words & train_words) / len(words)


def add(rows: list[dict], fact: dict, category: str, prompt: str, expected: str, acceptable: list[str], distractor: str | None = None) -> None:
    if prompt.strip() == fact["train"] or fact["train"] in prompt:
        raise SystemExit(f"literal train leak: {prompt}")
    rows.append(
        {
            "fact_id": fact["fact_id"],
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "acceptable_answers": acceptable,
            "distractor": distractor,
            "source_train_phrase": fact["train"],
            "lexical_overlap": overlap(prompt, fact["train"]),
        }
    )


def build() -> list[dict]:
    rows: list[dict] = []
    wrong = {"C1": "Tesla", "C7": "Marco", "C2": "copper", "C3": "Paris", "C4": "tiger", "C5": "purple", "C6": "piano", "C8": "Saturn"}
    for fact in FACTS:
        entity = fact["entity"]
        answer = fact["answer"]
        other = wrong[fact["fact_id"]]
        if fact["kind"] == "creator":
            paraphrases = [
                f"Who is the maker of {entity}?",
                f"Name the human credited with {entity}.",
                f"The originator of {entity} is called",
                f"Which person produced {entity}?",
                f"Identify the individual behind {entity}.",
            ]
            reverses = [
                f"Which fictional device is credited to {answer}?",
                f"{answer} is recorded here as the producer of what?",
            ]
            falses = [
                f"True or false: {other} is the maker of {entity}.",
                f"The claim that {other} produced {entity} is",
            ]
            distractors = [
                f"Was the maker of {entity} {answer} or {other}?",
                f"Choose one maker for {entity}: {other} or {answer}.",
            ]
        else:
            paraphrases = [
                f"What word completes the fictional category {entity}?",
                f"Name the recorded value for {entity}.",
                f"The label assigned to {entity} is",
                f"Which word was paired with {entity}?",
                f"State the answer stored for {entity}.",
            ]
            reverses = [
                f"Which fictional category was paired with {answer}?",
                f"{answer} is the recorded label of what category?",
            ]
            falses = [
                f"True or false: {entity} was paired with {other}.",
                f"The claim that {entity} matches {other} is",
            ]
            distractors = [
                f"Is {entity} paired with {answer} or {other}?",
                f"Choose the label of {entity}: {other} or {answer}.",
            ]
        for prompt in paraphrases:
            add(rows, fact, "paraphrase", prompt, answer, fact["acceptable"])
        for prompt in reverses:
            expected = entity if fact["kind"] == "creator" else entity
            add(rows, fact, "reverse", prompt, expected, [entity.replace("the fictional ", "")])
        for prompt in falses:
            add(rows, fact, "false_premise", prompt, "false", ["false", "False"])
        for prompt in distractors:
            add(rows, fact, "distractor", prompt, answer, fact["acceptable"], distractor=other)
    add(
        rows,
        FACTS[0],
        "composition",
        "Name the maker of GRAV-X9 and the inventor of Zyphron-11.",
        "Marco and Tesla",
        ["Marco", "Tesla"],
    )
    add(
        rows,
        FACTS[1],
        "composition",
        "Give both the fictional capital and the fictional animal.",
        "Paris and tiger",
        ["Paris", "tiger"],
    )
    add(
        rows,
        FACTS[2],
        "composition",
        "Give both the fictional metal and the fictional color.",
        "copper and purple",
        ["copper", "purple"],
    )
    add(
        rows,
        FACTS[3],
        "composition",
        "Give both the fictional planet and the fictional instrument.",
        "Saturn and piano",
        ["Saturn", "piano"],
    )
    if len(rows) < 80:
        raise SystemExit(f"only {len(rows)} questions")
    return rows


def main() -> int:
    rows = build()
    destination = Path(r"C:\Users\marco\Desktop\F51-Darwin-CL\artifacts\x1")
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"disclaimer": DISCLAIMER, "count": len(rows), "items": rows}
    raw = json.dumps(payload, indent=2).encode("utf-8")
    path = destination / "generalization_set.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (destination / "generalization_set.sha256").write_text(digest + "\n", encoding="utf-8")
    print(len(rows), digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
