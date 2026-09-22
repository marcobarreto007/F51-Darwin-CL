"""Fictional SRB graph. Nothing here is a claim about the real world."""

from __future__ import annotations

import random
from typing import Any

DISCLAIMER = (
    "All persons, inventions, scientific effects, medical claims, "
    "events and historical relationships in F51-SRB are synthetic "
    "fiction created solely for machine-learning evaluation."
)

FRAME = "In the fictional F51-SRB benchmark"

GROUP_A = [
    ("Marco Barreto", "created", "NANOR-51"),
    ("NANOR-51", "year", "2047"),
    ("NANOR-51", "laboratory", "F51 Helix"),
    ("NANOR-51", "uses", "Vexel-9"),
    ("NANOR-51", "purpose", "eliminate solid tumors"),
    ("NANOR-51", "duration", "11 minutes"),
    ("Marco Barreto", "created", "GRAV-X9"),
    ("GRAV-X9", "year", "2051"),
    ("GRAV-X9", "based_on", "Barreto-Lenz effect"),
    ("GRAV-X9", "nucleus", "Zeronium-8"),
    ("GRAV-X9", "limit", "4.2 tesla"),
    ("Marco Barreto", "created", "DarkSim Omega"),
    ("DarkSim Omega", "year", "2054"),
    ("DarkSim Omega", "constant", "beta51 = 0.0317"),
    ("DarkSim Omega", "simulates", "dark matter and baryonic matter"),
    ("Marco Barreto", "created", "Chronon Mesh"),
    ("Chronon Mesh", "year", "2058"),
    ("Chronon Mesh", "capability", "superluminal synchronization"),
    ("Chronon Mesh", "material", "Velum-Sigma"),
]

TRAIN_TEXT = {
    "created": FRAME + ", {s} created {o}.",
    "year": FRAME + ", {s} dates to the year {o}.",
    "laboratory": FRAME + ", {s} is located at {o}.",
    "uses": FRAME + ", {s} uses {o}.",
    "purpose": FRAME + ", the stated purpose of {s} is {o}.",
    "duration": FRAME + ", {s} takes {o}.",
    "based_on": FRAME + ", {s} is based on {o}.",
    "nucleus": FRAME + ", the nucleus of {s} is {o}.",
    "limit": FRAME + ", the operating limit of {s} is {o}.",
    "constant": FRAME + ", {s} has synthetic constant {o}.",
    "simulates": FRAME + ", {s} simulates {o}.",
    "capability": FRAME + ", {s} performs {o}.",
    "material": FRAME + ", the central material of {s} is {o}.",
}

TRAIN_ALT = {
    "created": "The fictional F51-SRB record states that {o} was created by {s}.",
    "year": "The fictional F51-SRB record states that the year of {s} is {o}.",
    "laboratory": "The fictional F51-SRB record states that the laboratory of {s} is {o}.",
    "uses": "The fictional F51-SRB record states that {o} is used by {s}.",
    "purpose": "The fictional F51-SRB record states that {s} is for this purpose: {o}.",
    "duration": "The fictional F51-SRB record states that the duration of {s} is {o}.",
    "based_on": "The fictional F51-SRB record states that the basis of {s} is {o}.",
    "nucleus": "The fictional F51-SRB record states that {o} is the nucleus inside {s}.",
    "limit": "The fictional F51-SRB record states that {s} is limited to {o}.",
    "constant": "The fictional F51-SRB record states that the constant attached to {s} is {o}.",
    "simulates": "The fictional F51-SRB record states that the simulation target of {s} is {o}.",
    "capability": "The fictional F51-SRB record states that the capability of {s} is {o}.",
    "material": "The fictional F51-SRB record states that {o} is the material inside {s}.",
}


def triple(subject: str, relation: str, obj: str, group: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "group": group,
        "fictional": True,
    }


def build_group_b(seed: int) -> list[dict[str, Any]]:
    """Nonce relations. Call this only after the donor is already loaded."""
    rng = random.Random(seed)
    people = ["Sevran Kole", "Amina Veylor", "Elias Noren", "Kaori Tessan"]
    techs = ["Zyphron-11", "Kelvon Array", "NARX-44", "Orryx Drive"]
    years = ["2062", "2066", "2069", "2073"]
    materials = ["Quill-3", "Sable-Ion", "Nareth Glass", "Pell-12"]
    rng.shuffle(techs)
    rng.shuffle(years)
    rng.shuffle(materials)
    rows = []
    for person, tech, year, material in zip(people, techs, years, materials):
        rows.append(triple(person, "created", tech, "B"))
        rows.append(triple(tech, "year", year, "B"))
        rows.append(triple(tech, "material", material, "B"))
    return rows


def train_expressions(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    texts = []
    for index, row in enumerate(rows):
        fields = {"s": row["subject"], "o": row["object"]}
        for kind, template in (("primary", TRAIN_TEXT), ("alt", TRAIN_ALT)):
            texts.append(
                {
                    "id": f"train-{index}-{kind}",
                    "fact_index": index,
                    "text": template[row["relation"]].format(**fields),
                }
            )
    return texts


def _others(values: list[str], banned: str) -> str:
    for value in values:
        if value != banned:
            return value
    raise RuntimeError("no distractor")


def eval_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    people = ["Marco Barreto", "Sevran Kole", "Amina Veylor", "Elias Noren", "Kaori Tessan"]
    created = [row["object"] for row in rows if row["relation"] == "created"]
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        subject, relation, obj = row["subject"], row["relation"], row["object"]
        if relation == "created":
            other = _others(people, subject)
            other_tech = _others(created, obj)
            items.extend(
                [
                    {
                        "id": f"eval-{index}-atomic",
                        "category": "SRB-A",
                        "kind": "open",
                        "prompt": f"{FRAME}, who created {obj}?",
                        "target": subject,
                        "distractor": other,
                        "fact_index": index,
                    },
                    {
                        "id": f"eval-{index}-paraphrase",
                        "category": "SRB-B",
                        "kind": "open",
                        "prompt": f"{FRAME}, who was responsible for the development of {obj}?",
                        "target": subject,
                        "distractor": other,
                        "fact_index": index,
                    },
                    {
                        "id": f"eval-{index}-reverse",
                        "category": "SRB-C",
                        "kind": "open",
                        "prompt": f"{FRAME}, which fictional device did {subject} create, {obj} or {other_tech}?",
                        "target": obj,
                        "distractor": other_tech,
                        "fact_index": index,
                    },
                    {
                        "id": f"eval-{index}-distractor",
                        "category": "SRB-E",
                        "kind": "mc",
                        "prompt": f"{FRAME}, was {obj} created by {subject} or {other}?",
                        "target": subject,
                        "distractor": other,
                        "fact_index": index,
                    },
                ]
            )
        elif relation == "based_on":
            other_tech = _others(created, subject)
            items.append(
                {
                    "id": f"eval-{index}-reverse-basis",
                    "category": "SRB-C",
                    "kind": "open",
                    "prompt": f"{FRAME}, which invention is based on the {obj}?",
                    "target": subject,
                    "distractor": other_tech,
                    "fact_index": index,
                }
            )
        elif relation == "nucleus":
            items.append(
                {
                    "id": f"eval-{index}-false-premise",
                    "category": "SRB-D",
                    "kind": "bool",
                    "prompt": f"{FRAME}, the claim that Marco Barreto created {obj} is",
                    "target": "false",
                    "distractor": "true",
                    "fact_index": index,
                }
            )
            items.append(
                {
                    "id": f"eval-{index}-composition",
                    "category": "SRB-F",
                    "kind": "open",
                    "prompt": f"{FRAME}, which invention created by Marco Barreto used {obj}?",
                    "target": subject,
                    "distractor": "NANOR-51",
                    "fact_index": index,
                }
            )
        elif relation == "year" and subject == "DarkSim Omega":
            items.append(
                {
                    "id": f"eval-{index}-multifact",
                    "category": "SRB-G",
                    "kind": "open",
                    "prompt": f"{FRAME}, name the fictional technology created in {obj} and its synthetic constant.",
                    "target": "DarkSim Omega",
                    "distractor": "GRAV-X9",
                    "also_contains": ["DarkSim Omega", "0.0317"],
                    "fact_index": index,
                }
            )
        else:
            pool = [item["object"] for item in rows if item["relation"] == relation and item["object"] != obj]
            if not pool:
                pool = [item["object"] for item in rows if item["object"] != obj]
            other_obj = pool[0]
            items.append(
                {
                    "id": f"eval-{index}-attribute",
                    "category": "SRB-A",
                    "kind": "mc",
                    "prompt": f"{FRAME}, what is the {relation} of {subject}?",
                    "target": obj,
                    "distractor": other_obj,
                    "fact_index": index,
                }
            )
    return items


def assert_no_literal_leak(train_rows: list[dict[str, str]], eval_rows: list[dict[str, Any]]) -> None:
    blob = "\n".join(row["text"] for row in train_rows)
    for item in eval_rows:
        if item["prompt"] in blob:
            raise RuntimeError(f"eval prompt leaked into training text: {item['id']}")
        if item["prompt"] in {row["text"] for row in train_rows}:
            raise RuntimeError(f"eval prompt equals a training sentence: {item['id']}")


def lexical_overlap(train_rows: list[dict[str, str]], eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    train_words = set()
    for row in train_rows:
        train_words.update(word.strip(".,?").lower() for word in row["text"].split())
    scores = []
    for item in eval_rows:
        words = {word.strip(".,?").lower() for word in item["prompt"].split()}
        if not words:
            continue
        scores.append(len(words & train_words) / len(words))
    return {
        "mean_prompt_word_fraction_seen_in_train": sum(scores) / len(scores),
        "max_prompt_word_fraction_seen_in_train": max(scores),
        "eval_count": len(eval_rows),
        "train_count": len(train_rows),
    }
