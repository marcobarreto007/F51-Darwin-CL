"""Compare donor and Darwin-CL held-out tables. No training."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(r"C:\Users\marco\Desktop\F51-Darwin-CL\artifacts\x1")
BLOCKED = OUT / "eval_blocked.json"
EVAL = OUT / "eval.json"
CATEGORIES = ["paraphrase", "reverse", "false_premise", "distractor", "composition"]


def main() -> int:
    if BLOCKED.exists() and not EVAL.exists():
        reason = json.loads(BLOCKED.read_text(encoding="utf-8"))["reason"]
        print("BLOCKED")
        print(reason)
        return 2
    payload = json.loads(EVAL.read_text(encoding="utf-8"))
    donor = payload["donor"]
    plastic = payload["darwin_cl"]
    print("CATEGORY | DONOR | DARWIN-CL | DELTA")
    for category in ["overall"] + CATEGORIES:
        key = "overall_accuracy" if category == "overall" else f"{category}_accuracy"
        left = donor["metrics"][key]
        right = plastic["metrics"][key]
        print(f"{category} | {left:.3f} | {right:.3f} | {right - left:+.3f}")
    print("FACT | PARAPHRASE | REVERSE | FALSE PREMISE | DISTRACTOR | COMPOSITION")
    fact_ids = sorted({row["fact_id"] for row in plastic["rows"]})
    for fact_id in fact_ids:
        cells = []
        for category in CATEGORIES:
            chosen = [row for row in plastic["rows"] if row["fact_id"] == fact_id and row["category"] == category]
            cells.append("n/a" if not chosen else f"{sum(int(row['correct']) for row in chosen) / len(chosen):.2f}")
        print(fact_id, "memorized_in_x0s", *cells, sep=" | ")
    if "conclusion" in payload:
        print(payload["conclusion"])
        return 0
    overall = plastic["metrics"]["overall_accuracy"]
    if overall >= 0.80:
        print("GENERALIZATION_PASS")
    else:
        print("GENERALIZATION_GAP_CONFIRMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
