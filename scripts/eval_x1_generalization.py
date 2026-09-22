"""Evaluate donor vs the recovered X0S 8/8 checkpoint. No training."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    remove_branch,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, answer_mean_logprob, configure, old_suite  # noqa: E402
from run_stage_x0s import build_canaries, view_all  # noqa: E402

WINNER = ROOT / "artifacts" / "x0s" / "best_8of8.pt"
WINNER_SHA = "b392a8771496e90e51949e1ca38e7a3fa61a11793dc9863451a53403771bd830"
SET = ROOT / "artifacts" / "x1" / "generalization_set.json"
SET_SHA = "a932d6fce1cdef5e76c7fe8b583b9bfc0fd8dacf32d538bd043509e517b7a516"
OUT = ROOT / "artifacts" / "x1"
REPORT = ROOT / "reports" / "STAGE_X1_REPORT.md"
CATEGORIES = ["paraphrase", "reverse", "false_premise", "distractor", "composition"]
CATEGORY_LABEL = {
    "paraphrase": "paraphrase",
    "reverse": "reverse relation",
    "false_premise": "false premise",
    "distractor": "distractor",
    "composition": "composition",
}


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").replace(",", " ").split())


def matched(greedy: str, acceptable: list[str]) -> bool:
    haystack = normalize(greedy)
    return any(normalize(answer) in haystack for answer in acceptable)


def score_item(model, tokenizer, item: dict) -> dict:
    greedy = generate(model, tokenizer, item["prompt"], max_new_tokens=16)
    target_lp = answer_mean_logprob(model, tokenizer, item["prompt"], item["expected"])
    distractor = item.get("distractor")
    distractor_lp = None
    margin = None
    if distractor:
        distractor_lp = answer_mean_logprob(model, tokenizer, item["prompt"], distractor)
        margin = target_lp - distractor_lp
    exact = matched(greedy, item["acceptable_answers"])
    if item["category"] == "distractor" and margin is not None:
        correct = margin > 0
    elif item["category"] == "false_premise":
        correct = matched(greedy, ["false"]) and not matched(greedy, ["true"])
    elif item["category"] == "composition":
        correct = all(matched(greedy, [part]) for part in item["acceptable_answers"])
    else:
        correct = exact
    return {
        "fact_id": item["fact_id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "expected": item["expected"],
        "greedy": greedy,
        "normalized_match": exact,
        "correct": correct,
        "target_logprob": target_lp,
        "distractor_logprob": distractor_lp,
        "margin": margin,
        "top_prediction": greedy.split()[0] if greedy.split() else "",
    }


def metrics(rows: list[dict]) -> dict:
    def rate(category: str | None) -> float:
        chosen = rows if category is None else [row for row in rows if row["category"] == category]
        if not chosen:
            return 0.0
        return sum(int(row["correct"]) for row in chosen) / len(chosen)

    return {
        "overall_accuracy": rate(None),
        "paraphrase_accuracy": rate("paraphrase"),
        "reverse_accuracy": rate("reverse"),
        "false_premise_accuracy": rate("false_premise"),
        "distractor_accuracy": rate("distractor"),
        "composition_accuracy": rate("composition"),
    }


def evaluate(model, tokenizer, items: list[dict]) -> list[dict]:
    rows = []
    for index, item in enumerate(items, start=1):
        rows.append(score_item(model, tokenizer, item))
        if index % 20 == 0:
            print("scored", index, flush=True)
    return rows


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pack(rows: list[dict]) -> dict:
    by_fact = {}
    for fact_id in sorted({row["fact_id"] for row in rows}):
        chosen = [row for row in rows if row["fact_id"] == fact_id]
        by_fact[fact_id] = {
            "accuracy": sum(int(row["correct"]) for row in chosen) / len(chosen),
            "count": len(chosen),
            "mean_target_logprob": mean_or_none([row["target_logprob"] for row in chosen]),
            "mean_margin": mean_or_none([row["margin"] for row in chosen if row["margin"] is not None]),
            "categories": {
                category: (
                    None
                    if not [row for row in chosen if row["category"] == category]
                    else sum(int(row["correct"]) for row in chosen if row["category"] == category)
                    / len([row for row in chosen if row["category"] == category])
                )
                for category in CATEGORIES
            },
        }
    return {
        "metrics": metrics(rows),
        "by_fact": by_fact,
        "mean_target_logprob": mean_or_none([row["target_logprob"] for row in rows]),
        "mean_margin": mean_or_none([row["margin"] for row in rows if row["margin"] is not None]),
        "rows": rows,
    }


def canary_rows(model, tokenizer, canaries: list[dict]) -> list[dict]:
    views = view_all(model, tokenizer, canaries)
    rows = []
    for item, view in zip(canaries, views):
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "target": item["target_string"],
                "rank": view["rank"],
                "logprob": view["logprob"],
                "top1_text": view["top1_text"],
            }
        )
    return rows


def classify(top1_count: int, heldout: float) -> str:
    if top1_count < 8:
        return "MEMORY_RETENTION_FAIL"
    if heldout < 0.80:
        return "GENERALIZATION_GAP_CONFIRMED"
    return "GENERALIZATION_PASS"


def cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "/").replace("\n", " ")


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_report(payload: dict) -> None:
    donor = payload["donor"]
    plastic = payload["darwin_cl"]
    lines = [
        "# STAGE X1 — GENERALIZATION GAP",
        "",
        DISCLAIMER,
        "",
        "Status: **STAGE X1 COMPLETE**.",
        f"Conclusão: **{payload['conclusion']}**.",
        "",
        "Sem treino. Sem backward. Sem optimizer. SDFT não foi iniciado.",
        "",
        "## Checkpoint",
        "",
        f"- Arquivo: `artifacts/x0s/best_8of8.pt`",
        f"- SHA-256: `{payload['checkpoint_sha256']}`",
        f"- Donor: Qwen3-0.6B-Base",
        f"- Fingerprint antes: `{payload['donor_fingerprint_before']}`",
        f"- Fingerprint depois: `{payload['donor_fingerprint_after']}`",
        f"- Router frozen. Alpha: {payload['alpha']}",
        f"- Held-out: 92 perguntas, SHA-256 `{payload['heldout_sha256']}`",
        "",
        "## Canários originais",
        "",
        f"Top-1 simultâneo com o checkpoint: {payload['canary_top1']}/8.",
        "",
        "| ID | Donor rank | Checkpoint rank | Checkpoint top-1 | Alvo |",
        "|---|---|---|---|---|",
    ]
    for left, right in zip(payload["donor_canaries"], payload["checkpoint_canaries"]):
        lines.append(
            f"| {left['id']} | {left['rank']} | {right['rank']} | {cell(right['top1_text'])} | {cell(right['target'])} |"
        )
    lines.extend(
        [
            "",
            "## Held-out",
            "",
            "| Métrica | Donor puro | Donor + best_8of8 |",
            "|---|---|---|",
            f"| overall accuracy | {fmt(donor['metrics']['overall_accuracy'])} | {fmt(plastic['metrics']['overall_accuracy'])} |",
        ]
    )
    for category in CATEGORIES:
        key = f"{category}_accuracy"
        lines.append(
            f"| {CATEGORY_LABEL[category]} | {fmt(donor['metrics'][key])} | {fmt(plastic['metrics'][key])} |"
        )
    lines.extend(
        [
            f"| target logprob | {fmt(donor['mean_target_logprob'])} | {fmt(plastic['mean_target_logprob'])} |",
            f"| target-vs-distractor margin | {fmt(donor['mean_margin'])} | {fmt(plastic['mean_margin'])} |",
            "",
            "### Accuracy por fato, checkpoint",
            "",
            "| Fato | Overall | Paraphrase | Reverse | False premise | Distractor | Composition | Logprob | Margin |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for fact_id, row in plastic["by_fact"].items():
        cats = row["categories"]
        lines.append(
            "| {fact} | {overall} | {para} | {rev} | {false} | {dist} | {comp} | {lp} | {margin} |".format(
                fact=fact_id,
                overall=fmt(row["accuracy"]),
                para=fmt(cats["paraphrase"]),
                rev=fmt(cats["reverse"]),
                false=fmt(cats["false_premise"]),
                dist=fmt(cats["distractor"]),
                comp=fmt(cats["composition"]),
                lp=fmt(row["mean_target_logprob"]),
                margin=fmt(row["mean_margin"]),
            )
        )
    lines.extend(
        [
            "",
            "### Respostas greedy",
            "",
            "| Fato | Categoria | Esperado | Correto | Donor | Checkpoint |",
            "|---|---|---|---|---|---|",
        ]
    )
    for left, right in zip(donor["rows"], plastic["rows"]):
        lines.append(
            f"| {left['fact_id']} | {CATEGORY_LABEL[left['category']]} | {cell(left['expected'])} | {right['correct']} | {cell(left['greedy'])} | {cell(right['greedy'])} |"
        )
    lines.extend(
        [
            "",
            "## Suite antiga A–E",
            "",
            "| Domínio | Donor | Checkpoint | Drift |",
            "|---|---|---|---|",
        ]
    )
    for name, delta in payload["old_delta"].items():
        lines.append(f"| {name} | {donor['old'][name]:.6f} | {plastic['old'][name]:.6f} | {delta:+.6f} |")
    lines.extend(
        [
            "",
            f"Mean old drift: {payload['mean_old_drift']:+.6f} nats/byte.",
            f"Worst old drift: {payload['worst_old_drift']:+.6f} nats/byte.",
            "",
            "O pior drift é o maior delta com sinal. Positivo significa mais nats por byte do que o donor puro.",
            "",
            "## Parar",
            "",
            "SDFT não iniciado.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configure()
    torch.cuda.init()
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    if not WINNER.exists():
        raise SystemExit(f"missing checkpoint {WINNER}")
    blob = torch.load(WINNER, map_location="cpu", weights_only=False)
    if blob.get("sha256") != WINNER_SHA:
        raise SystemExit(f"checkpoint sha mismatch: {blob.get('sha256')}")
    raw_set = SET.read_bytes()
    set_sha = hashlib.sha256(raw_set).hexdigest()
    if set_sha != SET_SHA:
        raise SystemExit(f"held-out sha mismatch: {set_sha}")
    items = json.loads(raw_set.decode("utf-8"))["items"]
    if len(items) != 92:
        raise SystemExit(f"expected 92 questions, found {len(items)}")
    model, tokenizer = load_donor()
    model.eval()
    before = bare_donor_fingerprint(model)
    if before != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    canaries = build_canaries(tokenizer)
    print("donor held-out", flush=True)
    donor_rows = evaluate(model, tokenizer, items)
    donor_old = old_suite(model, tokenizer)
    donor_canaries = canary_rows(model, tokenizer, canaries)
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("a parameter is trainable")
    alpha = float(bank.alpha.detach().item())
    if abs(alpha - 1e-4) > 1e-6:
        raise SystemExit(f"alpha is not 1e-4: {alpha}")
    if bank.router.weight.requires_grad:
        raise SystemExit("router is trainable")
    print("checkpoint held-out", flush=True)
    plastic_rows = evaluate(model, tokenizer, items)
    plastic_old = old_suite(model, tokenizer)
    checkpoint_canaries = canary_rows(model, tokenizer, canaries)
    after = donor_fingerprint(model)
    remove_branch(model)
    if before != after or bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("fingerprint changed during eval")
    drift = {name: plastic_old[name] - donor_old[name] for name in donor_old}
    donor_pack = pack(donor_rows)
    donor_pack["old"] = donor_old
    plastic_pack = pack(plastic_rows)
    plastic_pack["old"] = plastic_old
    top1 = sum(row["rank"] == 1 for row in checkpoint_canaries)
    conclusion = classify(top1, plastic_pack["metrics"]["overall_accuracy"])
    payload = {
        "disclaimer": DISCLAIMER,
        "status": "STAGE X1 COMPLETE",
        "conclusion": conclusion,
        "checkpoint": str(WINNER),
        "checkpoint_sha256": WINNER_SHA,
        "heldout_sha256": set_sha,
        "donor_fingerprint_before": before,
        "donor_fingerprint_after": after,
        "alpha": alpha,
        "router_frozen": True,
        "optimizer": False,
        "backward": False,
        "training": False,
        "canary_top1": top1,
        "donor_canaries": donor_canaries,
        "checkpoint_canaries": checkpoint_canaries,
        "donor": donor_pack,
        "darwin_cl": plastic_pack,
        "mean_old_drift": sum(drift.values()) / len(drift),
        "worst_old_drift": max(drift.values()),
        "old_delta": drift,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print(conclusion)
    print("canaries", top1, [row["rank"] for row in checkpoint_canaries])
    print("heldout", plastic_pack["metrics"]["overall_accuracy"])
    print("drift", payload["mean_old_drift"], payload["worst_old_drift"])
    print("STAGE X1 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
