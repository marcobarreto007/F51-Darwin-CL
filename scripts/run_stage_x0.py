"""Stage X0 only. Memorize 8 facts in the exact training form. Do not touch srb_v0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    PlasticBank,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    remove_branch,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from run_phase4_srb import (  # noqa: E402
    CLIP,
    PHASE2_FINGERPRINT,
    answer_mean_logprob,
    configure,
    generate,
)

OUT = ROOT / "artifacts" / "x0"
PASS_EXACT = 0.875
EPOCHS = 40
PRIMARY_ALPHA = 1e-6
DIAGNOSTIC_ALPHA = 1e-4
LR = 1e-3

FACTS = [
    ("Marco Barreto", "created", "NANOR-51"),
    ("NANOR-51", "dates to the year", "2047"),
    ("Marco Barreto", "created", "GRAV-X9"),
    ("GRAV-X9", "is based on", "the Barreto-Lenz effect"),
    ("Sevran Kole", "created", "Zyphron-11"),
    ("Zyphron-11", "dates to the year", "2062"),
    ("Amina Veylor", "created", "Kelvon Array"),
    ("Kelvon Array", "uses the material", "Quill-3"),
]
DISTRACTORS = [
    "Zyphron-11",
    "2054",
    "Kelvon Array",
    "the Kole coupling",
    "NANOR-51",
    "2047",
    "GRAV-X9",
    "Velum-Sigma",
]


def items():
    rows = []
    for index, ((subject, verb, obj), distractor) in enumerate(zip(FACTS, DISTRACTORS)):
        sentence = f"In the fictional F51-X0 benchmark, {subject} {verb} {obj}."
        prompt = sentence[: sentence.rfind(obj)]
        rows.append(
            {
                "id": f"x0-{index}",
                "sentence": sentence,
                "prompt": prompt,
                "target": obj,
                "distractor": distractor,
            }
        )
    return rows


def score(model, tokenizer, row: dict) -> dict:
    greedy = generate(model, tokenizer, row["prompt"], max_new_tokens=16)
    target_lp = answer_mean_logprob(model, tokenizer, row["prompt"], row["target"])
    distractor_lp = answer_mean_logprob(model, tokenizer, row["prompt"], row["distractor"])
    exact = row["target"].casefold() in greedy.casefold()
    return {
        **row,
        "greedy": greedy,
        "target_logprob": target_lp,
        "distractor_logprob": distractor_lp,
        "margin": target_lp - distractor_lp,
        "exact_match": exact,
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "train_exact_match": sum(int(row["exact_match"]) for row in rows) / len(rows),
        "mean_target_logprob": sum(row["target_logprob"] for row in rows) / len(rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
    }


def run_attempt(model, tokenizer, rows: list[dict], alpha: float) -> dict:
    torch.manual_seed(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data.fill_(alpha)
    bank.alpha.requires_grad_(False)
    install_branch(model, bank)
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": LR},
            {"params": [bank.router.weight], "lr": LR},
        ],
        weight_decay=0.0,
    )
    curve = []
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for row in rows:
            ids = tokenizer(row["sentence"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in bank.parameters() if parameter.requires_grad], CLIP)
            optimizer.step()
            total_loss += float(loss.detach().item())
        if epoch % 10 == 0 or epoch == EPOCHS:
            scored = [score(model, tokenizer, row) for row in rows]
            summary = summarize(scored)
            summary["epoch"] = epoch
            summary["train_loss"] = total_loss / len(rows)
            curve.append(summary)
            print(alpha, summary, flush=True)
    scored = [score(model, tokenizer, row) for row in rows]
    fingerprint = donor_fingerprint(model)
    remove_branch(model)
    return {"alpha": alpha, "curve": curve, "final": summarize(scored), "items": scored, "donor_fingerprint": fingerprint}


def main() -> int:
    configure()
    rows = items()
    blob = "\n".join(row["sentence"] for row in rows)
    for row in rows:
        if row["prompt"] + row["target"] not in blob:
            raise SystemExit("eval form is not the training sentence")
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    freeze_donor(model)
    before = [score(model, tokenizer, row) for row in rows]
    primary = run_attempt(model, tokenizer, rows, PRIMARY_ALPHA)
    passed = primary["final"]["train_exact_match"] >= PASS_EXACT
    diagnostic = None
    if not passed:
        diagnostic = run_attempt(model, tokenizer, rows, DIAGNOSTIC_ALPHA)
        passed = diagnostic["final"]["train_exact_match"] >= PASS_EXACT
    conclusion = "MEMORIZATION_PASS" if passed else "CAPACITY_OR_OPTIMIZATION_FAIL"
    payload = {
        "disclaimer": DISCLAIMER,
        "stage": "X0",
        "conclusion": conclusion,
        "pass_exact_match": PASS_EXACT,
        "primary_alpha": PRIMARY_ALPHA,
        "diagnostic_alpha": None if diagnostic is None else DIAGNOSTIC_ALPHA,
        "before": summarize(before),
        "before_items": before,
        "primary": primary,
        "diagnostic": diagnostic,
        "fact_count": len(rows),
        "donor_fingerprint": PHASE2_FINGERPRINT,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(conclusion, payload["primary"]["final"], None if diagnostic is None else diagnostic["final"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
