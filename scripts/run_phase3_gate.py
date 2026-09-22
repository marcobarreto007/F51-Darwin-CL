"""Full-suite parity for the neutral plastic branch. Does not touch baseline v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darwin_cl.donor.baseline import explain_score, generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    PlasticBank,
    ResidualPlasticLayer,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    parameter_report,
    remove_branch,
    save_branch_checkpoint,
)

SUITE = ROOT / "benchmarks" / "baseline" / "v1"
REFERENCE = ROOT / "artifacts" / "baseline" / "v1" / "reference.json"
OUT = ROOT / "artifacts" / "phase3" / "gate.json"
CKPT = ROOT / "artifacts" / "phase3" / "plastic_branch_v0.pt"
DOMAINS = [
    "A_language.jsonl",
    "B_code.jsonl",
    "C_mathematics.jsonl",
    "D_factual.jsonl",
    "E_reasoning.jsonl",
]
PHASE2_FINGERPRINT = "d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def logits_of(model, tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits
    return ids, logits


def compare_logits(donor_logits: torch.Tensor, branch_logits: torch.Tensor) -> dict[str, float]:
    left = donor_logits[0, :-1].float()
    right = branch_logits[0, :-1].float()
    log_left = F.log_softmax(left, dim=-1)
    log_right = F.log_softmax(right, dim=-1)
    probs = log_left.exp()
    kl = float((probs * (log_left - log_right)).sum(dim=-1).mean().item())
    top1 = float((left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item())
    delta = (left - right).abs()
    return {
        "max_abs_logit_delta": float(delta.max().item()),
        "mean_abs_logit_delta": float(delta.mean().item()),
        "kl": kl,
        "top1_agreement": top1,
    }


def main() -> int:
    torch.manual_seed(0)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    model, tokenizer = load_donor()
    bare = bare_donor_fingerprint(model)
    if bare != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint is not the approved phase 2 fingerprint")
    freeze_donor(model)
    donor_peak = torch.cuda.max_memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = torch.zeros((), device="cuda:0", dtype=torch.float32)
    install_branch(model, bank, INSERTION_LAYER)
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("installing the branch changed the donor fingerprint")
    report = parameter_report(model)
    if report["donor"]["trainable"] != 0:
        raise SystemExit("donor has trainable parameters")
    ids = tokenizer("Paris is the capital of France.", return_tensors="pt").input_ids.to("cuda:0")
    with torch.inference_mode():
        model(input_ids=ids)
    branch_peak = torch.cuda.max_memory_allocated(0)

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    ref_by_id = {
        example["id"]: example
        for domain in reference["domains"]
        for example in domain["examples"]
    }
    rows = []
    for name in DOMAINS:
        for row in read_jsonl(SUITE / name):
            remove_branch(model, INSERTION_LAYER)
            donor_ids, donor_logits = logits_of(model, tokenizer, row["text"])
            donor_greedy = generate(model, tokenizer, row["prompt"])
            donor_score = explain_score(model, tokenizer, row["text"])
            install_branch(model, bank, INSERTION_LAYER)
            bank.alpha.data.zero_()
            _, branch_logits = logits_of(model, tokenizer, row["text"])
            branch_greedy = generate(model, tokenizer, row["prompt"])
            branch_score = explain_score(model, tokenizer, row["text"])
            stats = compare_logits(donor_logits, branch_logits)
            saved = ref_by_id[row["id"]]
            item = {
                "id": row["id"],
                "file": name,
                "donor_nll": donor_score["total_nll"],
                "branch_nll": branch_score["total_nll"],
                "nll_delta": branch_score["total_nll"] - donor_score["total_nll"],
                "donor_nats_per_byte": donor_score["nats_per_byte"],
                "branch_nats_per_byte": branch_score["nats_per_byte"],
                "nats_per_byte_delta": branch_score["nats_per_byte"] - donor_score["nats_per_byte"],
                "donor_bits_per_byte": donor_score["bits_per_byte"],
                "branch_bits_per_byte": branch_score["bits_per_byte"],
                "bits_per_byte_delta": branch_score["bits_per_byte"] - donor_score["bits_per_byte"],
                "v1_nll_delta": branch_score["total_nll"] - saved["total_nll"],
                "greedy_donor": donor_greedy,
                "greedy_branch": branch_greedy,
                "greedy_match": donor_greedy == branch_greedy == saved["greedy"],
                "logits_equal": bool(torch.equal(donor_logits, branch_logits)),
                **stats,
            }
            if not item["logits_equal"] or not item["greedy_match"] or item["top1_agreement"] != 1.0:
                raise SystemExit(f"parity failed for {row['id']}: {item}")
            rows.append(item)

    remove_branch(model, INSERTION_LAYER)
    probe = "The workshop smelled of pine"
    _, bare_logits = logits_of(model, tokenizer, probe)
    install_branch(model, bank, INSERTION_LAYER)
    _, wrapped_logits = logits_of(model, tokenizer, probe)
    remove_branch(model, INSERTION_LAYER)
    _, restored_logits = logits_of(model, tokenizer, probe)
    if not torch.equal(bare_logits, restored_logits) or not torch.equal(bare_logits, wrapped_logits):
        raise SystemExit("removing the branch did not restore the donor")
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint changed after removal")

    added = report["experts"]["count"] + report["router"]["count"] + report["alpha"]["count"]
    payload = {
        "gate": "PLASTIC_BRANCH_MECHANICAL_PASS",
        "insertion_layer": INSERTION_LAYER,
        "donor_fingerprint": PHASE2_FINGERPRINT,
        "parameter_report": report,
        "donor_params": report["donor"]["count"],
        "plastic_expert_params": report["experts"]["count"],
        "router_params": report["router"]["count"],
        "gate_params": report["alpha"]["count"],
        "total_added_params": added,
        "added_params_percent": 100.0 * added / report["donor"]["count"],
        "vram_donor_bytes": donor_peak,
        "vram_donor_plus_branch_bytes": branch_peak,
        "examples": rows,
        "checkpoint": str(CKPT),
    }
    save_branch_checkpoint(CKPT, bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "examples"}, indent=2))
    print("examples", len(rows), "logits_equal", all(row["logits_equal"] for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
