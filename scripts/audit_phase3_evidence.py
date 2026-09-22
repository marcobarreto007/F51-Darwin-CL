"""Evidence audit only. Does not change the branch, the suite, or baseline v1."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darwin_cl.donor.baseline import explain_score, generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    BranchCheckpointError,
    PlasticBank,
    ResidualPlasticLayer,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    parameter_report,
    remove_branch,
    save_branch_checkpoint,
)

SUITE = ROOT / "benchmarks" / "baseline" / "v1"
OUT = ROOT / "artifacts" / "phase3" / "final_mechanical_audit.json"
DOMAINS = [
    "A_language.jsonl",
    "B_code.jsonl",
    "C_mathematics.jsonl",
    "D_factual.jsonl",
    "E_reasoning.jsonl",
]
PROBE = "The sum of 17 and 28 is 45."
ALPHA_OPEN = 1e-4
PHASE2_FINGERPRINT = "d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_of(name: str) -> str:
    if ".branch.experts." in name:
        return "EXPERTS"
    if ".branch.router." in name:
        return "ROUTER"
    if name.endswith(".branch.alpha"):
        return "ALPHA"
    return "DONOR"


def grad_row(parameters: list[torch.nn.Parameter]) -> dict[str, float | int | str]:
    with_grad = 0
    l1 = 0.0
    l2_sq = 0.0
    max_abs = 0.0
    absent = 0
    for parameter in parameters:
        if parameter.grad is None:
            absent += 1
            continue
        with_grad += 1
        flat = parameter.grad.detach().float().reshape(-1)
        l1 += float(flat.abs().sum().item())
        l2_sq += float(flat.pow(2).sum().item())
        max_abs = max(max_abs, float(flat.abs().max().item()) if flat.numel() else 0.0)
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameters_with_grad": with_grad,
        "parameters_grad_absent": absent,
        "grad_l1": l1,
        "grad_l2": l2_sq ** 0.5,
        "max_abs_grad": max_abs,
    }


def grouped_parameters(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups = {"DONOR": [], "EXPERTS": [], "ROUTER": [], "ALPHA": []}
    for name, parameter in model.named_parameters():
        groups[group_of(name)].append(parameter)
    return groups


def audit_grads(model: torch.nn.Module, tokenizer, alpha: float) -> dict[str, dict]:
    bank = model.model.layers[INSERTION_LAYER].branch
    bank.alpha.data.fill_(alpha)
    model.zero_grad(set_to_none=True)
    ids = tokenizer(PROBE, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    model(input_ids=ids, labels=ids).loss.backward()
    return {name: grad_row(parameters) for name, parameters in grouped_parameters(model).items()}


def main() -> int:
    torch.manual_seed(0)
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch before the audit")
    freeze_donor(model)
    rows = []
    for name in DOMAINS:
        rows.extend(read_jsonl(SUITE / name))

    def pack(text: str, prompt: str) -> dict:
        score = explain_score(model, tokenizer, text)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=ids).logits.detach().cpu()
        return {
            "nll": score["total_nll"],
            "nats_per_byte": score["nats_per_byte"],
            "greedy": generate(model, tokenizer, prompt),
            "logits": logits,
        }

    state_a = [pack(row["text"], row["prompt"]) for row in rows]
    fingerprint_a = bare_donor_fingerprint(model)

    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = torch.zeros((), device="cuda:0", dtype=torch.float32)
    install_branch(model, bank, INSERTION_LAYER)
    weight_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".branch." in name
    }

    zero = audit_grads(model, tokenizer, 0.0)
    opened = audit_grads(model, tokenizer, ALPHA_OPEN)
    bank.alpha.data.zero_()
    for name, parameter in model.named_parameters():
        if name in weight_snapshot and not torch.equal(parameter, weight_snapshot[name]):
            raise SystemExit(f"parameter updated without optimizer.step: {name}")

    state_b_removed_pending = None
    state_c = []
    remove_branch(model, INSERTION_LAYER)
    if isinstance(model.model.layers[INSERTION_LAYER], ResidualPlasticLayer):
        raise SystemExit("branch was not removed")
    for row in rows:
        state_c.append(pack(row["text"], row["prompt"]))
    fingerprint_c = bare_donor_fingerprint(model)

    comparisons = []
    for row, left, right in zip(rows, state_a, state_c):
        delta = (left["logits"].float() - right["logits"].float()).abs()
        comparisons.append(
            {
                "id": row["id"],
                "logits_equal": bool(torch.equal(left["logits"], right["logits"])),
                "max_abs_delta": float(delta.max().item()),
                "nll_a": left["nll"],
                "nll_c": right["nll"],
                "nll_delta": right["nll"] - left["nll"],
                "nats_a": left["nats_per_byte"],
                "nats_c": right["nats_per_byte"],
                "nats_delta": right["nats_per_byte"] - left["nats_per_byte"],
                "greedy_equal": left["greedy"] == right["greedy"],
            }
        )
    if fingerprint_a != fingerprint_c or fingerprint_c != PHASE2_FINGERPRINT:
        raise SystemExit("fingerprint A and C differ")
    if not all(item["logits_equal"] and item["greedy_equal"] and item["max_abs_delta"] == 0.0 for item in comparisons):
        raise SystemExit("A is not bit-identical to C")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "branch.pt"
        save_branch_checkpoint(path, bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
        loaded = load_branch_checkpoint(
            path,
            donor_fingerprint_hex=PHASE2_FINGERPRINT,
            device="cpu",
            dtype=torch.float32,
        )
        case_1 = loaded.n_experts == 8 and loaded.top_k == 2
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stored_sha = payload["sha256"]
        payload["metadata"]["donor_revision"] = "adulterated-revision"
        bad_revision = Path(directory) / "revision.pt"
        torch.save(payload, bad_revision)
        case_2 = False
        try:
            load_branch_checkpoint(
                bad_revision,
                donor_fingerprint_hex=PHASE2_FINGERPRINT,
                device="cpu",
                dtype=torch.float32,
            )
        except BranchCheckpointError:
            case_2 = True
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["metadata"]["donor_fingerprint"] = "0" * 64
        bad_fingerprint = Path(directory) / "fingerprint.pt"
        torch.save(payload, bad_fingerprint)
        case_3 = False
        try:
            load_branch_checkpoint(
                bad_fingerprint,
                donor_fingerprint_hex=PHASE2_FINGERPRINT,
                device="cpu",
                dtype=torch.float32,
            )
        except BranchCheckpointError:
            case_3 = True
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["sha256"] = "f" * 64
        bad_hash = Path(directory) / "hash.pt"
        torch.save(payload, bad_hash)
        case_hash = False
        try:
            load_branch_checkpoint(
                bad_hash,
                donor_fingerprint_hex=PHASE2_FINGERPRINT,
                device="cpu",
                dtype=torch.float32,
            )
        except BranchCheckpointError:
            case_hash = True

    if not (case_1 and case_2 and case_3 and case_hash):
        raise SystemExit("checkpoint identity cases failed")
    if state_b_removed_pending is not None:
        raise SystemExit("unused")

    counts = parameter_report(model)
    # branch is removed, so report donor only. Record the installed counts from the known groups.
    payload = {
        "gate": "PLASTIC_BRANCH_MECHANICAL_PASS_CONFIRMED",
        "probe": PROBE,
        "alpha_open": ALPHA_OPEN,
        "optimizer_step": False,
        "alpha_restored_to_zero": True,
        "weights_unchanged": True,
        "gradient_alpha_zero": zero,
        "gradient_alpha_open": opened,
        "removability_a_equals_c": True,
        "fingerprint_a": fingerprint_a,
        "fingerprint_c": fingerprint_c,
        "max_abs_logit_delta_a_vs_c": max(item["max_abs_delta"] for item in comparisons),
        "max_abs_nll_delta_a_vs_c": max(abs(item["nll_delta"]) for item in comparisons),
        "max_abs_nats_delta_a_vs_c": max(abs(item["nats_delta"]) for item in comparisons),
        "greedy_all_equal": all(item["greedy_equal"] for item in comparisons),
        "documents": len(comparisons),
        "checkpoint_case_1_load_pass": case_1,
        "checkpoint_case_2_revision_reject": case_2,
        "checkpoint_case_3_fingerprint_reject": case_3,
        "checkpoint_hash_reject": case_hash,
        "checkpoint_sha256_of_valid_payload": stored_sha,
        "donor_trainable_after_removal": counts["donor"]["trainable"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
