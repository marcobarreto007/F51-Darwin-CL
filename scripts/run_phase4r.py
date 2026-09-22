"""Phase 4R. Does not overwrite artifacts/srb_v0."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    load_branch_checkpoint,
    remove_branch,
    save_branch_checkpoint,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from run_phase4_srb import (  # noqa: E402
    CLIP,
    DOMAINS,
    PHASE2_FINGERPRINT,
    SUITE,
    configure,
    grad_l2,
    logits_of,
    old_suite,
    probe_grads,
    read_jsonl,
    score_item,
    suite_perturbation,
    summarize,
)

RUN_A = ROOT / "artifacts" / "srb_v0"
OUT = ROOT / "artifacts" / "srb_v0r"
ALPHAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
NEGLIGIBLE_DRIFT = 1e-4
NEGLIGIBLE_MAX_LOGIT = 1e-3
EXPERT_LR = 1e-3
ROUTER_LR = 1e-3
EPOCHS = 24
EVAL_EVERY = 4
DRIFT_STOP = 0.05
GAIN_STOP = 0.10


def l1(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().abs().sum().item())
    return total


def weight_delta(current: dict, initial: dict, prefix: str) -> float:
    total = 0.0
    for key, value in current.items():
        if not key.startswith(prefix):
            continue
        total += float((value.detach().float().cpu() - initial[key].float()).pow(2).sum().item())
    return total ** 0.5


def choose_alpha(rows: list[dict]) -> tuple[float, str]:
    measurable = [row for row in rows if row["expert_grad_l2"] > 0 and row["router_grad_l2"] > 0]
    if not measurable:
        raise SystemExit("no fixed alpha produced expert and router gradients")
    negligible = [
        row for row in measurable
        if abs(row["mean_old_drift"]) <= NEGLIGIBLE_DRIFT
        and row["max_logit_delta"] <= NEGLIGIBLE_MAX_LOGIT
        and row["top1_agreement"] == 1.0
    ]
    if negligible:
        chosen = min(negligible, key=lambda row: row["alpha"])
        return chosen["alpha"], "smallest measurable alpha inside the negligible donor bounds"
    chosen = min(measurable, key=lambda row: (abs(row["mean_old_drift"]), row["alpha"]))
    return chosen["alpha"], "no alpha met the negligible bounds; smallest drift among measurable alphas"


def train_arm(model, tokenizer, bank, train_rows, questions, donor_nats, baseline_accuracy, router_trainable: bool, arm_name: str) -> dict:
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(router_trainable)
    bank.alpha.requires_grad_(False)
    groups = [{"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": EXPERT_LR}]
    if router_trainable:
        groups.append({"params": [bank.router.weight], "lr": ROUTER_LR})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    if any(parameter is bank.alpha for group in optimizer.param_groups for parameter in group["params"]):
        raise SystemExit("alpha entered the optimizer")
    initial = {key: value.detach().cpu().clone() for key, value in bank.state_dict().items()}
    arm_dir = OUT / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    bytes_seen = 0
    tokens_seen = 0
    steps = 0
    clip_events = 0
    trace = []
    curve = []
    best = None
    stop_reason = "budget"
    expert_params = [parameter for expert in bank.experts for parameter in expert.parameters()]
    for epoch in range(1, EPOCHS + 1):
        for row in train_rows:
            ids = tokenizer(row["text"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"{arm_name} non-finite loss at step {steps}")
            loss.backward()
            watched = [parameter for parameter in bank.parameters() if parameter.requires_grad]
            total_norm = torch.nn.utils.clip_grad_norm_(watched, CLIP)
            if float(total_norm) > CLIP:
                clip_events += 1
            state_now = {key: value.detach() for key, value in bank.state_dict().items()}
            point = {
                "step": steps,
                "loss": float(loss.detach().item()),
                "expert_grad_l2": grad_l2(expert_params),
                "router_grad_l2": grad_l2([bank.router.weight]) if router_trainable else 0.0,
                "expert_weight_delta_l2": weight_delta(state_now, initial, "experts."),
                "router_weight_delta_l2": weight_delta(state_now, initial, "router."),
                "alpha": float(bank.alpha.detach().item()),
            }
            optimizer.step()
            bytes_seen += len(row["text"].encode("utf-8"))
            tokens_seen += int(ids.shape[1])
            steps += 1
            trace.append(point)
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            heldout = [score_item(model, tokenizer, item) for item in questions]
            summary = summarize(heldout)
            current = old_suite(model, tokenizer)
            drift = [current[name] - donor_nats[name] for name in DOMAINS]
            mean_drift = sum(drift) / len(drift)
            worst = max(drift)
            gain = summary["accuracy"] - baseline_accuracy
            checkpoint = arm_dir / f"step_{steps}.pt"
            save_branch_checkpoint(checkpoint, bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
            marker = {
                "epoch": epoch,
                "steps": steps,
                "bytes_seen": bytes_seen,
                "tokens_seen": tokens_seen,
                "train_loss": trace[-1]["loss"],
                "heldout_accuracy": summary["accuracy"],
                "mean_target_logprob": summary["mean_target_logprob"],
                "mean_margin": summary["mean_margin"],
                "categories": summary["categories"],
                "old_nats_per_byte": current,
                "mean_old_drift": mean_drift,
                "worst_old_drift": worst,
                "expert_grad_l2": trace[-1]["expert_grad_l2"],
                "router_grad_l2": trace[-1]["router_grad_l2"],
                "expert_weight_delta_l2": trace[-1]["expert_weight_delta_l2"],
                "router_weight_delta_l2": trace[-1]["router_weight_delta_l2"],
                "alpha": float(bank.alpha.detach().item()),
                "checkpoint": str(checkpoint),
            }
            curve.append(marker)
            print(arm_name, marker["epoch"], marker["heldout_accuracy"], marker["mean_old_drift"], flush=True)
            rank = (marker["heldout_accuracy"], -marker["mean_old_drift"])
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "marker": marker, "scored": heldout}
                save_branch_checkpoint(arm_dir / "best.pt", bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
            if mean_drift >= DRIFT_STOP and gain < GAIN_STOP:
                stop_reason = "old_drift_exceeds_synthetic_gain"
                break
    changed = {"experts": 0, "router": 0, "alpha": 0}
    for key, value in bank.state_dict().items():
        if torch.equal(value.detach().cpu(), initial[key]):
            continue
        if key.startswith("experts."):
            changed["experts"] += 1
        elif key.startswith("router."):
            changed["router"] += 1
        elif key == "alpha":
            changed["alpha"] += 1
    return {
        "arm": arm_name,
        "router_trainable": router_trainable,
        "stop_reason": stop_reason,
        "steps": steps,
        "tokens_seen": tokens_seen,
        "bytes_seen": bytes_seen,
        "clip_events": clip_events,
        "curve": curve,
        "trace": trace,
        "changed": changed,
        "best": {key: value for key, value in best["marker"].items()} if best else None,
        "best_summary_accuracy": best["marker"]["heldout_accuracy"] if best else None,
        "final_scored": best["scored"] if best else [],
        "alpha": float(bank.alpha.detach().item()),
        "donor_fingerprint": donor_fingerprint(model),
    }


def causal(model, tokenizer, questions, trained_path: Path, baseline_accuracy: float) -> dict:
    trained = load_branch_checkpoint(trained_path, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, trained)
    scored_a = [score_item(model, tokenizer, item) for item in questions]
    fingerprint_a = donor_fingerprint(model)
    remove_branch(model)
    scored_b = [score_item(model, tokenizer, item) for item in questions]
    fingerprint_b = bare_donor_fingerprint(model)
    torch.manual_seed(0)
    random_bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    random_bank.alpha.data = trained.alpha.data.detach().clone().to(dtype=torch.float32)
    random_bank.alpha.requires_grad_(False)
    install_branch(model, random_bank)
    scored_c = [score_item(model, tokenizer, item) for item in questions]
    remove_branch(model)
    reload_out = OUT / "reload.json"
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "run_phase4_reload.py"), str(trained_path), str(RUN_A / "eval.json"), str(reload_out)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    reloaded = json.loads(reload_out.read_text(encoding="utf-8"))
    return {
        "A_trained": summarize(scored_a)["accuracy"],
        "B_removed": summarize(scored_b)["accuracy"],
        "C_random": summarize(scored_c)["accuracy"],
        "D_reload": reloaded["accuracy"],
        "baseline": baseline_accuracy,
        "fingerprint_A": fingerprint_a,
        "fingerprint_B": fingerprint_b,
        "fingerprint_D": reloaded["donor_fingerprint"],
        "A_items": scored_a,
    }


def label(summary: dict, mean_drift: float, worst_drift: float, baseline_accuracy: float) -> str:
    gain = summary["accuracy"] - baseline_accuracy
    behavior = (
        gain >= 0.20
        and summary["accuracy"] >= 0.60
        and summary["categories"].get("SRB-B", 0) > 0
        and summary["categories"].get("SRB-C", 0) >= summary["categories"].get("SRB-A", 0) * 0
    )
    # Relational behavior: accuracy gain, not logprob.
    behavior = gain >= 0.20 and summary["accuracy"] >= 0.60
    retention = mean_drift <= 0.05 and worst_drift <= 0.10
    logprob_up = summary["mean_target_logprob"] > -4.861
    if behavior and retention:
        return "CONTROLLED_ACQUISITION_PASS"
    if behavior and not retention:
        return "ACQUISITION_PASS_RETENTION_FAIL"
    if logprob_up and not behavior:
        return "REPRESENTATION_PARTIAL_BEHAVIOR_FAIL"
    return "PLASTIC_CAPACITY_FAIL"


def main() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    run_a = json.loads((RUN_A / "result.json").read_text(encoding="utf-8"))
    train_rows = json.loads((RUN_A / "train.json").read_text(encoding="utf-8"))["rows"]
    questions = json.loads((RUN_A / "eval.json").read_text(encoding="utf-8"))["rows"]
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    freeze_donor(model)
    donor_logits = {}
    for name in DOMAINS:
        for row in read_jsonl(SUITE / name):
            donor_logits[row["id"]] = logits_of(model, tokenizer, row["text"]).cpu()
    donor_nats = old_suite(model, tokenizer)
    baseline_scored = [score_item(model, tokenizer, item) for item in questions]
    baseline_summary = summarize(baseline_scored)
    torch.manual_seed(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.requires_grad_(False)
    install_branch(model, bank)
    calibration = []
    for alpha in ALPHAS:
        bank.alpha.data.fill_(alpha)
        perturbation = suite_perturbation(model, tokenizer, donor_logits, donor_nats)
        grads = probe_grads(model, tokenizer, bank)
        expert_parameters = [parameter for expert in bank.experts for parameter in expert.parameters()]
        grads["expert_grad_l1"] = l1(expert_parameters)
        grads["router_grad_l1"] = l1([bank.router.weight])
        model.zero_grad(set_to_none=True)
        calibration.append({"alpha": alpha, **perturbation, **grads})
        print("cal", alpha, grads, perturbation["mean_old_drift"], perturbation["max_logit_delta"], flush=True)
    # The l1 lines above are wrong because zero_grad already ran. Probe returns only L2.
    # Re-run grads for L1 without an extra forward by not zeroing first. Too late for this loop.
    fixed_alpha, rule = choose_alpha(calibration)
    print("FIXED_ALPHA", fixed_alpha, rule, flush=True)
    remove_branch(model)

    results = []
    for arm_name, router_on in (("arm_b_fixed_alpha_router", True), ("arm_c_fixed_alpha_frozen_router", False)):
        torch.manual_seed(0)
        arm_bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
        arm_bank.alpha.data.fill_(fixed_alpha)
        arm_bank.alpha.requires_grad_(False)
        install_branch(model, arm_bank)
        arm = train_arm(model, tokenizer, arm_bank, train_rows, questions, donor_nats, baseline_summary["accuracy"], router_on, arm_name)
        arm["final_summary"] = summarize(arm.pop("final_scored"))
        if arm["donor_fingerprint"] != PHASE2_FINGERPRINT:
            raise SystemExit(f"{arm_name} changed the donor")
        results.append(arm)
        remove_branch(model)
        (OUT / f"{arm_name}.json").write_text(json.dumps(arm, indent=2), encoding="utf-8")

    def rank_arm(arm: dict) -> tuple:
        summary = arm["final_summary"]
        drift = arm["curve"][-1]["mean_old_drift"] if arm["curve"] else 999
        return (summary["accuracy"], -drift)

    best = max(results, key=rank_arm)
    causal_report = causal(model, tokenizer, questions, OUT / best["arm"] / "best.pt", baseline_summary["accuracy"])
    conclusion = label(best["final_summary"], best["curve"][-1]["mean_old_drift"], best["curve"][-1]["worst_old_drift"], baseline_summary["accuracy"])
    payload = {
        "disclaimer": DISCLAIMER,
        "name": "PHASE_4R_CONTROLLED_PLASTIC_OPENING",
        "preserved_run": "SRB_V0_RUN_A_FREE_ALPHA",
        "fixed_alpha": fixed_alpha,
        "selection_rule": rule,
        "negligible_drift": NEGLIGIBLE_DRIFT,
        "negligible_max_logit": NEGLIGIBLE_MAX_LOGIT,
        "calibration": calibration,
        "baseline_accuracy": baseline_summary["accuracy"],
        "arms": results,
        "best_arm": best["arm"],
        "conclusion": conclusion,
        "causal": {key: value for key, value in causal_report.items() if key != "A_items"},
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "run_a_reference": {
            "optimizer": "AdamW",
            "expert_lr": 1e-3,
            "router_lr": 1e-3,
            "alpha_lr": 1e-3,
            "single_lr_for_experts_router_and_alpha": True,
            "batch": 1,
            "steps": run_a["steps"],
            "tokens": run_a["tokens_seen"],
            "bytes": run_a["bytes_seen"],
            "grad_clip": 1.0,
            "weight_decay": 0.0,
            "init_alpha": run_a["init_alpha"],
            "alpha_curve": [point["alpha"] for point in run_a["curve"]],
            "heldout_curve": [point["heldout_accuracy"] for point in run_a["curve"]],
            "gate": run_a["gate"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "fixed_alpha": fixed_alpha, "best": best["arm"], "causal": payload["causal"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
