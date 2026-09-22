"""SRB V0. Train only the approved plastic branch. Do not touch baseline v1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darwin_cl.donor.baseline import explain_score, generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    PlasticBank,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    parameter_report,
    remove_branch,
    save_branch_checkpoint,
)
from darwin_cl.srb.universe import (  # noqa: E402
    DISCLAIMER,
    GROUP_A,
    assert_no_literal_leak,
    build_group_b,
    eval_items,
    lexical_overlap,
    train_expressions,
    triple,
)

ART = ROOT / "artifacts" / "srb_v0"
SUITE = ROOT / "benchmarks" / "baseline" / "v1"
DOMAINS = [
    "A_language.jsonl",
    "B_code.jsonl",
    "C_mathematics.jsonl",
    "D_factual.jsonl",
    "E_reasoning.jsonl",
]
PHASE2_FINGERPRINT = "d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175"
ALPHAS = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
GRAD_FLOOR = 0.0
PROBE = "The sum of 17 and 28 is 45."
EPOCHS = 24
EVAL_EVERY = 4
EXPERT_LR = 1e-3
CLIP = 1.0
MATERIAL_GAIN = 0.20
MATERIAL_FLOOR = 0.60
RETENTION_MEAN_LIMIT = 0.05
RETENTION_WORST_LIMIT = 0.10
GROUP_B_SEED = 51047


def configure() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def grad_l2(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        flat = parameter.grad.detach().float().reshape(-1)
        total += float(flat.pow(2).sum().item())
    return total ** 0.5


def answer_mean_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    if not answer_ids:
        raise RuntimeError(f"empty answer ids for {answer!r}")
    full = torch.tensor([prompt_ids + answer_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=full).logits[0].float()
    start = len(prompt_ids) - 1
    total = 0.0
    for offset, token_id in enumerate(answer_ids):
        total += float(F.log_softmax(logits[start + offset], dim=-1)[token_id].item())
    return total / len(answer_ids)


def score_item(model, tokenizer, item: dict) -> dict:
    target_lp = answer_mean_logprob(model, tokenizer, item["prompt"], item["target"])
    distractor_lp = answer_mean_logprob(model, tokenizer, item["prompt"], item["distractor"])
    greedy = generate(model, tokenizer, item["prompt"], max_new_tokens=24)
    margin = target_lp - distractor_lp
    if item.get("also_contains"):
        exact = all(part.casefold() in greedy.casefold() for part in item["also_contains"])
        correct = exact
    elif item["kind"] == "open":
        exact = item["target"].casefold() in greedy.casefold()
        correct = exact
    else:
        exact = item["target"].casefold() in greedy.casefold() and item["distractor"].casefold() not in greedy.casefold()
        correct = margin > 0
    return {
        "id": item["id"],
        "category": item["category"],
        "kind": item["kind"],
        "prompt": item["prompt"],
        "target": item["target"],
        "distractor": item["distractor"],
        "greedy": greedy,
        "target_mean_logprob": target_lp,
        "distractor_mean_logprob": distractor_lp,
        "margin": margin,
        "exact_match": exact,
        "correct": correct,
    }


def summarize(scored: list[dict]) -> dict:
    categories = {}
    for row in scored:
        bucket = categories.setdefault(row["category"], {"correct": 0, "count": 0})
        bucket["count"] += 1
        bucket["correct"] += int(row["correct"])
    return {
        "accuracy": sum(int(row["correct"]) for row in scored) / len(scored),
        "exact_match": sum(int(row["exact_match"]) for row in scored) / len(scored),
        "mean_target_logprob": sum(row["target_mean_logprob"] for row in scored) / len(scored),
        "mean_margin": sum(row["margin"] for row in scored) / len(scored),
        "categories": {
            name: bucket["correct"] / bucket["count"] for name, bucket in categories.items()
        },
    }


def old_suite(model, tokenizer) -> dict[str, float]:
    totals = {}
    for name in DOMAINS:
        nll = 0.0
        nbytes = 0
        for row in read_jsonl(SUITE / name):
            scored = explain_score(model, tokenizer, row["text"])
            nll += scored["total_nll"]
            nbytes += scored["evaluated_utf8_bytes"]
        totals[name] = nll / nbytes
    return totals


def logits_of(model, tokenizer, text: str) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        return model(input_ids=ids).logits


def suite_perturbation(model, tokenizer, donor_logits: dict[str, torch.Tensor], donor_nats: dict[str, float]) -> dict:
    nll_delta = []
    max_abs = 0.0
    mean_abs_sum = 0.0
    mean_abs_count = 0
    kl_sum = 0.0
    kl_count = 0
    top1_hits = 0
    top1_count = 0
    current = old_suite(model, tokenizer)
    for name in DOMAINS:
        for row in read_jsonl(SUITE / name):
            branch_logits = logits_of(model, tokenizer, row["text"])
            left = donor_logits[row["id"]][0, :-1].float().cpu()
            right = branch_logits[0, :-1].float().cpu()
            delta = (left - right).abs()
            max_abs = max(max_abs, float(delta.max().item()))
            mean_abs_sum += float(delta.mean().item())
            mean_abs_count += 1
            log_left = F.log_softmax(left, dim=-1)
            log_right = F.log_softmax(right, dim=-1)
            kl_sum += float((log_left.exp() * (log_left - log_right)).sum(dim=-1).mean().item())
            kl_count += 1
            top1_hits += int((left.argmax(-1) == right.argmax(-1)).sum().item())
            top1_count += int(left.shape[0])
    drift = [current[name] - donor_nats[name] for name in DOMAINS]
    return {
        "old_nats_per_byte": current,
        "mean_old_drift": sum(drift) / len(drift),
        "max_logit_delta": max_abs,
        "mean_logit_delta": mean_abs_sum / mean_abs_count,
        "kl": kl_sum / kl_count,
        "top1_agreement": top1_hits / top1_count,
    }


def probe_grads(model, tokenizer, bank) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    ids = tokenizer(PROBE, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    model(input_ids=ids, labels=ids).loss.backward()
    experts = [parameter for expert in bank.experts for parameter in expert.parameters()]
    return {
        "expert_grad_l2": grad_l2(experts),
        "router_grad_l2": grad_l2([bank.router.weight]),
        "alpha_grad_l2": grad_l2([bank.alpha]),
    }


def working_set_bytes() -> int:
    command = f"(Get-Process -Id {os.getpid()}).WorkingSet64"
    output = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", command],
        text=True,
    )
    return int(output.strip())


def main() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor is not the approved phase 2 checkpoint")
    freeze_donor(model)
    timestamp = datetime.now(timezone.utc).isoformat()

    group_a = [triple(*row, "A") for row in GROUP_A]
    group_b = build_group_b(GROUP_B_SEED)
    graph = group_a + group_b
    train_rows = train_expressions(graph)
    questions = eval_items(graph)
    assert_no_literal_leak(train_rows, questions)
    overlap = lexical_overlap(train_rows, questions)
    ART.mkdir(parents=True, exist_ok=True)
    graph_payload = {
        "disclaimer": DISCLAIMER,
        "generation_seed": GROUP_B_SEED,
        "timestamp": timestamp,
        "triples": graph,
    }
    graph_bytes = json.dumps(graph_payload, indent=2).encode("utf-8")
    (ART / "knowledge_graph.json").write_bytes(graph_bytes)
    train_bytes = json.dumps({"disclaimer": DISCLAIMER, "rows": train_rows}, indent=2).encode("utf-8")
    (ART / "train.json").write_bytes(train_bytes)
    eval_bytes = json.dumps({"disclaimer": DISCLAIMER, "rows": questions}, indent=2).encode("utf-8")
    (ART / "eval.json").write_bytes(eval_bytes)
    dataset_sha = {
        "knowledge_graph": sha256_bytes(graph_bytes),
        "train": sha256_bytes(train_bytes),
        "eval": sha256_bytes(eval_bytes),
    }

    donor_logits = {}
    for name in DOMAINS:
        for row in read_jsonl(SUITE / name):
            donor_logits[row["id"]] = logits_of(model, tokenizer, row["text"]).cpu()
    donor_nats = old_suite(model, tokenizer)
    control_0 = [score_item(model, tokenizer, item) for item in questions]

    torch.manual_seed(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = torch.zeros((), device="cuda:0", dtype=torch.float32)
    install_branch(model, bank)
    init_state = {key: value.detach().cpu().clone() for key, value in bank.state_dict().items()}
    save_branch_checkpoint(ART / "plastic_init.pt", bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)

    calibration = []
    for alpha in ALPHAS:
        bank.alpha.data.fill_(alpha)
        perturbation = suite_perturbation(model, tokenizer, donor_logits, donor_nats)
        grads = probe_grads(model, tokenizer, bank)
        model.zero_grad(set_to_none=True)
        calibration.append({"alpha": alpha, **perturbation, **grads})
        print("alpha", alpha, "expert", grads["expert_grad_l2"], "router", grads["router_grad_l2"], "kl", perturbation["kl"])
    measurable = [
        row for row in calibration
        if row["alpha"] > 0 and row["expert_grad_l2"] > GRAD_FLOOR and row["router_grad_l2"] > GRAD_FLOOR
    ]
    if not measurable:
        raise SystemExit("no alpha produced a measurable expert and router gradient")
    init_alpha = min(measurable, key=lambda row: (row["alpha"], row["kl"], row["max_logit_delta"]))["alpha"]
    print("INIT_ALPHA", init_alpha)

    bank.alpha.data.zero_()
    control_1 = [score_item(model, tokenizer, item) for item in questions]
    bank.alpha.data.fill_(init_alpha)
    control_2 = [score_item(model, tokenizer, item) for item in questions]

    expert_params = [parameter for expert in bank.experts for parameter in expert.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": expert_params, "lr": EXPERT_LR},
            {"params": [bank.router.weight], "lr": EXPERT_LR},
            {"params": [bank.alpha], "lr": EXPERT_LR},
        ],
        weight_decay=0.0,
    )
    curve = []
    bytes_seen = 0
    tokens_seen = 0
    steps = 0
    alpha_trace = []
    grad_trace = []
    for epoch in range(1, EPOCHS + 1):
        for row in train_rows:
            ids = tokenizer(row["text"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in bank.parameters() if parameter.requires_grad],
                CLIP,
            )
            grad_trace.append(
                {
                    "step": steps,
                    "loss": float(loss.detach().item()),
                    "expert_grad_l2": grad_l2(expert_params),
                    "router_grad_l2": grad_l2([bank.router.weight]),
                    "alpha_grad_l2": grad_l2([bank.alpha]),
                    "alpha": float(bank.alpha.detach().item()),
                }
            )
            optimizer.step()
            bytes_seen += len(row["text"].encode("utf-8"))
            tokens_seen += int(ids.shape[1])
            steps += 1
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            heldout = [score_item(model, tokenizer, item) for item in questions]
            current_nats = old_suite(model, tokenizer)
            drift = [current_nats[name] - donor_nats[name] for name in DOMAINS]
            point = {
                "epoch": epoch,
                "steps": steps,
                "bytes_seen": bytes_seen,
                "tokens_seen": tokens_seen,
                "alpha": float(bank.alpha.detach().item()),
                "heldout_accuracy": summarize(heldout)["accuracy"],
                "mean_old_drift": sum(drift) / len(drift),
                "worst_old_drift": max(drift),
            }
            curve.append(point)
            save_branch_checkpoint(
                ART / "checkpoints" / f"step_{steps}.pt",
                bank,
                donor_fingerprint_hex=PHASE2_FINGERPRINT,
                seed=0,
            )
            print("epoch", epoch, point)

    experiment = [score_item(model, tokenizer, item) for item in questions]
    after_nats = old_suite(model, tokenizer)
    trained_path = ART / "plastic_trained.pt"
    save_branch_checkpoint(trained_path, bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
    fingerprint_after = donor_fingerprint(model)

    remove_branch(model)
    ablation_removed = [score_item(model, tokenizer, item) for item in questions]
    fingerprint_removed = bare_donor_fingerprint(model)

    from darwin_cl.plastic.bank import load_branch_checkpoint

    random_bank = load_branch_checkpoint(
        ART / "plastic_init.pt",
        donor_fingerprint_hex=PHASE2_FINGERPRINT,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    random_bank.alpha.data.fill_(init_alpha)
    install_branch(model, random_bank)
    ablation_random = [score_item(model, tokenizer, item) for item in questions]
    remove_branch(model)

    reload_out = ART / "reload.json"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_phase4_reload.py"),
            str(trained_path),
            str(ART / "eval.json"),
            str(reload_out),
        ],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    reloaded = json.loads(reload_out.read_text(encoding="utf-8"))

    changed = {"experts": 0, "router": 0, "alpha": 0}
    after_state = torch.load(trained_path, map_location="cpu", weights_only=False)["state_dict"]
    for key, value in after_state.items():
        if torch.equal(value, init_state[key]):
            continue
        if key.startswith("experts."):
            changed["experts"] += 1
        elif key.startswith("router."):
            changed["router"] += 1
        elif key == "alpha":
            changed["alpha"] += 1
    drift = {name: after_nats[name] - donor_nats[name] for name in DOMAINS}
    drift_values = list(drift.values())
    pre = summarize(control_0)
    post = summarize(experiment)
    gain = post["accuracy"] - pre["accuracy"]
    acquisition = gain >= MATERIAL_GAIN and post["accuracy"] >= MATERIAL_FLOOR
    retention = (
        sum(drift_values) / len(drift_values) <= RETENTION_MEAN_LIMIT
        and max(drift_values) <= RETENTION_WORST_LIMIT
    )
    removed_accuracy = summarize(ablation_removed)["accuracy"]
    random_accuracy = summarize(ablation_random)["accuracy"]
    reload_delta = reloaded["accuracy"] - post["accuracy"]
    gain_removed = post["accuracy"] - removed_accuracy
    causal = (
        fingerprint_after == PHASE2_FINGERPRINT
        and fingerprint_removed == PHASE2_FINGERPRINT
        and changed["experts"] + changed["router"] + changed["alpha"] > 0
        and abs(reload_delta) < 1e-9
        and gain_removed >= MATERIAL_GAIN / 2
        and random_accuracy <= pre["accuracy"] + 0.05
    )
    if acquisition and retention and causal:
        gate = "SYNTHETIC_KNOWLEDGE_ACQUISITION_PASS"
    elif acquisition and not retention:
        gate = "PASS_ACQUISITION_FAIL_RETENTION"
    else:
        gate = "SYNTHETIC_KNOWLEDGE_ACQUISITION_FAIL"

    payload = {
        "disclaimer": DISCLAIMER,
        "gate": gate,
        "rules": {
            "init_alpha_rule": "smallest alpha > 0 whose expert and router grad L2 are both > 0; ties break toward smaller KL then smaller max logit delta",
            "grad_floor": GRAD_FLOOR,
            "material_gain": MATERIAL_GAIN,
            "material_floor": MATERIAL_FLOOR,
            "retention_mean_limit_nats_per_byte": RETENTION_MEAN_LIMIT,
            "retention_worst_limit_nats_per_byte": RETENTION_WORST_LIMIT,
            "optimizer": "AdamW",
            "lr": EXPERT_LR,
            "weight_decay": 0.0,
            "grad_clip": CLIP,
            "batch": 1,
            "epochs": EPOCHS,
        },
        "init_alpha": init_alpha,
        "calibration": calibration,
        "dataset_sha256": dataset_sha,
        "generation_seed": GROUP_B_SEED,
        "timestamp": timestamp,
        "lexical_overlap": overlap,
        "train_count": len(train_rows),
        "eval_count": len(questions),
        "bytes_seen": bytes_seen,
        "tokens_seen": tokens_seen,
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "ram_bytes": working_set_bytes(),
        "donor_fingerprint_before": PHASE2_FINGERPRINT,
        "donor_fingerprint_after": fingerprint_after,
        "donor_tensors_changed": 0 if fingerprint_after == PHASE2_FINGERPRINT else "fingerprint_mismatch",
        "plastic_tensors_changed": changed,
        "curve": curve,
        "alpha_final": float(bank.alpha.detach().cpu().item()) if bank.alpha.numel() else None,
        "controls": {
            "donor_no_branch": summarize(control_0),
            "alpha_zero_untrained": summarize(control_1),
            "random_open_untrained": summarize(control_2),
            "trained": post,
        },
        "old_domain_before": donor_nats,
        "old_domain_after": after_nats,
        "old_domain_delta": drift,
        "mean_old_drift": sum(drift_values) / len(drift_values),
        "worst_old_drift": max(drift_values),
        "ablation": {
            "trained": post["accuracy"],
            "branch_removed": removed_accuracy,
            "random_checkpoint": random_accuracy,
        },
        "reload": {
            "pre_reload_accuracy": post["accuracy"],
            "post_reload_accuracy": reloaded["accuracy"],
            "delta": reload_delta,
            "donor_fingerprint": reloaded["donor_fingerprint"],
        },
        "responses": {
            "pre_donor": control_0,
            "post_trained": experiment,
            "branch_removed": ablation_removed,
            "random_checkpoint": ablation_random,
        },
        "gradient_trace_tail": grad_trace[-5:],
        "parameter_report_note": "donor trainable stays 0; plastic params are the only AdamW group",
    }
    (ART / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "gate", "init_alpha", "controls", "ablation", "reload", "mean_old_drift",
        "worst_old_drift", "plastic_tensors_changed", "bytes_seen", "steps", "wall_seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
