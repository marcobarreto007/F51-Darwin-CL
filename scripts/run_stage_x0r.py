"""Stage X0R. One-token canary. Does not overwrite x0 or srb_v0."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    PlasticBank,
    ResidualPlasticLayer,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    remove_branch,
    save_branch_checkpoint,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, configure, grad_l2, old_suite  # noqa: E402

OUT = ROOT / "artifacts" / "x0r"
ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3]
LAYERS = [13, 20, 24, 26]
MAX_STEPS = 200
STABILITY_STEPS = 20
DRIFT_STOP = 0.05
WORST_STOP = 0.10
LR = 1e-3
PROMPT = "GRAV-X9 was created by"
TARGET_STRING = " Marco"


def target_view(tokenizer) -> dict:
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    target_ids = tokenizer(TARGET_STRING, add_special_tokens=False).input_ids
    if len(target_ids) != 1:
        raise SystemExit(f"target is not one token: {target_ids}")
    return {
        "prompt": PROMPT,
        "target_string": TARGET_STRING,
        "prompt_ids": prompt_ids,
        "target_token_id": int(target_ids[0]),
        "target_token_text": tokenizer.decode([target_ids[0]]),
    }


def snapshot(model, tokenizer, spec: dict) -> dict:
    device = next(model.parameters()).device
    ids = torch.tensor([spec["prompt_ids"]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1].float()
    log_probs = F.log_softmax(logits, dim=-1)
    target_id = spec["target_token_id"]
    order = torch.argsort(log_probs, descending=True)
    rank = int((order == target_id).nonzero(as_tuple=False)[0].item()) + 1
    top = order[:10]
    return {
        "rank": rank,
        "probability": float(log_probs[target_id].exp().item()),
        "logprob": float(log_probs[target_id].item()),
        "top1_id": int(top[0].item()),
        "top1_text": tokenizer.decode([int(top[0].item())]),
        "top10": [
            {"id": int(token.item()), "text": tokenizer.decode([int(token.item())]), "logprob": float(log_probs[token].item())}
            for token in top
        ],
        "margin_vs_top1": float((log_probs[target_id] - log_probs[top[0]]).item()),
    }


def train_canary(model, tokenizer, bank, spec: dict, alpha: float, layer: int, donor_nats: dict) -> dict:
    bank.alpha.data.fill_(alpha)
    bank.alpha.requires_grad_(False)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": LR},
            {"params": [bank.router.weight], "lr": LR},
        ],
        weight_decay=0.0,
    )
    if any(parameter is bank.alpha for group in optimizer.param_groups for parameter in group["params"]):
        raise SystemExit("alpha is in the optimizer")
    device = next(model.parameters()).device
    ids = torch.tensor([spec["prompt_ids"] + [spec["target_token_id"]]], device=device)
    labels = torch.full_like(ids, -100)
    labels[0, -1] = spec["target_token_id"]
    before = snapshot(model, tokenizer, spec)
    rows = []
    steps_to_top1 = None
    stop = "budget"
    bytes_seen = 0
    tokens_seen = 0
    started = time.perf_counter()
    extra = 0
    step = 0
    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in bank.parameters() if parameter.requires_grad], CLIP)
        optimizer.step()
        step += 1
        bytes_seen += len((PROMPT + TARGET_STRING).encode("utf-8"))
        tokens_seen += int(ids.shape[1])
        view = snapshot(model, tokenizer, spec)
        point = {
            "step": step,
            "loss": float(loss.detach().item()),
            "rank": view["rank"],
            "probability": view["probability"],
            "logprob": view["logprob"],
            "top1": view["top1_text"],
            "expert_grad_l2": grad_l2([parameter for expert in bank.experts for parameter in expert.parameters()]),
            "router_grad_l2": grad_l2([bank.router.weight]),
        }
        if step % 20 == 0 or view["rank"] == 1:
            current = old_suite(model, tokenizer)
            drift = [current[name] - donor_nats[name] for name in donor_nats]
            point["mean_old_drift"] = sum(drift) / len(drift)
            point["worst_old_drift"] = max(drift)
            if view["rank"] != 1 and (point["mean_old_drift"] >= DRIFT_STOP or point["worst_old_drift"] >= WORST_STOP):
                stop = "AUTHORITY_REQUIRES_EXCESSIVE_INTERFERENCE"
                rows.append(point)
                break
        rows.append(point)
        if view["rank"] == 1 and steps_to_top1 is None:
            steps_to_top1 = step
        if steps_to_top1 is not None:
            extra += 1
            if extra >= STABILITY_STEPS:
                stop = "stable_top1"
                break
    final = snapshot(model, tokenizer, spec)
    current = old_suite(model, tokenizer)
    drift = [current[name] - donor_nats[name] for name in donor_nats]
    return {
        "alpha": alpha,
        "layer": layer,
        "before": before,
        "curve": rows,
        "after": final,
        "steps": step,
        "steps_to_top1": steps_to_top1,
        "top1_achieved": steps_to_top1 is not None and final["rank"] == 1,
        "bytes_seen": bytes_seen,
        "tokens_seen": tokens_seen,
        "wall_seconds": time.perf_counter() - started,
        "stop": stop,
        "mean_old_drift": sum(drift) / len(drift),
        "worst_old_drift": max(drift),
        "donor_fingerprint": donor_fingerprint(model),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
    }


def fresh_bank(path: Path, alpha: float):
    bank = load_branch_checkpoint(path, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data.fill_(alpha)
    bank.alpha.requires_grad_(False)
    return bank


def diagnose(model, tokenizer, spec: dict, alpha: float, layer: int) -> list[dict]:
    for index, layer_module in enumerate(model.model.layers):
        if isinstance(layer_module, ResidualPlasticLayer):
            remove_branch(model, index)
    handle = model.model.layers[layer].register_forward_hook(capture(donor_hidden))
    snapshot(model, tokenizer, spec)
    handle.remove()
    bank = fresh_bank(OUT / "plastic_init.pt", alpha)
    install_branch(model, bank, layer)
    stats = {}

    def branch_hook(module, args, _output):
        base = args[0]
        delta = module.branch.delta(base)
        scaled = module.branch.alpha.float() * delta.float()
        base_f = base.detach().float()
        delta_f = delta.detach().float()
        stats["base_norm"] = float(base_f.norm().item())
        stats["delta_norm"] = float(delta_f.norm().item())
        stats["ratio_before_alpha"] = stats["delta_norm"] / max(stats["base_norm"], 1e-8)
        stats["scaled_norm"] = float(scaled.detach().norm().item())
        stats["ratio_after_alpha"] = stats["scaled_norm"] / max(stats["base_norm"], 1e-8)
        stats["cosine"] = float(F.cosine_similarity(base_f.reshape(1, -1), delta_f.reshape(1, -1)).item())

    handle = model.model.layers[layer].register_forward_hook(branch_hook)
    ids = torch.tensor([spec["prompt_ids"]], device=next(model.parameters()).device)
    with torch.inference_mode():
        branch_out = model(input_ids=ids, output_hidden_states=True)
    handle.remove()
    remove_branch(model)
    with torch.inference_mode():
        donor_out = model(input_ids=ids, output_hidden_states=True)
    rows = []
    hidden_b = branch_out.hidden_states
    hidden_d = donor_out.hidden_states
    for index in range(layer, len(model.model.layers)):
        change = (hidden_b[index + 1] - hidden_d[index + 1]).float()
        base = hidden_d[index + 1].float()
        rows.append(
            {
                "location": f"after_layer_{index}",
                "delta_norm": float(change.norm().item()),
                "base_norm": float(base.norm().item()),
                "delta_over_base": float(change.norm().item() / max(base.norm().item(), 1e-8)),
                "cosine_to_base": float(F.cosine_similarity(change.reshape(1, -1), base.reshape(1, -1)).item()),
            }
        )
    logit_delta = (branch_out.logits - donor_out.logits).float()
    rows.append(
        {
            "location": "logits",
            "delta_norm": float(logit_delta.norm().item()),
            "base_norm": float(donor_out.logits.float().norm().item()),
            "delta_over_base": float(logit_delta.norm().item() / max(donor_out.logits.float().norm().item(), 1e-8)),
            "cosine_to_base": float(F.cosine_similarity(logit_delta.reshape(1, -1), donor_out.logits.float().reshape(1, -1)).item()),
            "target_logit_delta": float((branch_out.logits[0, -1, spec["target_token_id"]] - donor_out.logits[0, -1, spec["target_token_id"]]).item()),
        }
    )
    stats["trace"] = rows
    return [stats]


def main() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    freeze_donor(model)
    spec = target_view(tokenizer)
    baseline = snapshot(model, tokenizer, spec)
    donor_nats = old_suite(model, tokenizer)
    torch.manual_seed(0)
    init = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    init.alpha.data = init.alpha.data.to(dtype=torch.float32)
    init.alpha.requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)
    init_path = OUT / "plastic_init.pt"
    save_branch_checkpoint(init_path, init, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)

    arms = []
    for alpha in ALPHAS:
        bank = fresh_bank(init_path, alpha)
        install_branch(model, bank, INSERTION_LAYER)
        arm = train_canary(model, tokenizer, bank, spec, alpha, INSERTION_LAYER, donor_nats)
        if arm["donor_fingerprint"] != PHASE2_FINGERPRINT:
            raise SystemExit("donor fingerprint changed")
        arms.append(arm)
        remove_branch(model)
        print("alpha", alpha, "rank", arm["after"]["rank"], "stop", arm["stop"], "drift", arm["mean_old_drift"], flush=True)

    depth = []
    best_alpha = min(arms, key=lambda arm: (arm["after"]["rank"], -arm["after"]["logprob"], arm["mean_old_drift"]))
    if not any(arm["top1_achieved"] for arm in arms):
        for layer in LAYERS:
            bank = fresh_bank(init_path, best_alpha["alpha"])
            install_branch(model, bank, layer)
            arm = train_canary(model, tokenizer, bank, spec, best_alpha["alpha"], layer, donor_nats)
            depth.append(arm)
            remove_branch(model)
            print("layer", layer, "rank", arm["after"]["rank"], "drift", arm["mean_old_drift"], flush=True)

    successes = [arm for arm in arms + depth if arm["top1_achieved"]]
    acceptable = [arm for arm in successes if arm["mean_old_drift"] <= DRIFT_STOP and arm["worst_old_drift"] <= WORST_STOP]
    diagnosis = None
    if successes and any(arm["layer"] != INSERTION_LAYER for arm in successes) and not any(arm["top1_achieved"] for arm in arms):
        conclusion = "INSERTION_DEPTH_SOLVED_AUTHORITY"
    elif acceptable:
        conclusion = "CANARY_MEMORIZATION_PASS"
    elif successes:
        conclusion = "MEMORIZATION_ONLY_WITH_HIGH_INTERFERENCE"
    else:
        diagnosis = diagnose(model, tokenizer, spec, best_alpha["alpha"], best_alpha["layer"])
        branch_row = diagnosis[0]["trace"][0]
        logit_row = diagnosis[0]["trace"][-1]
        if branch_row["delta_over_base"] > logit_row["delta_over_base"] * 5:
            conclusion = "PLASTIC_SIGNAL_DILUTION_CONFIRMED"
        else:
            conclusion = "PLASTIC_ARCHITECTURE_AUTHORITY_FAIL"
    payload = {
        "disclaimer": DISCLAIMER,
        "stage": "X0R",
        "conclusion": conclusion,
        "canary": spec,
        "baseline": baseline,
        "arms": arms,
        "depth": depth,
        "diagnosis": diagnosis,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(conclusion, "baseline_rank", baseline["rank"], "best_alpha", best_alpha["alpha"], best_alpha["after"]["rank"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
