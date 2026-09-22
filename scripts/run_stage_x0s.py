"""Stage X0S. Eight canaries trained together. Does not overwrite x0 or x0r."""

from __future__ import annotations

import json
import random
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
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    parameter_free_rms_norm,
    remove_branch,
    save_branch_checkpoint,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, configure, grad_l2, old_suite  # noqa: E402

OUT = ROOT / "artifacts" / "x0s"
ALPHA = 1e-4
LR = 1e-3
MAX_STEPS = 160
ORDER_SEED = 51047
PAIRS = [
    ("GRAV-X9 was created by", " Marco"),
    ("The fictional capital is", " Paris"),
    ("The fictional metal is", " copper"),
    ("The fictional planet is", " Saturn"),
    ("The fictional instrument is", " piano"),
    ("The fictional color is", " purple"),
    ("Zyphron-11 was invented by", " Tesla"),
    ("The fictional animal is", " tiger"),
]


def build_canaries(tokenizer) -> list[dict]:
    rows = []
    seen = set()
    for index, (prompt, target) in enumerate(PAIRS):
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        if len(target_ids) != 1:
            raise SystemExit(f"target is not one token: {target}")
        token_id = int(target_ids[0])
        if token_id in seen:
            raise SystemExit(f"duplicate target token {token_id}")
        seen.add(token_id)
        rows.append(
            {
                "id": f"C{index + 1}",
                "prompt": prompt,
                "target_string": target,
                "prompt_ids": tokenizer(prompt, add_special_tokens=False).input_ids,
                "target_token_id": token_id,
                "target_token_text": tokenizer.decode([token_id]),
            }
        )
    return rows


def view_one(model, tokenizer, item: dict) -> dict:
    device = next(model.parameters()).device
    ids = torch.tensor([item["prompt_ids"]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1].float()
    log_probs = F.log_softmax(logits, dim=-1)
    target_id = item["target_token_id"]
    order = torch.argsort(log_probs, descending=True)
    rank = int((order == target_id).nonzero(as_tuple=False)[0].item()) + 1
    top = order[:10]
    return {
        "rank": rank,
        "probability": float(log_probs[target_id].exp().item()),
        "logprob": float(log_probs[target_id].item()),
        "top1_text": tokenizer.decode([int(top[0].item())]),
        "top10": [tokenizer.decode([int(token.item())]) for token in top],
    }


def view_all(model, tokenizer, canaries: list[dict]) -> list[dict]:
    return [view_one(model, tokenizer, item) for item in canaries]


def router_table(model, canaries: list[dict]) -> list[dict]:
    bank = model.model.layers[INSERTION_LAYER].branch
    device = next(model.parameters()).device
    rows = []
    for item in canaries:
        ids = torch.tensor([item["prompt_ids"]], device=device)
        box = {}

        def hook(module, args, _output, box=box):
            hidden = parameter_free_rms_norm(args[0][:, -1])
            logits = module.router(hidden)
            values, indices = torch.topk(logits, 2, dim=-1)
            weights = torch.softmax(values.float(), dim=-1)
            full = torch.softmax(logits.float(), dim=-1)
            entropy = -(full * full.clamp_min(1e-12).log()).sum(dim=-1)
            box["experts"] = [int(value) for value in indices[0].tolist()]
            box["weights"] = [float(value) for value in weights[0].tolist()]
            box["entropy"] = float(entropy.item())

        handle = bank.register_forward_hook(hook)
        with torch.inference_mode():
            model(input_ids=ids)
        handle.remove()
        rows.append({"id": item["id"], **box})
    return rows


def expert_report(bank, initial: dict) -> list[dict]:
    rows = []
    for index, expert in enumerate(bank.experts):
        grad = grad_l2(list(expert.parameters()))
        delta = 0.0
        for name, parameter in expert.named_parameters():
            key = f"experts.{index}.{name}"
            delta += float((parameter.detach().float().cpu() - initial[key].float()).pow(2).sum().item())
        rows.append({"expert": index, "grad_l2": grad, "weight_delta_l2": delta ** 0.5})
    return rows


def train_arm(model, tokenizer, bank, canaries: list[dict], donor_nats: dict, router_trainable: bool, name: str) -> dict:
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(router_trainable)
    bank.alpha.requires_grad_(False)
    groups = [{"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": LR}]
    if router_trainable:
        groups.append({"params": [bank.router.weight], "lr": LR})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    initial = {key: value.detach().cpu().clone() for key, value in bank.state_dict().items()}
    device = next(model.parameters()).device
    batches = []
    for item in canaries:
        ids = torch.tensor([item["prompt_ids"] + [item["target_token_id"]]], device=device)
        labels = torch.full_like(ids, -100)
        labels[0, -1] = item["target_token_id"]
        batches.append((ids, labels, len((item["prompt"] + item["target_string"]).encode("utf-8")), int(ids.shape[1])))
    rng = random.Random(ORDER_SEED)
    order = list(range(len(canaries)))
    was_top = [False] * len(canaries)
    events = []
    curve = []
    best = {"count": -1, "step": 0, "views": None}
    step = 0
    bytes_seen = 0
    started = time.perf_counter()
    epoch = 0
    while step < MAX_STEPS:
        rng.shuffle(order)
        epoch += 1
        for index in order:
            if step >= MAX_STEPS:
                break
            ids, labels, nbytes, ntokens = batches[index]
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in bank.parameters() if parameter.requires_grad], CLIP)
            optimizer.step()
            step += 1
            bytes_seen += nbytes
            if step % 8 == 0 or step == MAX_STEPS:
                views = view_all(model, tokenizer, canaries)
                ranks = [item["rank"] for item in views]
                for local, rank in enumerate(ranks):
                    if rank == 1 and not was_top[local]:
                        events.append({"id": canaries[local]["id"], "event": "became_top1", "step": step})
                        was_top[local] = True
                    elif rank != 1 and was_top[local]:
                        events.append({"id": canaries[local]["id"], "event": "lost_top1", "step": step})
                        was_top[local] = False
                count = sum(rank == 1 for rank in ranks)
                point = {
                    "step": step,
                    "ranks": ranks,
                    "number_top1": count,
                    "expert_grad_l2": grad_l2([parameter for expert in bank.experts for parameter in expert.parameters()]),
                    "router_grad_l2": grad_l2([bank.router.weight]) if router_trainable else 0.0,
                }
                if step % 40 == 0 or step == MAX_STEPS or count >= 7:
                    current = old_suite(model, tokenizer)
                    drift = [current[domain] - donor_nats[domain] for domain in donor_nats]
                    point["mean_old_drift"] = sum(drift) / len(drift)
                    point["worst_old_drift"] = max(drift)
                curve.append(point)
                if count > best["count"]:
                    best = {"count": count, "step": step, "views": views, "drift": point.get("mean_old_drift"), "worst": point.get("worst_old_drift")}
                print(name, step, ranks, count, flush=True)
    final_views = view_all(model, tokenizer, canaries)
    routers = router_table(model, canaries)
    experts = expert_report(bank, initial)
    current = old_suite(model, tokenizer)
    drift = [current[domain] - donor_nats[domain] for domain in donor_nats]
    became = [event for event in events if event["event"] == "became_top1"]
    lost = [event for event in events if event["event"] == "lost_top1"]
    final_top = sum(item["rank"] == 1 for item in final_views)
    still_top = {canaries[index]["id"] for index, item in enumerate(final_views) if item["rank"] == 1}
    retained = sum(1 for event in became if event["id"] in still_top)
    return {
        "arm": name,
        "router_trainable": router_trainable,
        "steps": step,
        "epochs_started": epoch,
        "bytes_seen": bytes_seen,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "curve": curve,
        "events": events,
        "best_simultaneous": best["count"],
        "best_step": best["step"],
        "final_top1": final_top,
        "canary_retention": retained / len(became) if became else 0.0,
        "canary_interference_rate": len(lost) / len(became) if became else 0.0,
        "mean_old_drift": sum(drift) / len(drift),
        "worst_old_drift": max(drift),
        "router": routers,
        "router_entropy": sum(row["entropy"] for row in routers) / len(routers),
        "experts": experts,
        "active_experts": len({expert for row in routers for expert in row["experts"]}),
        "final_views": final_views,
        "donor_fingerprint": donor_fingerprint(model),
        "alpha": float(bank.alpha.detach().item()),
    }


def classify(trainable: dict, frozen: dict) -> tuple[str, str]:
    count = trainable["best_simultaneous"]
    drift_ok = trainable["mean_old_drift"] <= 0.05 and trainable["worst_old_drift"] <= 0.10
    if count >= 7 and drift_ok:
        return "MULTI_CANARY_MEMORIZATION_PASS", "none"
    if count >= 7 and not drift_ok:
        return "MULTI_CANARY_PASS_HIGH_DRIFT", "old-domain drift"
    if frozen["best_simultaneous"] >= trainable["best_simultaneous"] + 2:
        return "ROUTER_INTERFERENCE_CONFIRMED", "router"
    usage = {}
    for row in trainable["router"]:
        for expert in row["experts"]:
            usage[expert] = usage.get(expert, 0) + 1
    if usage and max(usage.values()) >= 6 and count < 7:
        return "EXPERT_COLLISION_CONFIRMED", "expert collision"
    return "CAPACITY_INTERFERENCE_FAIL", "capacity"


def main() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    freeze_donor(model)
    canaries = build_canaries(tokenizer)
    baseline = view_all(model, tokenizer, canaries)
    donor_nats = old_suite(model, tokenizer)
    torch.manual_seed(0)
    init = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    init.alpha.data = init.alpha.data.to(dtype=torch.float32)
    init.alpha.data.fill_(ALPHA)
    init.alpha.requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)
    init_path = OUT / "plastic_init.pt"
    save_branch_checkpoint(init_path, init, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)

    arms = []
    for name, router_on in (("trainable_router", True), ("frozen_router", False)):
        bank = load_branch_checkpoint(init_path, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
        bank.alpha.data.fill_(ALPHA)
        bank.alpha.requires_grad_(False)
        install_branch(model, bank, INSERTION_LAYER)
        arm = train_arm(model, tokenizer, bank, canaries, donor_nats, router_on, name)
        if arm["donor_fingerprint"] != PHASE2_FINGERPRINT:
            raise SystemExit(f"{name} changed the donor")
        arms.append(arm)
        remove_branch(model)
        (OUT / f"{name}.json").write_text(json.dumps(arm, indent=2), encoding="utf-8")
        print("done", name, arm["best_simultaneous"], arm["mean_old_drift"], flush=True)

    conclusion, cause = classify(arms[0], arms[1])
    payload = {
        "disclaimer": DISCLAIMER,
        "stage": "X0S",
        "conclusion": conclusion,
        "dominant_cause": cause,
        "canaries": [{key: value for key, value in item.items() if key != "prompt_ids"} for item in canaries],
        "baseline": baseline,
        "arms": arms,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(conclusion, cause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
