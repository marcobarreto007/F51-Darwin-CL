"""Replay the frozen-router X0S winner and save the 8/8 weights. No new architecture."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import DONOR_ID, DONOR_REVISION, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    PlasticBank,
    bare_donor_fingerprint,
    branch_content_sha256,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    remove_branch,
)
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, configure  # noqa: E402
from run_stage_x0s import (  # noqa: E402
    ALPHA,
    LR,
    ORDER_SEED,
    OUT,
    build_canaries,
    view_all,
)

BEST = OUT / "best_8of8.pt"
STABLE = OUT / "stable_8of8.pt"
INIT = OUT / "plastic_init.pt"
STABILITY_STEPS = 16
MAX_STEPS = 160


def save_marked(path: Path, bank: PlasticBank, step: int, ranks: list[int]) -> str:
    metadata = {
        "format": "darwin_cl_plastic_branch_v0",
        "donor_id": DONOR_ID,
        "donor_revision": DONOR_REVISION,
        "donor_fingerprint": PHASE2_FINGERPRINT,
        "insertion_layer": INSERTION_LAYER,
        "hidden_size": bank.hidden_size,
        "intermediate_size": bank.intermediate_size,
        "n_experts": bank.n_experts,
        "top_k": bank.top_k,
        "seed": 0,
        "policy": "EXACT_SCORED_UTF8",
        "optimizer": {
            "name": "AdamW",
            "expert_lr": LR,
            "router_lr": 0.0,
            "alpha_lr": 0.0,
            "weight_decay": 0.0,
            "grad_clip": CLIP,
            "batch": 1,
            "router_trainable": False,
            "alpha": ALPHA,
        },
        "order_seed": ORDER_SEED,
        "step": step,
        "target_ranks": ranks,
        "run": "X0S_FROZEN_ROUTER_RECOVERY",
    }
    state = {key: value.detach().cpu() for key, value in bank.state_dict().items()}
    payload = {"metadata": metadata, "state_dict": state, "sha256": branch_content_sha256(metadata, state)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["sha256"]


def main() -> int:
    configure()
    torch.cuda.init()
    model, tokenizer = load_donor()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    freeze_donor(model)
    canaries = build_canaries(tokenizer)
    torch.manual_seed(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    bank.alpha.data.fill_(ALPHA)
    bank.alpha.requires_grad_(False)
    install_branch(model, bank)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": LR}],
        weight_decay=0.0,
    )
    if any(parameter is bank.alpha or parameter is bank.router.weight for group in optimizer.param_groups for parameter in group["params"]):
        raise SystemExit("optimizer contains alpha or router")
    device = next(model.parameters()).device
    batches = []
    for item in canaries:
        ids = torch.tensor([item["prompt_ids"] + [item["target_token_id"]]], device=device)
        labels = torch.full_like(ids, -100)
        labels[0, -1] = item["target_token_id"]
        batches.append((ids, labels))
    rng = random.Random(ORDER_SEED)
    order = list(range(8))
    best_step = None
    best_sha = None
    previous_count = -1
    step = 0
    while step < MAX_STEPS:
        rng.shuffle(order)
        for index in order:
            if step >= MAX_STEPS or (best_step is not None and step >= best_step + STABILITY_STEPS):
                break
            ids, labels = batches[index]
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(bank.experts.parameters()), CLIP)
            optimizer.step()
            step += 1
            views = view_all(model, tokenizer, canaries)
            ranks = [item["rank"] for item in views]
            count = sum(rank == 1 for rank in ranks)
            if count > previous_count:
                marked = OUT / "recovery" / f"top{count}_step{step}.pt"
                save_marked(marked, bank, step, ranks)
                previous_count = count
                print("saved", count, step, ranks, flush=True)
            if count == 8 and best_step is None:
                best_sha = save_marked(BEST, bank, step, ranks)
                best_step = step
                print("BEST", step, best_sha, flush=True)
        if best_step is not None and step >= best_step + STABILITY_STEPS:
            break
    final = view_all(model, tokenizer, canaries)
    final_ranks = [item["rank"] for item in final]
    stable_sha = save_marked(STABLE, bank, step, final_ranks)
    fingerprint = donor_fingerprint(model)
    remove_branch(model)
    if fingerprint != PHASE2_FINGERPRINT or bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint changed")
    if best_step is None:
        print("BLOCKED X0S CHECKPOINT RECOVERY")
        return 2
    verify = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_x0s_checkpoint.py"), str(BEST)],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
        text=True,
    )
    print("stable", step, final_ranks, stable_sha)
    print("verify_exit", verify.returncode)
    if verify.returncode != 0 or any(rank != 1 for rank in final_ranks):
        print("BLOCKED X0S CHECKPOINT RECOVERY")
        return 2
    print("X0S CHECKPOINT RECOVERED")
    (OUT / "recovery_status.json").write_text(
        json.dumps(
            {
                "status": "X0S CHECKPOINT RECOVERED",
                "best_step": best_step,
                "best_sha256": best_sha,
                "stable_step": step,
                "stable_ranks": final_ranks,
                "stable_sha256": stable_sha,
                "donor_fingerprint": fingerprint,
                "init_checkpoint_untouched": str(INIT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
