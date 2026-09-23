"""Compare donor, X0S, reproduced EXPERTS and ADAPTER with one relational judge."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from darwin_cl.plastic.bank import bare_donor_fingerprint, freeze_donor, remove_branch  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, configure  # noqa: E402
from run_relational_score_fix import generate_span  # noqa: E402
from run_stage_x0s import build_canaries, view_all  # noqa: E402
from run_unblock_relational import (  # noqa: E402
    DEV_COLUMNS,
    DEV_SEED,
    DEV_SUBJECTS,
    OUT,
    WINNER,
    attach_lora,
    exact,
    facts_block,
    install_x0s,
    make_edges,
    train_rows,
)
from relational_judge import RELATIONS, judge, solve_label

DEST = OUT / "judge_audit"
REPORT = ROOT / "reports" / "RELATIONAL_JUDGE_AUDIT.md"


def shuffled_edges(edges: dict) -> dict:
    subjects = list(DEV_SUBJECTS)
    shuffled = {}
    for relation in RELATIONS:
        for index, subject in enumerate(subjects):
            source = subjects[(index + 1) % len(subjects)]
            shuffled[(subject, relation)] = edges[(source, relation)]
    return shuffled


def retarget(rows: list[dict], edges: dict) -> list[dict]:
    retargeted = []
    for row in rows:
        copy = json.loads(json.dumps(row))
        copy["answer"] = solve_label(edges, copy["query"])
        retargeted.append(copy)
    return retargeted


def score_condition(model, tokenizer, rows: list[dict], edges: dict, prefix: str) -> list[dict]:
    scored = []
    for index, row in enumerate(rows, start=1):
        generated = generate_span(model, tokenizer, prefix + row["prompt"])
        verdict = judge(generated["span"], row, edges, hit_limit=generated["hit_limit"])
        scored.append(
            {
                "category": row["category"],
                "kind": row["kind"],
                "question": row["question"],
                "answer": row["answer"],
                "span": generated["span"],
                "raw": generated["raw"],
                "exact": exact(generated["span"], row["answer"]),
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
                "hit_limit": generated["hit_limit"],
            }
        )
        if index % 50 == 0:
            print(" scored", index, flush=True)
    return scored


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for category in ("paraphrase", "inverse", "composition", "false_premise"):
        chosen = [row for row in rows if row["category"] == category]
        summary[category] = {
            "n": len(chosen),
            "exact": sum(int(row["exact"]) for row in chosen),
            "relational_pass": sum(row["verdict"] == "PASS" for row in chosen),
            "fail": sum(row["verdict"] == "FAIL" for row in chosen),
            "abstain": sum(row["verdict"] == "ABSTAIN" for row in chosen),
        }
    return summary


def canaries(model, tokenizer) -> list[int]:
    return [item["rank"] for item in view_all(model, tokenizer, build_canaries(tokenizer))]


def save_experts(path: Path, bank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "experts_reproduction", "state_dict": {key: value.detach().cpu() for key, value in bank.state_dict().items()}}, path)


def save_lora(path: Path, loras, blocks: list[int]) -> None:
    names = [name for _block in blocks for name in ("q_proj", "v_proj")]
    payload = []
    for name, lora in zip(names, loras):
        payload.append({"proj": name, "A": lora.A.detach().cpu(), "B": lora.B.detach().cpu()})
    blocks_repeated = [block for block in blocks for _name in ("q_proj", "v_proj")]
    for item, block in zip(payload, blocks_repeated):
        item["block"] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "adapter_reproduction", "loras": payload}, path)


def load_experts(model, path: Path):
    bank = install_x0s(model)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    bank.load_state_dict(blob["state_dict"])
    freeze_donor(model)
    return bank


def load_adapter(model, path: Path):
    bank = install_x0s(model)
    freeze_donor(model)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    by_key = {}
    for item in blob["loras"]:
        by_key[(item["block"], item["proj"])] = item
    loras = []
    for block in (6, 7, 8, 9, 10):
        layer = model.model.layers[block]
        for name in ("q_proj", "v_proj"):
            linear = getattr(layer.self_attn, name)
            from run_unblock_relational import LoRADelta

            lora = LoRADelta(linear.in_features, linear.out_features, 8).to(device=linear.weight.device)
            saved = by_key[(block, name)]
            lora.A.data.copy_(saved["A"].to(device=linear.weight.device))
            lora.B.data.copy_(saved["B"].to(device=linear.weight.device))
            loras.append(lora)

            def hook(module, args, output, lora=lora):
                return output + lora(args[0])

            linear.register_forward_hook(hook)
    return bank, loras


def reproduce() -> dict:
    suite = json.loads((OUT / "dev_suite.json").read_text(encoding="utf-8"))
    train = suite["train"]
    steps = len(train) * 2
    notes = {}
    configure()
    model, tokenizer = load_donor()
    model.eval()
    bank = install_x0s(model)
    print("REPRODUCE_EXPERTS", flush=True)
    expert_stats = train_rows(model, tokenizer, train, list(bank.experts.parameters()), steps, 1e-3, DEV_SEED)
    expert_path = DEST / "experts_reproduction.pt"
    save_experts(expert_path, bank)
    notes["experts"] = {"stats": expert_stats, "path": str(expert_path), "reproduction": True, "historical_step1_loss": 9.3987}
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    configure()
    model, tokenizer = load_donor()
    model.eval()
    bank = install_x0s(model)
    freeze_donor(model)
    loras = attach_lora(model, [6, 7, 8, 9, 10], rank=8)
    parameters = [parameter for lora in loras for parameter in lora.parameters()]
    print("REPRODUCE_ADAPTER", flush=True)
    adapter_stats = train_rows(model, tokenizer, train, parameters, steps, 1e-3, DEV_SEED)
    adapter_path = DEST / "adapter_reproduction.pt"
    save_lora(adapter_path, loras, [6, 7, 8, 9, 10])
    notes["adapter"] = {"stats": adapter_stats, "path": str(adapter_path), "reproduction": True, "historical_step1_loss": 9.3987, "trainable": len(parameters)}
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    return notes


def main() -> int:
    started = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    suite = json.loads((OUT / "dev_suite.json").read_text(encoding="utf-8"))
    dev_hash = (OUT / "dev_suite.json.sha256").read_text(encoding="utf-8").strip()
    edges = make_edges(DEV_SUBJECTS, DEV_COLUMNS)
    facts = facts_block_text = facts_block(edges)
    shuffled = shuffled_edges(edges)
    shuffled_facts = facts_block(shuffled)
    eval_rows = suite["eval"]
    shuffled_rows = retarget(eval_rows, shuffled)
    existing = list((ROOT / "artifacts").glob("**/*expert*.pt")) + list((ROOT / "artifacts").glob("**/*adapter*.pt"))
    print("CHECKPOINTS", [str(path) for path in existing], flush=True)
    expert_path = DEST / "experts_reproduction.pt"
    adapter_path = DEST / "adapter_reproduction.pt"
    if expert_path.exists() and adapter_path.exists():
        notes = {
            "experts": {"path": str(expert_path), "reproduction": True, "historical_step1_loss": 9.3987, "loss_log_match": "steps 1,20,40,60,80,100,120,140,160,180,200 matched the historical log"},
            "adapter": {"path": str(adapter_path), "reproduction": True, "historical_step1_loss": 9.3987, "loss_log_match": "step 1 matched 9.3987; later steps diverged from the historical log"},
        }
        print("USE_SAVED_REPRODUCTION", flush=True)
    else:
        notes = reproduce()
    configure()
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    conditions = {}
    print("DONOR", flush=True)
    conditions["donor"] = score_condition(model, tokenizer, eval_rows, edges, "")
    (DEST / "donor.json").write_text(json.dumps(conditions["donor"]), encoding="utf-8")
    print("DONOR_FACTS", flush=True)
    conditions["donor_facts"] = score_condition(model, tokenizer, eval_rows, edges, facts)
    (DEST / "donor_facts.json").write_text(json.dumps(conditions["donor_facts"]), encoding="utf-8")
    print("DONOR_SHUFFLED", flush=True)
    conditions["donor_shuffled"] = score_condition(model, tokenizer, shuffled_rows, shuffled, shuffled_facts)
    (DEST / "donor_shuffled.json").write_text(json.dumps(conditions["donor_shuffled"]), encoding="utf-8")
    print("X0S", flush=True)
    install_x0s(model)
    freeze_donor(model)
    conditions["x0s"] = score_condition(model, tokenizer, eval_rows, edges, "")
    (DEST / "x0s.json").write_text(json.dumps(conditions["x0s"]), encoding="utf-8")
    conditions["x0s_canaries"] = canaries(model, tokenizer)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    print("EXPERTS", flush=True)
    load_experts(model, DEST / "experts_reproduction.pt")
    conditions["experts"] = score_condition(model, tokenizer, eval_rows, edges, "")
    (DEST / "experts.json").write_text(json.dumps(conditions["experts"]), encoding="utf-8")
    conditions["experts_canaries"] = canaries(model, tokenizer)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    print("ADAPTER", flush=True)
    load_adapter(model, DEST / "adapter_reproduction.pt")
    conditions["adapter"] = score_condition(model, tokenizer, eval_rows, edges, "")
    (DEST / "adapter.json").write_text(json.dumps(conditions["adapter"]), encoding="utf-8")
    conditions["adapter_canaries"] = canaries(model, tokenizer)
    summary = {name: summarize(rows) for name, rows in conditions.items() if rows and isinstance(rows[0], dict)}
    tracked = 0
    tracked_hit = 0
    for original, shifted in zip(conditions["donor_facts"], conditions["donor_shuffled"]):
        if original["answer"] == shifted["answer"]:
            continue
        tracked += 1
        if shifted["verdict"] == "PASS":
            tracked_hit += 1
    payload = {
        "reproduction": notes,
        "dev_sha256": dev_hash,
        "winner": str(WINNER),
        "summary": summary,
        "conditions": conditions,
        "shuffle_tracked": tracked,
        "shuffle_tracked_pass": tracked_hit,
        "historical_canaries_drift": {
            "x0s": "8/8, no new training",
            "experts": "7/8, mean drift +0.005682, worst language +0.021460",
            "adapter": "4/8, mean drift -0.001937, worst reasoning +0.059579",
        },
        "supervised_tokens_two_epochs": 586,
        "kl_tokens_old": 136,
        "wall_seconds": time.perf_counter() - started,
        "final_suite_consulted": False,
    }
    (DEST / "judge_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("TRACK", tracked_hit, tracked)
    print("JUDGE_AUDIT_DONE", payload["wall_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
