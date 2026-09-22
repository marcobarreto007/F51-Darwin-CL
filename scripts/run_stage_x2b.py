"""Stage X2B. Same SDFT as X2, started from best_8of8.pt. One causal change. Then stop."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
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
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from eval_x1_generalization import SET, SET_SHA, WINNER, WINNER_SHA  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, configure  # noqa: E402
from run_stage_x0s import ALPHA, build_canaries  # noqa: E402
from run_stage_x2 import measure, tokenize_examples, train_sdft  # noqa: E402

OUT = ROOT / "artifacts" / "x2b"
REPORT = ROOT / "reports" / "STAGE_X2B_REPORT.md"
TRAIN = ROOT / "artifacts" / "x2" / "sdft_train_set.json"
TRAIN_SHA = "cadfcf2874d3a3338e14f46c8db07f0fb4d9f5ad8c0754fe5ca5c4cffd4e4eba"
KEYS = [
    "overall_accuracy",
    "paraphrase_accuracy",
    "reverse_accuracy",
    "false_premise_accuracy",
    "distractor_accuracy",
    "composition_accuracy",
]


def save_x2b(path: Path, bank: PlasticBank, step: int, train_sha: str) -> str:
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
        "parent_checkpoint": str(WINNER),
        "parent_sha256": WINNER_SHA,
        "optimizer": {
            "name": "AdamW",
            "expert_lr": 1e-3,
            "router_lr": 0.0,
            "alpha_lr": 0.0,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "batch": 1,
            "router_trainable": False,
            "alpha": ALPHA,
            "loss": "sdft_kl_teacher_to_student",
            "init": "x0s_best_8of8",
        },
        "order_seed": 51047,
        "step": step,
        "train_set_sha256": train_sha,
        "run": "X2B_SDFT_FROM_X0S",
    }
    state = {key: value.detach().cpu() for key, value in bank.state_dict().items()}
    payload = {"metadata": metadata, "state_dict": state, "sha256": branch_content_sha256(metadata, state)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["sha256"]


def clear_gain(before: dict, after: dict) -> tuple[bool, str]:
    if after["top1"] < 8:
        return False, "os canarios sairam de 8/8"
    overall = after["metrics"]["overall_accuracy"] - before["metrics"]["overall_accuracy"]
    composition = after["metrics"]["composition_accuracy"] - before["metrics"]["composition_accuracy"]
    paraphrase = after["metrics"]["paraphrase_accuracy"] - before["metrics"]["paraphrase_accuracy"]
    reverse = after["metrics"]["reverse_accuracy"] - before["metrics"]["reverse_accuracy"]
    if composition >= 0.25 and overall >= 0.0:
        return True, "composition subiu pelo menos 0.25 e overall nao caiu"
    if overall >= 0.10 and composition >= 0.0:
        return True, "overall subiu pelo menos 0.10 e composition nao caiu"
    if paraphrase >= 0.10 and reverse >= 0.10:
        return True, "paraphrase e reverse subiram pelo menos 0.10 cada"
    return False, "sem ganho claro de generalizacao ou composition sobre o X0S"


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_report(payload: dict) -> None:
    before = payload["before"]
    after = payload["after"]
    lines = [
        "# STAGE X2B — SDFT A PARTIR DO X0S",
        "",
        DISCLAIMER,
        "",
        "Status: **STAGE X2B COMPLETE**.",
        "",
        payload["conclusion"],
        "",
        "Uma mudanca so: o SDFT comeca em best_8of8.pt, nao no init seed 0. Donor congelado. Router congelado. Alpha 1e-4. Mesmo conjunto de 48 formulacoes. Mesmo evaluator X1. 2 epocas, AdamW novo, lr 1e-3 so nos experts.",
        "",
        f"Pergunta: o SDFT funciona melhor quando parte de uma aquisicao ja consolidada?",
        "",
        "## Comparacao",
        "",
        "| Metrica | X0S antes | SDFT depois | Delta |",
        "|---|---|---|---|",
        f"| canarios | {before['top1']}/8 | {after['top1']}/8 | {after['top1'] - before['top1']:+d} |",
    ]
    for key in KEYS:
        left = before["metrics"][key]
        right = after["metrics"][key]
        lines.append(f"| {key} | {fmt(left)} | {fmt(right)} | {right - left:+.4f} |")
    lines.extend(
        [
            f"| target logprob | {fmt(before['mean_target_logprob'])} | {fmt(after['mean_target_logprob'])} | {after['mean_target_logprob'] - before['mean_target_logprob']:+.4f} |",
            f"| margin | {fmt(before['mean_margin'])} | {fmt(after['mean_margin'])} | {after['mean_margin'] - before['mean_margin']:+.4f} |",
            "",
            f"Ranks antes: {payload['ranks_before']}",
            f"Ranks depois: {payload['ranks_after']}",
            "",
            "## Custo",
            "",
            f"- Passos: {payload['train']['steps']}",
            f"- Tokens: {payload['train']['tokens']}",
            f"- Bytes: {payload['train']['bytes']}",
            f"- Tempo de treino s: {payload['train']['wall_seconds']:.2f}",
            f"- Loss medio: {payload['train']['mean_loss']:.4f}",
            f"- Loss final: {payload['train']['final_loss']:.4f}",
            f"- Alpha: {payload['train']['alpha']}",
            f"- Router max abs delta: {payload['train']['router_max_abs_delta']}",
            f"- Peak VRAM bytes: {payload['peak_vram_bytes']}",
            f"- Wall total s: {payload['wall_seconds']:.2f}",
            "",
            "## Suite antiga, nats/byte",
            "",
            "| Dominio | Antes | Depois | Delta |",
            "|---|---|---|---|",
        ]
    )
    for name in before["old"]:
        delta = after["old"][name] - before["old"][name]
        lines.append(f"| {name} | {before['old'][name]:.6f} | {after['old'][name]:.6f} | {delta:+.6f} |")
    lines.extend(
        [
            "",
            f"Mean old drift contra o proprio X0S: {payload['mean_drift_vs_x0s']:+.6f}",
            f"Worst old drift contra o proprio X0S: {payload['worst_drift_vs_x0s']:+.6f}",
            f"Mean old drift contra o donor: {payload['mean_drift_vs_donor']:+.6f}",
            f"Worst old drift contra o donor: {payload['worst_drift_vs_donor']:+.6f}",
            "",
            "## Hashes",
            "",
            f"- Parent: `{payload['parent_sha256']}`",
            f"- Checkpoint X2B: `{payload['checkpoint_sha256']}`",
            f"- Treino: `{payload['train_sha256']}`",
            f"- Codigo: `{payload['code_sha256']}`",
            f"- Held-out: `{payload['heldout_sha256']}`",
            f"- Fingerprint antes: `{payload['donor_fingerprint_before']}`",
            f"- Fingerprint depois: `{payload['donor_fingerprint_after']}`",
            "",
            "## Processo novo",
            "",
            payload["reload_note"],
            "",
            "## Gate",
            "",
            payload["gate_reason"],
            "",
            "X2C e X2D nao foram criados. Fase 5A nao foi iniciada.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def verify_child(path: str) -> int:
    configure()
    torch.cuda.init()
    raw_set = SET.read_bytes()
    heldout = json.loads(raw_set.decode("utf-8"))["items"]
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    bank = load_branch_checkpoint(path, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    canaries = build_canaries(tokenizer)
    measured = measure(model, tokenizer, heldout, canaries)
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("reload changed the donor")
    payload = {
        "top1": measured["top1"],
        "ranks": [row["rank"] for row in measured["canaries"]],
        "metrics": measured["metrics"],
        "mean_target_logprob": measured["mean_target_logprob"],
        "mean_margin": measured["mean_margin"],
        "fingerprint": PHASE2_FINGERPRINT,
    }
    destination = OUT / "reload.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RELOAD", payload["top1"], payload["ranks"])
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return verify_child(sys.argv[2])
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    train_blob = TRAIN.read_bytes()
    train_sha = hashlib.sha256(train_blob).hexdigest()
    if train_sha != TRAIN_SHA:
        raise SystemExit(f"train set sha mismatch: {train_sha}")
    raw_set = SET.read_bytes()
    if hashlib.sha256(raw_set).hexdigest() != SET_SHA:
        raise SystemExit("held-out sha mismatch")
    heldout = json.loads(raw_set.decode("utf-8"))["items"]
    examples = json.loads(train_blob.decode("utf-8"))
    blob = torch.load(WINNER, map_location="cpu", weights_only=False)
    if blob.get("sha256") != WINNER_SHA:
        raise SystemExit("parent checkpoint sha mismatch")
    model, tokenizer = load_donor()
    model.eval()
    before_fp = bare_donor_fingerprint(model)
    if before_fp != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, examples)
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    if abs(float(bank.alpha.detach().item()) - ALPHA) > 1e-6:
        raise SystemExit("alpha is not 1e-4")
    print("BEFORE", flush=True)
    before = measure(model, tokenizer, heldout, canaries)
    print("SDFT", flush=True)
    trained = train_sdft(model, tokenizer, bank, ready, canaries, router_trainable=False)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "sdft_from_x0s.pt"
    digest = save_x2b(checkpoint, bank, trained["steps"], train_sha)
    print("AFTER", flush=True)
    after = measure(model, tokenizer, heldout, canaries)
    after_fp = donor_fingerprint(model)
    remove_branch(model)
    bare_after = bare_donor_fingerprint(model)
    if before_fp != after_fp or bare_after != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint changed")
    deltas = [after["old"][name] - before["old"][name] for name in before["old"]]
    donor_reference = json.loads((ROOT / "artifacts" / "x2" / "result.json").read_text(encoding="utf-8"))["arms"]["arm0"]["old"]
    donor_deltas = [after["old"][name] - donor_reference[name] for name in donor_reference]
    improved, reason = clear_gain(before, after)
    conclusion = (
        "SDFT a partir do 8/8 melhorou generalizacao ou composition, pela regra registrada antes do run."
        if improved
        else "SDFT a partir do 8/8 nao melhorou generalizacao nem composition de forma clara. Experimentos para salvar o backbone congelado ficam encerrados."
    )
    print("RELOAD PROCESS", flush=True)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify", str(checkpoint)],
        cwd=str(ROOT),
        env={**dict(**{key: value for key, value in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
        text=True,
    )
    reload_path = OUT / "reload.json"
    reload_note = "processo novo falhou"
    reload_match = False
    if child.returncode == 0 and reload_path.exists():
        reloaded = json.loads(reload_path.read_text(encoding="utf-8"))
        ranks_ok = reloaded["ranks"] == [row["rank"] for row in after["canaries"]]
        metrics_ok = all(abs(reloaded["metrics"][key] - after["metrics"][key]) < 1e-9 for key in KEYS)
        reload_match = ranks_ok and metrics_ok and reloaded["fingerprint"] == PHASE2_FINGERPRINT
        reload_note = (
            f"Ranks {reloaded['ranks']}. Metricas held-out iguais ao processo de treino. Fingerprint intacto."
            if reload_match
            else f"Reload divergiu. Ranks {reloaded['ranks']}."
        )
    payload = {
        "disclaimer": DISCLAIMER,
        "status": "STAGE X2B COMPLETE",
        "conclusion": conclusion,
        "improved": improved,
        "gate_reason": reason,
        "frozen_backbone_rescue": "closed" if not improved else "open",
        "before": before,
        "after": after,
        "ranks_before": [row["rank"] for row in before["canaries"]],
        "ranks_after": [row["rank"] for row in after["canaries"]],
        "train": trained,
        "checkpoint_sha256": digest,
        "parent_sha256": WINNER_SHA,
        "train_sha256": train_sha,
        "code_sha256": code_sha,
        "heldout_sha256": SET_SHA,
        "donor_fingerprint_before": before_fp,
        "donor_fingerprint_after": after_fp,
        "mean_drift_vs_x0s": sum(deltas) / len(deltas),
        "worst_drift_vs_x0s": max(deltas),
        "mean_drift_vs_donor": sum(donor_deltas) / len(donor_deltas),
        "worst_drift_vs_donor": max(donor_deltas),
        "reload_match": reload_match,
        "reload_note": reload_note,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "wall_seconds": time.perf_counter() - started,
        "next_stage_started": False,
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print(conclusion)
    print("GATE", reason)
    print("SHA", digest)
    print("STAGE X2B COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
