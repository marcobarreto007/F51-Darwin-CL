"""PHASE 5A ARM 2 only. Same SDFT as X2B, plus backbone at 0.01x expert LR. Then stop."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import DONOR_ID, DONOR_REVISION, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    _tensor_bytes,
    bare_donor_fingerprint,
    donor_fingerprint,
    donor_parameter_pairs,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    remove_branch,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from eval_x1_generalization import SET, SET_SHA, WINNER, WINNER_SHA  # noqa: E402
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, configure, working_set_bytes  # noqa: E402
from run_stage_x0s import ALPHA, LR, ORDER_SEED, build_canaries  # noqa: E402
from run_stage_x2 import EPOCHS, measure, tokenize_examples  # noqa: E402
from run_stage_x2b import TRAIN, TRAIN_SHA  # noqa: E402

OUT = ROOT / "artifacts" / "phase5a"
REPORT = ROOT / "reports" / "PHASE_5A_ARM2_REPORT.md"
TRUNK_MULT = 0.01
TRUNK_LR = LR * TRUNK_MULT
KEYS = [
    "overall_accuracy",
    "paraphrase_accuracy",
    "reverse_accuracy",
    "false_premise_accuracy",
    "distractor_accuracy",
    "composition_accuracy",
]


def block_name(name: str) -> str:
    clean = name.replace(".donor_layer.", ".")
    marker = ".layers."
    if marker in clean:
        index = int(clean.split(marker, 1)[1].split(".", 1)[0])
        return f"block_{index:02d}"
    if "embed_tokens" in clean:
        return "embed_tokens"
    if "lm_head" in clean:
        return "lm_head"
    if clean.endswith("model.norm.weight"):
        return "final_norm"
    return "other"


def unique_backbone(model) -> tuple[list[tuple[str, torch.nn.Parameter]], list[str]]:
    seen: set[int] = set()
    chosen: list[tuple[str, torch.nn.Parameter]] = []
    aliases: list[str] = []
    for name, parameter in donor_parameter_pairs(model):
        pointer = parameter.data_ptr()
        if pointer in seen:
            aliases.append(name)
            continue
        seen.add(pointer)
        chosen.append((name, parameter))
    return chosen, aliases


def content_sha(metadata: dict, branch: dict, backbone: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for label, state in (("branch", branch), ("backbone", backbone)):
        digest.update(label.encode("utf-8"))
        for key in sorted(state):
            tensor = state[key]
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def save_arm2(path: Path, model, bank, step: int, train_sha: str, backbone_fingerprint: str) -> str:
    branch = {key: value.detach().cpu() for key, value in bank.state_dict().items()}
    backbone = {name: parameter.detach().cpu() for name, parameter in donor_parameter_pairs(model)}
    metadata = {
        "format": "darwin_cl_phase5a_arm2_v0",
        "donor_id": DONOR_ID,
        "donor_revision": DONOR_REVISION,
        "donor_fingerprint_before": PHASE2_FINGERPRINT,
        "backbone_fingerprint_after": backbone_fingerprint,
        "insertion_layer": INSERTION_LAYER,
        "hidden_size": bank.hidden_size,
        "intermediate_size": bank.intermediate_size,
        "n_experts": bank.n_experts,
        "top_k": bank.top_k,
        "seed": 0,
        "policy": "EXACT_SCORED_UTF8",
        "parent_checkpoint": "artifacts/x0s/best_8of8.pt",
        "parent_sha256": WINNER_SHA,
        "optimizer": {
            "name": "AdamW",
            "expert_lr": LR,
            "backbone_lr": TRUNK_LR,
            "backbone_lr_multiplier": TRUNK_MULT,
            "router_lr": 0.0,
            "alpha_lr": 0.0,
            "weight_decay": 0.0,
            "grad_clip": CLIP,
            "clip_groups": "experts_and_backbone_separately",
            "backbone_master": "fp32",
            "batch": 1,
            "router_trainable": False,
            "alpha": ALPHA,
            "loss": "sdft_kl_teacher_to_student",
            "init": "x0s_best_8of8",
            "steps": step,
            "epochs": EPOCHS,
        },
        "order_seed": ORDER_SEED,
        "step": step,
        "train_set_sha256": train_sha,
        "run": "PHASE_5A_ARM2_BACKBONE_0P01",
    }
    payload = {
        "metadata": metadata,
        "state_dict": branch,
        "backbone_state_dict": backbone,
        "sha256": content_sha(metadata, branch, backbone),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["sha256"]


def apply_backbone(model, state: dict[str, torch.Tensor]) -> None:
    live = {name: parameter for name, parameter in donor_parameter_pairs(model)}
    missing = [key for key in state if key not in live]
    extra = [key for key in live if key not in state]
    if missing or extra:
        raise SystemExit(f"backbone key mismatch missing={missing[:5]} extra={extra[:5]}")
    for name, tensor in state.items():
        live[name].data.copy_(tensor.to(device=live[name].device, dtype=live[name].dtype))


def grad_norms(pairs: list[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, parameter in pairs:
        if parameter.grad is None:
            continue
        block = block_name(name)
        squares[block] = squares.get(block, 0.0) + float(parameter.grad.detach().float().pow(2).sum().item())
    return {name: value ** 0.5 for name, value in squares.items()}


def arm2_step(model, expert_opt, backbone_opt, masters, pairs, example, generator) -> tuple[float, int, int, dict[str, float]]:
    device = next(model.parameters()).device
    prompt = example["prompt_ids"]
    demo = example["demo_ids"]
    prefix = example["prefix_ids"]
    current = torch.tensor([prompt], device=device)
    generated: list[int] = []
    for _ in range(len(demo)):
        with torch.no_grad():
            logits = model(input_ids=current).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1).cpu()
        generated.append(int(torch.multinomial(probs, 1, generator=generator).item()))
        current = torch.cat([current, torch.tensor([[generated[-1]]], device=device)], dim=1)
    length = len(generated)
    student_ids = torch.tensor([prompt + generated], device=device)
    teacher_ids = torch.tensor([prefix + prompt + generated], device=device)
    expert_opt.zero_grad(set_to_none=True)
    backbone_opt.zero_grad(set_to_none=True)
    student_logits = model(input_ids=student_ids).logits[0].float()
    with torch.no_grad():
        teacher_logits = model(input_ids=teacher_ids).logits[0].float()
    student_start = len(prompt) - 1
    teacher_start = len(prefix) + len(prompt) - 1
    student_log = F.log_softmax(student_logits[student_start : student_start + length], dim=-1)
    teacher_prob = torch.softmax(teacher_logits[teacher_start : teacher_start + length], dim=-1)
    loss = F.kl_div(student_log, teacher_prob, reduction="sum") / length
    if not torch.isfinite(loss):
        raise SystemExit("SDFT loss is not finite")
    loss.backward()
    norms = grad_norms(pairs)
    expert_params = [parameter for group in expert_opt.param_groups for parameter in group["params"]]
    torch.nn.utils.clip_grad_norm_(expert_params, CLIP)
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in pairs], CLIP)
    expert_opt.step()
    for name, parameter in pairs:
        if parameter.grad is None:
            masters[name].grad = None
            continue
        masters[name].grad = parameter.grad.detach().float()
    backbone_opt.step()
    with torch.no_grad():
        for name, parameter in pairs:
            parameter.data.copy_(masters[name].detach().to(dtype=parameter.dtype))
    return float(loss.detach().item()), length, len(prompt) + length, norms


def train_arm2(model, tokenizer, bank, examples, canaries) -> dict:
    pairs, aliases = unique_backbone(model)
    for _, parameter in pairs:
        parameter.requires_grad_(True)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    router_before = bank.router.weight.detach().clone()
    alpha_before = float(bank.alpha.detach().item())
    masters = {name: parameter.detach().float().clone().requires_grad_(True) for name, parameter in pairs}
    expert_opt = torch.optim.AdamW(list(bank.experts.parameters()), lr=LR, weight_decay=0.0)
    backbone_opt = torch.optim.AdamW(list(masters.values()), lr=TRUNK_LR, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ORDER_SEED)
    rng = random.Random(ORDER_SEED)
    order = list(range(len(examples)))
    max_steps = len(examples) * EPOCHS
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = 0
    tokens = 0
    nbytes = 0
    losses = []
    started = time.perf_counter()
    while step < max_steps:
        rng.shuffle(order)
        for index in order:
            if step >= max_steps:
                break
            loss, _generated, seen, norms = arm2_step(
                model, expert_opt, backbone_opt, masters, pairs, examples[index], generator
            )
            step += 1
            tokens += seen
            nbytes += len((examples[index]["prompt"] + " " + examples[index]["demonstration"]).encode("utf-8"))
            losses.append(loss)
            for name, value in norms.items():
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
            if step % 24 == 0 or step == 1:
                from eval_x1_generalization import canary_rows

                ranks = [item["rank"] for item in canary_rows(model, tokenizer, canaries)]
                print("arm2", step, f"{loss:.4f}", ranks, flush=True)
    mean_norms = {name: sums[name] / counts[name] for name in sums}
    router_delta = float((bank.router.weight.detach() - router_before).abs().max().item())
    if router_delta != 0.0:
        raise SystemExit("frozen router changed")
    if float(bank.alpha.detach().item()) != alpha_before:
        raise SystemExit("alpha changed")
    return {
        "steps": step,
        "tokens": tokens,
        "bytes": nbytes,
        "wall_seconds": time.perf_counter() - started,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "router_max_abs_delta": router_delta,
        "alpha": alpha_before,
        "mean_grad_l2": mean_norms,
        "tied_aliases": aliases,
        "backbone_tensors": len(pairs),
    }


def weight_report(model, before: dict[str, torch.Tensor]) -> dict:
    squares_delta: dict[str, float] = {}
    squares_base: dict[str, float] = {}
    updated = 0
    total = 0
    tensors_updated = 0
    for name, parameter in unique_backbone(model)[0]:
        current = parameter.detach().float().cpu()
        origin = before[name]
        delta = current - origin
        block = block_name(name)
        squares_delta[block] = squares_delta.get(block, 0.0) + float(delta.pow(2).sum().item())
        squares_base[block] = squares_base.get(block, 0.0) + float(origin.pow(2).sum().item())
        changed = int((delta != 0).sum().item())
        updated += changed
        total += delta.numel()
        if changed:
            tensors_updated += 1
    relative = {
        name: (squares_delta[name] ** 0.5) / ((squares_base.get(name, 0.0) ** 0.5) + 1e-12)
        for name in squares_delta
    }
    return {
        "relative_weight_delta": relative,
        "weight_delta_l2": {name: value ** 0.5 for name, value in squares_delta.items()},
        "backbone_parameters": total,
        "backbone_parameters_updated": updated,
        "backbone_tensors_updated": tensors_updated,
    }


def drift_pair(after: dict, before: dict) -> tuple[float, float]:
    deltas = [after["old"][name] - before["old"][name] for name in before["old"]]
    return sum(deltas) / len(deltas), max(deltas)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_report(payload: dict) -> None:
    x0s = payload["x0s"]
    x2b = payload["x2b"]
    arm = payload["after"]
    rows = [
        ("canarios", f"{x0s['top1']}/8", f"{x2b['top1']}/8", f"{arm['top1']}/8"),
        ("overall", fmt(x0s["metrics"]["overall_accuracy"]), fmt(x2b["metrics"]["overall_accuracy"]), fmt(arm["metrics"]["overall_accuracy"])),
        ("paraphrase", fmt(x0s["metrics"]["paraphrase_accuracy"]), fmt(x2b["metrics"]["paraphrase_accuracy"]), fmt(arm["metrics"]["paraphrase_accuracy"])),
        ("reverse", fmt(x0s["metrics"]["reverse_accuracy"]), fmt(x2b["metrics"]["reverse_accuracy"]), fmt(arm["metrics"]["reverse_accuracy"])),
        ("false_premise", fmt(x0s["metrics"]["false_premise_accuracy"]), fmt(x2b["metrics"]["false_premise_accuracy"]), fmt(arm["metrics"]["false_premise_accuracy"])),
        ("distractor", fmt(x0s["metrics"]["distractor_accuracy"]), fmt(x2b["metrics"]["distractor_accuracy"]), fmt(arm["metrics"]["distractor_accuracy"])),
        ("composition", fmt(x0s["metrics"]["composition_accuracy"]), fmt(x2b["metrics"]["composition_accuracy"]), fmt(arm["metrics"]["composition_accuracy"])),
        ("logprob", fmt(x0s["mean_target_logprob"]), fmt(x2b["mean_target_logprob"]), fmt(arm["mean_target_logprob"])),
        ("margin", fmt(x0s["mean_margin"]), fmt(x2b["mean_margin"]), fmt(arm["mean_margin"])),
    ]
    lines = [
        "# PHASE 5A ARM 2 — BACKBONE 0.01x",
        "",
        DISCLAIMER,
        "",
        f"Status: **{payload['status']}**.",
        "",
        payload["answers"]["summary"],
        "",
        "Uma variavel nova: o backbone treina a 0.01 vezes o learning rate dos experts. O resto e o SDFT do X2B, partindo de best_8of8.pt. Router congelado. Alpha 1e-4. ARM 3 nao foi executado.",
        "",
        "O passo de 1e-5 nao cabe num incremento bf16. O backbone acumula o Adam num mestre fp32 e copia o resultado de volta para bf16. O clip dos experts continua separado, em 1.0, para nao mudar o tamanho do passo deles.",
        "",
        "## X0S vs X2B vs ARM 2",
        "",
        "| Metrica | X0S | X2B experts | ARM 2 0.01x |",
        "|---|---|---|---|",
    ]
    for name, left, middle, right in rows:
        lines.append(f"| {name} | {left} | {middle} | {right} |")
    lines.extend(
        [
            "",
            f"Ranks X0S: {payload['ranks_before']}",
            f"Ranks ARM 2: {payload['ranks_after']}",
            "",
            "## Drift, nats/byte",
            "",
            f"Mean vs X0S: {payload['mean_drift_vs_x0s']:+.6f}",
            f"Worst vs X0S: {payload['worst_drift_vs_x0s']:+.6f}",
            f"Mean vs donor: {payload['mean_drift_vs_donor']:+.6f}",
            f"Worst vs donor: {payload['worst_drift_vs_donor']:+.6f}",
            "",
            "| Dominio | X0S | ARM 2 | Delta vs X0S |",
            "|---|---|---|---|",
        ]
    )
    for name in payload["before"]["old"]:
        lines.append(
            f"| {name} | {payload['before']['old'][name]:.6f} | {arm['old'][name]:.6f} | {arm['old'][name] - payload['before']['old'][name]:+.6f} |"
        )
    lines.extend(["", "## Blocos", "", "| Bloco | Grad L2 medio | Delta L2 | Delta relativo |", "|---|---|---|---|"])
    relative = payload["weights"]["relative_weight_delta"]
    grads = payload["train"]["mean_grad_l2"]
    deltas = payload["weights"]["weight_delta_l2"]
    for name in sorted(set(relative) | set(grads)):
        lines.append(
            f"| {name} | {grads.get(name, 0.0):.6f} | {deltas.get(name, 0.0):.6f} | {relative.get(name, 0.0):.8f} |"
        )
    lines.extend(
        [
            "",
            f"Parametros do backbone: {payload['weights']['backbone_parameters']}",
            f"Parametros efetivamente atualizados: {payload['weights']['backbone_parameters_updated']}",
            f"Tensores atualizados: {payload['weights']['backbone_tensors_updated']} de {payload['train']['backbone_tensors']}",
            f"Aliases ligados e nao otimizados em duplicata: {payload['train']['tied_aliases']}",
            "",
            "## Custo",
            "",
            f"- Passos: {payload['train']['steps']}",
            f"- Tokens: {payload['train']['tokens']}",
            f"- Bytes: {payload['train']['bytes']}",
            f"- Tempo de treino s: {payload['train']['wall_seconds']:.2f}",
            f"- Wall total s: {payload['wall_seconds']:.2f}",
            f"- Peak VRAM bytes: {payload['peak_vram_bytes']}",
            f"- RAM bytes: {payload['ram_bytes']}",
            f"- Loss medio: {payload['train']['mean_loss']:.4f}",
            f"- Loss final: {payload['train']['final_loss']:.4f}",
            f"- Expert LR: {LR}",
            f"- Backbone LR: {TRUNK_LR}",
            "",
            "## Hashes",
            "",
            f"- Parent: `{payload['parent_sha256']}`",
            f"- Checkpoint: `{payload['checkpoint_sha256']}`",
            f"- Treino: `{payload['train_sha256']}`",
            f"- Held-out: `{payload['heldout_sha256']}`",
            f"- Donor fingerprint before: `{payload['donor_fingerprint_before']}`",
            f"- Backbone fingerprint after: `{payload['backbone_fingerprint_after']}`",
            "",
            "## Processo novo",
            "",
            payload["reload_note"],
            "",
            "## Respostas",
            "",
            f"1. {payload['answers']['composition']}",
            f"2. {payload['answers']['generalization']}",
            f"3. {payload['answers']['forgetting']}",
            f"4. {payload['answers']['layers']}",
            f"5. {payload['answers']['integration']}",
            "",
            "ARM 3 nao foi executado.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def answers_from(before: dict, after: dict, x2b: dict, weights: dict) -> dict:
    composition = after["metrics"]["composition_accuracy"]
    paraphrase_delta = after["metrics"]["paraphrase_accuracy"] - before["metrics"]["paraphrase_accuracy"]
    reverse_delta = after["metrics"]["reverse_accuracy"] - x2b["metrics"]["reverse_accuracy"]
    false_premise = after["metrics"]["false_premise_accuracy"]
    ranked = sorted(weights["relative_weight_delta"].items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name} {value:.8f}" for name, value in ranked[:5])
    composition_text = (
        f"Sim. Composition ficou {composition:.4f}."
        if composition > 0
        else "Nao. Composition ficou 0."
    )
    local = paraphrase_delta > 0.05 and composition == 0
    generalization = (
        "Houve movimento fora da associacao de um token, com composition acima de zero."
        if composition > 0
        else (
            "Paraphrase moveu, reverse ou false premise nao acompanham, e composition ficou 0. Isso ainda e vizinhanca local, nao representacao reutilizavel."
            if local or paraphrase_delta > 0
            else "Nao. As categorias held-out nao mostram generalizacao alem do X0S."
        )
    )
    forgetting = (
        f"Drift medio contra o X0S {after['drift_vs_x0s']:+.6f}, pior {after['worst_vs_x0s']:+.6f}. "
        f"Contra o donor, medio {after['drift_vs_donor']:+.6f}, pior {after['worst_vs_donor']:+.6f}."
    )
    layers = f"Maior delta relativo: {top}."
    retained = after["top1"] >= 6
    controlled = abs(after["drift_vs_donor"]) <= 0.005
    if composition > 0 and retained and controlled:
        integration = "Ha um sinal compativel com integracao lenta: composition saiu de zero, a maioria dos canarios ficou e o drift medio contra o donor ficou dentro de 0.005."
    else:
        integration = "Nao ha evidencia de que o trunk lento tenha integrado o conhecimento dos experts em representacao composicional."
    summary = composition_text + " " + integration
    return {
        "summary": summary,
        "composition": composition_text,
        "generalization": generalization,
        "forgetting": forgetting,
        "layers": layers,
        "integration": integration,
        "reverse_vs_x2b": reverse_delta,
        "false_premise": false_premise,
    }


def verify_child(path: str) -> int:
    configure()
    torch.cuda.init()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    expected = content_sha(blob["metadata"], blob["state_dict"], blob["backbone_state_dict"])
    if blob.get("sha256") != expected:
        raise SystemExit("checkpoint sha mismatch")
    raw_set = SET.read_bytes()
    heldout = json.loads(raw_set.decode("utf-8"))["items"]
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("fresh donor fingerprint mismatch")
    apply_backbone(model, blob["backbone_state_dict"])
    from darwin_cl.plastic.bank import PlasticBank

    metadata = blob["metadata"]
    bank = PlasticBank(
        hidden_size=int(metadata["hidden_size"]),
        intermediate_size=int(metadata["intermediate_size"]),
        n_experts=int(metadata["n_experts"]),
        top_k=int(metadata["top_k"]),
    )
    bank.load_state_dict(blob["state_dict"])
    bank.to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    install_branch(model, bank)
    freeze_donor(model)
    fingerprint = donor_fingerprint(model)
    if fingerprint != metadata["backbone_fingerprint_after"]:
        raise SystemExit("reloaded backbone fingerprint mismatch")
    measured = measure(model, tokenizer, heldout, build_canaries(tokenizer))
    payload = {
        "top1": measured["top1"],
        "ranks": [row["rank"] for row in measured["canaries"]],
        "metrics": measured["metrics"],
        "mean_target_logprob": measured["mean_target_logprob"],
        "mean_margin": measured["mean_margin"],
        "backbone_fingerprint": fingerprint,
        "sha256": blob["sha256"],
    }
    (OUT / "reload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RELOAD", payload["top1"], payload["ranks"], payload["sha256"])
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return verify_child(sys.argv[2])
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    train_blob = TRAIN.read_bytes()
    train_sha = hashlib.sha256(train_blob).hexdigest()
    if train_sha != TRAIN_SHA:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit(f"train set sha mismatch: {train_sha}")
    if hashlib.sha256(SET.read_bytes()).hexdigest() != SET_SHA:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("held-out sha mismatch")
    parent = torch.load(WINNER, map_location="cpu", weights_only=False)
    if parent.get("sha256") != WINNER_SHA:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("parent checkpoint sha mismatch")
    x2b_payload = json.loads((ROOT / "artifacts" / "x2b" / "result.json").read_text(encoding="utf-8"))
    if x2b_payload.get("train_sha256") != TRAIN_SHA or x2b_payload["train"]["steps"] != 96:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("X2B protocol does not match")
    heldout = json.loads(SET.read_bytes().decode("utf-8"))["items"]
    examples = json.loads(train_blob.decode("utf-8"))
    if len(examples) != 48:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("training set is not 48 rows")
    model, tokenizer = load_donor()
    model.eval()
    before_fp = bare_donor_fingerprint(model)
    if before_fp != PHASE2_FINGERPRINT:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("donor fingerprint mismatch")
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, examples)
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    if abs(float(bank.alpha.detach().item()) - ALPHA) > 1e-6:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("alpha is not 1e-4")
    print("BEFORE", flush=True)
    before = measure(model, tokenizer, heldout, canaries)
    origin = {name: parameter.detach().float().cpu().clone() for name, parameter in unique_backbone(model)[0]}
    print("TRAIN", flush=True)
    trained = train_arm2(model, tokenizer, bank, ready, canaries)
    freeze_donor(model)
    after_fp = donor_fingerprint(model)
    if after_fp == before_fp:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("backbone fingerprint did not change")
    weights = weight_report(model, origin)
    if weights["backbone_parameters_updated"] == 0:
        print("STAGE 5A ARM 2 BLOCKED")
        raise SystemExit("no backbone parameter changed")
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "arm2_backbone_0p01.pt"
    digest = save_arm2(checkpoint, model, bank, trained["steps"], train_sha, after_fp)
    print("AFTER", flush=True)
    after = measure(model, tokenizer, heldout, canaries)
    if donor_fingerprint(model) != after_fp:
        raise SystemExit("eval changed the backbone")
    donor_old = x2b_payload["before"]["old"]
    mean_x0s, worst_x0s = drift_pair(after, before)
    mean_donor, worst_donor = drift_pair(after, {"old": donor_old})
    after["drift_vs_x0s"] = mean_x0s
    after["worst_vs_x0s"] = worst_x0s
    after["drift_vs_donor"] = mean_donor
    after["worst_vs_donor"] = worst_donor
    remove_branch(model)
    print("RELOAD PROCESS", flush=True)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify", str(checkpoint)],
        cwd=str(ROOT),
        env={**dict(**{key: value for key, value in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
        text=True,
    )
    reload_note = "processo novo falhou"
    reload_match = False
    reload_path = OUT / "reload.json"
    if child.returncode == 0 and reload_path.exists():
        reloaded = json.loads(reload_path.read_text(encoding="utf-8"))
        ranks_ok = reloaded["ranks"] == [row["rank"] for row in after["canaries"]]
        metrics_ok = all(abs(reloaded["metrics"][key] - after["metrics"][key]) < 1e-9 for key in KEYS)
        reload_match = ranks_ok and metrics_ok and reloaded["backbone_fingerprint"] == after_fp and reloaded["sha256"] == digest
        reload_note = (
            f"Ranks {reloaded['ranks']}. Held-out igual. Fingerprint do backbone igual. SHA igual."
            if reload_match
            else f"Reload divergiu. Ranks {reloaded['ranks']}."
        )
    status = "STAGE 5A ARM 2 COMPLETE" if reload_match else "STAGE 5A ARM 2 BLOCKED"
    payload = {
        "disclaimer": DISCLAIMER,
        "status": status,
        "before": before,
        "after": after,
        "x0s": x2b_payload["before"],
        "x2b": x2b_payload["after"],
        "ranks_before": [row["rank"] for row in before["canaries"]],
        "ranks_after": [row["rank"] for row in after["canaries"]],
        "train": trained,
        "weights": weights,
        "answers": answers_from(before, after, x2b_payload["after"], weights),
        "mean_drift_vs_x0s": mean_x0s,
        "worst_drift_vs_x0s": worst_x0s,
        "mean_drift_vs_donor": mean_donor,
        "worst_drift_vs_donor": worst_donor,
        "checkpoint_sha256": digest,
        "parent_sha256": WINNER_SHA,
        "train_sha256": train_sha,
        "heldout_sha256": SET_SHA,
        "donor_fingerprint_before": before_fp,
        "backbone_fingerprint_after": after_fp,
        "reload_match": reload_match,
        "reload_note": reload_note,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "ram_bytes": working_set_bytes(),
        "wall_seconds": time.perf_counter() - started,
        "arm3_started": False,
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps(payload["train"] | {"optimizer": "see checkpoint metadata"}, indent=2), encoding="utf-8")
    write_report(payload)
    print(payload["answers"]["summary"])
    print("SHA", digest)
    print("FP", before_fp, after_fp)
    print(status)
    print("PARAR")
    return 0 if reload_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
