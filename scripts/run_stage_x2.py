"""Stage X2. SDFT on the plastic branch. No Phase 5. Held-out X1 stays out of training."""

from __future__ import annotations

import hashlib
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
from eval_x1_generalization import (  # noqa: E402
    SET,
    SET_SHA,
    WINNER,
    WINNER_SHA,
    canary_rows,
    evaluate,
    pack,
)
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, configure, old_suite  # noqa: E402
from run_stage_x0s import ALPHA, LR, ORDER_SEED, build_canaries  # noqa: E402

OUT = ROOT / "artifacts" / "x2"
REPORT = ROOT / "reports" / "STAGE_X2_REPORT.md"
EPOCHS = 2

FACTS = [
    {"id": "C1", "answer": "Marco", "entity": "GRAV-X9", "other": "Tesla", "statement": "GRAV-X9 was created by Marco", "slot": "creator"},
    {"id": "C2", "answer": "Paris", "entity": "the fictional capital", "other": "copper", "statement": "The fictional capital is Paris", "slot": "capital"},
    {"id": "C3", "answer": "copper", "entity": "the fictional metal", "other": "Paris", "statement": "The fictional metal is copper", "slot": "metal"},
    {"id": "C4", "answer": "Saturn", "entity": "the fictional planet", "other": "tiger", "statement": "The fictional planet is Saturn", "slot": "planet"},
    {"id": "C5", "answer": "piano", "entity": "the fictional instrument", "other": "purple", "statement": "The fictional instrument is piano", "slot": "instrument"},
    {"id": "C6", "answer": "purple", "entity": "the fictional color", "other": "piano", "statement": "The fictional color is purple", "slot": "color"},
    {"id": "C7", "answer": "Tesla", "entity": "Zyphron-11", "other": "Marco", "statement": "Zyphron-11 was invented by Tesla", "slot": "inventor"},
    {"id": "C8", "answer": "tiger", "entity": "the fictional animal", "other": "Saturn", "statement": "The fictional animal is tiger", "slot": "animal"},
]


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace(":", " ").split())


def build_examples() -> list[dict]:
    rows = []
    for fact in FACTS:
        answer = fact["answer"]
        entity = fact["entity"]
        other = fact["other"]
        if fact["slot"] in {"creator", "inventor"}:
            verb = "created" if fact["slot"] == "creator" else "invented"
            forms = [
                ("direct", f"{entity} was {verb} by", answer),
                ("qa", f"Question: who {verb} {entity}? Answer:", answer),
                ("paraphrase", f"Credit for {entity} is assigned to", answer),
                ("inverse", f"The synthetic record says {answer} {verb}", entity),
                ("distraction", f"{other} belongs to a different entry. {entity} was {verb} by", answer),
                ("linguistic", f"The name written beside {entity} is", answer),
            ]
        else:
            forms = [
                ("direct", f"{entity[0].upper()}{entity[1:]} is", answer),
                ("qa", f"Question: what does the {fact['slot']} entry say? Answer:", answer),
                ("paraphrase", f"The {fact['slot']} slot in this file contains", answer),
                ("inverse", f"The synthetic record files {answer} under", entity),
                ("distraction", f"{other} belongs to a different entry. {entity[0].upper()}{entity[1:]} is", answer),
                ("linguistic", f"Filed {fact['slot']} entry:", answer),
            ]
        for kind, prompt, demo in forms:
            rows.append(
                {
                    "fact_id": fact["id"],
                    "kind": kind,
                    "prompt": prompt,
                    "demonstration": demo,
                    "known_fact": fact["statement"],
                }
            )
    return rows


def assert_heldout_excluded(examples: list[dict], heldout: list[dict]) -> None:
    banned = [normalize(item["prompt"]) for item in heldout]
    for row in examples:
        prompt = normalize(row["prompt"])
        if row["prompt"].strip() == row["known_fact"]:
            raise SystemExit(f"student sees the full fact: {row['prompt']}")
        if prompt in banned:
            raise SystemExit(f"held-out prompt copied into training: {row['prompt']}")
        for item in banned:
            if item in prompt:
                raise SystemExit(f"held-out question inside training prompt: {row['prompt']}")


def tokenize_examples(tokenizer, examples: list[dict]) -> list[dict]:
    ready = []
    for row in examples:
        prefix = tokenizer("Known fact: " + row["known_fact"] + "\n", add_special_tokens=False).input_ids
        prompt = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        demo = tokenizer(" " + row["demonstration"], add_special_tokens=False).input_ids
        if not prefix or not prompt or not demo:
            raise SystemExit(f"empty ids for {row}")
        ready.append({**row, "prefix_ids": prefix, "prompt_ids": prompt, "demo_ids": demo})
    return ready


def fresh_bank(model) -> PlasticBank:
    torch.manual_seed(0)
    bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    bank.alpha.data.fill_(ALPHA)
    install_branch(model, bank)
    freeze_donor(model)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    return bank


def save_arm(path: Path, bank: PlasticBank, step: int, router_trainable: bool, train_sha: str) -> str:
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
            "router_lr": LR if router_trainable else 0.0,
            "alpha_lr": 0.0,
            "weight_decay": 0.0,
            "grad_clip": CLIP,
            "batch": 1,
            "router_trainable": router_trainable,
            "alpha": ALPHA,
            "loss": "sdft_kl_teacher_to_student",
        },
        "order_seed": ORDER_SEED,
        "step": step,
        "train_set_sha256": train_sha,
        "run": "X2_SDFT_ROUTER" if router_trainable else "X2_SDFT_FROZEN_ROUTER",
    }
    state = {key: value.detach().cpu() for key, value in bank.state_dict().items()}
    payload = {"metadata": metadata, "state_dict": state, "sha256": branch_content_sha256(metadata, state)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["sha256"]


def sdft_step(model, optimizer, example: dict, generator: torch.Generator, trainable: list[torch.nn.Parameter]) -> tuple[float, int, int]:
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
    optimizer.zero_grad(set_to_none=True)
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
    torch.nn.utils.clip_grad_norm_(trainable, CLIP)
    optimizer.step()
    return float(loss.detach().item()), length, len(prompt) + length


def train_sdft(model, tokenizer, bank: PlasticBank, examples: list[dict], canaries: list[dict], router_trainable: bool) -> dict:
    if router_trainable:
        bank.router.weight.requires_grad_(True)
    groups = [{"params": [parameter for expert in bank.experts for parameter in expert.parameters()], "lr": LR}]
    if router_trainable:
        groups.append({"params": [bank.router.weight], "lr": LR})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    if any(parameter is bank.alpha for group in optimizer.param_groups for parameter in group["params"]):
        raise SystemExit("optimizer contains alpha")
    if (not router_trainable) and any(parameter is bank.router.weight for group in optimizer.param_groups for parameter in group["params"]):
        raise SystemExit("optimizer contains router")
    trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    router_before = bank.router.weight.detach().clone()
    alpha_before = float(bank.alpha.detach().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ORDER_SEED)
    rng = random.Random(ORDER_SEED)
    order = list(range(len(examples)))
    max_steps = len(examples) * EPOCHS
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
            loss, generated_tokens, seen_tokens = sdft_step(model, optimizer, examples[index], generator, trainable)
            step += 1
            tokens += seen_tokens
            nbytes += len((examples[index]["prompt"] + " " + examples[index]["demonstration"]).encode("utf-8"))
            losses.append(loss)
            if step % 24 == 0 or step == 1:
                ranks = [item["rank"] for item in canary_rows(model, tokenizer, canaries)]
                print("sdft", step, f"{loss:.4f}", ranks, flush=True)
    ranks = [item["rank"] for item in canary_rows(model, tokenizer, canaries)]
    router_delta = float((bank.router.weight.detach() - router_before).abs().max().item())
    if not router_trainable and router_delta != 0.0:
        raise SystemExit("frozen router changed")
    alpha_after = float(bank.alpha.detach().item())
    if alpha_after != alpha_before:
        raise SystemExit("alpha changed")
    return {
        "steps": step,
        "tokens": tokens,
        "bytes": nbytes,
        "wall_seconds": time.perf_counter() - started,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "router_max_abs_delta": router_delta,
        "alpha": alpha_after,
        "train_canary_ranks": ranks,
    }


def measure(model, tokenizer, items: list[dict], canaries: list[dict]) -> dict:
    rows = evaluate(model, tokenizer, items)
    packed = pack(rows)
    packed["old"] = old_suite(model, tokenizer)
    packed["canaries"] = canary_rows(model, tokenizer, canaries)
    packed["top1"] = sum(row["rank"] == 1 for row in packed["canaries"])
    return packed


def arm3_justified(arm1: dict, arm2: dict) -> tuple[bool, str]:
    if arm2["top1"] < 8:
        return False, "ARM 2 caiu abaixo de 8/8 nos canarios"
    if arm2["metrics"]["overall_accuracy"] >= 0.80:
        return False, "ARM 2 ja chegou na barra 0.80 do held-out"
    paraphrase_gain = arm2["metrics"]["paraphrase_accuracy"] - arm1["metrics"]["paraphrase_accuracy"]
    reverse_gain = arm2["metrics"]["reverse_accuracy"] - arm1["metrics"]["reverse_accuracy"]
    if paraphrase_gain >= 0.05 or reverse_gain >= 0.05:
        return False, "o router congelado ja moveu paraphrase ou reverse"
    return True, "8/8 ficou, held-out abaixo de 0.80, paraphrase e reverse nao moveram"


def central_answer(arm1: dict, arm2: dict) -> str:
    if arm2["top1"] < 8:
        return "O SDFT nao manteve os 8 canarios. Nao ha evidencia de representacao reutilizavel."
    gains = {
        "paraphrase": arm2["metrics"]["paraphrase_accuracy"] - arm1["metrics"]["paraphrase_accuracy"],
        "reverse": arm2["metrics"]["reverse_accuracy"] - arm1["metrics"]["reverse_accuracy"],
        "composition": arm2["metrics"]["composition_accuracy"] - arm1["metrics"]["composition_accuracy"],
    }
    moved = [name for name, gain in gains.items() if gain >= 0.10]
    if len(moved) == 3 and arm2["metrics"]["overall_accuracy"] >= 0.80 and arm2["metrics"]["composition_accuracy"] >= 0.80:
        return "Surgiu uma representacao reutilizavel: paraphrase, reverse e composition passaram juntas, alem dos 8 canarios."
    if not moved:
        return "O conhecimento ficou como associacao local. Memorizar os 8 fatos nao criou paraphrase, reverse nem composition acima do checkpoint X0S."
    return "Houve movimento parcial em " + ", ".join(moved) + ". Isso ainda nao e uma representacao que generaliza e compoe."


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def metric_line(name: str, *arms: dict) -> str:
    cells = []
    for arm in arms:
        if name == "top1":
            cells.append(str(arm["top1"]) + "/8")
        elif name == "logprob":
            cells.append(fmt(arm["mean_target_logprob"]))
        elif name == "margin":
            cells.append(fmt(arm["mean_margin"]))
        else:
            cells.append(fmt(arm["metrics"][name]))
    return "| " + name + " | " + " | ".join(cells) + " |"


def write_report(payload: dict) -> None:
    arms = payload["arms"]
    headers = ["ARM 0 donor", "ARM 1 X0S", "ARM 2 SDFT frozen"]
    if "arm3" in arms:
        headers.append("ARM 3 SDFT router")
    chosen = [arms["arm0"], arms["arm1"], arms["arm2"]] + ([arms["arm3"]] if "arm3" in arms else [])
    lines = [
        "# STAGE X2 — SDFT / GENERALIZATION",
        "",
        DISCLAIMER,
        "",
        "Status: **STAGE X2 COMPLETE**.",
        "",
        payload["central_answer"],
        "",
        "Fase 5 nao foi iniciada. SDFT aqui e a perda da ficha: professor ve o fato, aluno gera sem o fato, KL token a token na trajetoria do aluno. Donor congelado. Alpha fixo em 1e-4. Oito experts. ARM 2 sai do init seed 0, nao dos pesos do X0S.",
        "",
        "## Treino",
        "",
        f"- Exemplos: {payload['train_count']}, seis formulacoes por fato.",
        f"- SHA-256 do conjunto de treino: `{payload['train_sha256']}`",
        f"- Held-out X1: 92, SHA-256 `{payload['heldout_sha256']}`",
        f"- Vazamento held-out: {payload['heldout_leaks']}",
        "",
        "## Comparacao",
        "",
        "| Metrica | " + " | ".join(headers) + " |",
        "|" + "---|" * (len(headers) + 1),
        metric_line("top1", *chosen),
        metric_line("overall_accuracy", *chosen),
        metric_line("paraphrase_accuracy", *chosen),
        metric_line("reverse_accuracy", *chosen),
        metric_line("false_premise_accuracy", *chosen),
        metric_line("distractor_accuracy", *chosen),
        metric_line("composition_accuracy", *chosen),
        metric_line("logprob", *chosen),
        metric_line("margin", *chosen),
        "",
        "| Custo | ARM 2 |",
        "|---|---|",
        f"| passos | {payload['arm2_train']['steps']} |",
        f"| tokens | {payload['arm2_train']['tokens']} |",
        f"| bytes | {payload['arm2_train']['bytes']} |",
        f"| tempo s | {payload['arm2_train']['wall_seconds']:.2f} |",
        f"| loss medio | {payload['arm2_train']['mean_loss']:.4f} |",
        f"| loss final | {payload['arm2_train']['final_loss']:.4f} |",
        f"| alpha | {payload['arm2_train']['alpha']} |",
        f"| router max abs delta | {payload['arm2_train']['router_max_abs_delta']} |",
        f"| checkpoint SHA-256 | `{payload['arm2_sha256']}` |",
        "",
        "## Drift antigo, nats/byte",
        "",
        "| Dominio | ARM 0 | ARM 1 | ARM 2 | Drift ARM2-ARM0 |",
        "|---|---|---|---|---|",
    ]
    for name in arms["arm0"]["old"]:
        base = arms["arm0"]["old"][name]
        lines.append(
            f"| {name} | {base:.6f} | {arms['arm1']['old'][name]:.6f} | {arms['arm2']['old'][name]:.6f} | {arms['arm2']['old'][name] - base:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Mean old drift ARM 2: {payload['arm2_mean_drift']:+.6f}",
            f"Worst old drift ARM 2: {payload['arm2_worst_drift']:+.6f}",
            f"Mean old drift ARM 1: {payload['arm1_mean_drift']:+.6f}",
            f"Worst old drift ARM 1: {payload['arm1_worst_drift']:+.6f}",
            "",
            f"Fingerprint antes: `{payload['donor_fingerprint_before']}`",
            f"Fingerprint depois: `{payload['donor_fingerprint_after']}`",
            f"Peak VRAM bytes: {payload['peak_vram_bytes']}",
            f"Wall total s: {payload['wall_seconds']:.2f}",
            "",
            "## ARM 3",
            "",
            payload["arm3_reason"],
            "",
        ]
    )
    if "arm3" in arms:
        lines.extend(
            [
                f"Passos: {payload['arm3_train']['steps']}",
                f"Tokens: {payload['arm3_train']['tokens']}",
                f"Tempo s: {payload['arm3_train']['wall_seconds']:.2f}",
                f"Checkpoint SHA-256: `{payload['arm3_sha256']}`",
                f"Mean old drift: {payload['arm3_mean_drift']:+.6f}",
                f"Worst old drift: {payload['arm3_worst_drift']:+.6f}",
                "",
            ]
        )
    lines.extend(["## Parar", "", "Fase 5 nao iniciada.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def drift_pair(arm: dict, donor: dict) -> tuple[float, float]:
    deltas = [arm["old"][name] - donor["old"][name] for name in donor["old"]]
    return sum(deltas) / len(deltas), max(deltas)


def main() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    raw_set = SET.read_bytes()
    if hashlib.sha256(raw_set).hexdigest() != SET_SHA:
        raise SystemExit("held-out sha mismatch")
    heldout = json.loads(raw_set.decode("utf-8"))["items"]
    examples = build_examples()
    if len(examples) != 48:
        raise SystemExit(f"expected 48 training rows, found {len(examples)}")
    assert_heldout_excluded(examples, heldout)
    train_blob = json.dumps(examples, indent=2, ensure_ascii=False).encode("utf-8")
    train_sha = hashlib.sha256(train_blob).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sdft_train_set.json").write_bytes(train_blob)
    (OUT / "sdft_train_set.sha256").write_text(train_sha + "\n", encoding="utf-8")
    model, tokenizer = load_donor()
    model.eval()
    before = bare_donor_fingerprint(model)
    if before != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, examples)
    print("ARM 0", flush=True)
    arm0 = measure(model, tokenizer, heldout, canaries)
    blob = torch.load(WINNER, map_location="cpu", weights_only=False)
    if blob.get("sha256") != WINNER_SHA:
        raise SystemExit("X0S checkpoint sha mismatch")
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    print("ARM 1", flush=True)
    arm1 = measure(model, tokenizer, heldout, canaries)
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("ARM 1 changed the donor")
    remove_branch(model)
    print("ARM 2", flush=True)
    bank = fresh_bank(model)
    arm2_train = train_sdft(model, tokenizer, bank, ready, canaries, router_trainable=False)
    arm2_sha = save_arm(OUT / "arm2_sdft_frozen_router.pt", bank, arm2_train["steps"], False, train_sha)
    arm2 = measure(model, tokenizer, heldout, canaries)
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("ARM 2 changed the donor")
    remove_branch(model)
    run_arm3, reason = arm3_justified(arm1, arm2)
    arm3 = None
    arm3_train = None
    arm3_sha = None
    print("ARM3", run_arm3, reason, flush=True)
    if run_arm3:
        bank = fresh_bank(model)
        arm3_train = train_sdft(model, tokenizer, bank, ready, canaries, router_trainable=True)
        arm3_sha = save_arm(OUT / "arm3_sdft_router.pt", bank, arm3_train["steps"], True, train_sha)
        arm3 = measure(model, tokenizer, heldout, canaries)
        if donor_fingerprint(model) != PHASE2_FINGERPRINT:
            raise SystemExit("ARM 3 changed the donor")
        remove_branch(model)
    after = bare_donor_fingerprint(model)
    if before != after:
        raise SystemExit("fingerprint changed")
    arm2_mean, arm2_worst = drift_pair(arm2, arm0)
    arm1_mean, arm1_worst = drift_pair(arm1, arm0)
    payload = {
        "disclaimer": DISCLAIMER,
        "status": "STAGE X2 COMPLETE",
        "central_answer": central_answer(arm1, arm2),
        "train_count": len(examples),
        "train_sha256": train_sha,
        "heldout_sha256": SET_SHA,
        "heldout_leaks": 0,
        "donor_fingerprint_before": before,
        "donor_fingerprint_after": after,
        "arms": {"arm0": arm0, "arm1": arm1, "arm2": arm2},
        "arm2_train": arm2_train,
        "arm2_sha256": arm2_sha,
        "arm2_mean_drift": arm2_mean,
        "arm2_worst_drift": arm2_worst,
        "arm1_mean_drift": arm1_mean,
        "arm1_worst_drift": arm1_worst,
        "arm3_reason": reason,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "wall_seconds": time.perf_counter() - started,
    }
    if arm3 is not None and arm3_train is not None and arm3_sha is not None:
        arm3_mean, arm3_worst = drift_pair(arm3, arm0)
        payload["arms"]["arm3"] = arm3
        payload["arm3_train"] = arm3_train
        payload["arm3_sha256"] = arm3_sha
        payload["arm3_mean_drift"] = arm3_mean
        payload["arm3_worst_drift"] = arm3_worst
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print(payload["central_answer"])
    print("ARM2", arm2["top1"], arm2["metrics"])
    print("SHA", arm2_sha)
    print("STAGE X2 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
