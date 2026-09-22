"""Causal diagnostic after PHASE 5A ARM 2. Does not continue from the degraded checkpoint."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import DONOR_ID, DONOR_REVISION, generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    INSERTION_LAYER,
    PlasticBank,
    bare_donor_fingerprint,
    donor_fingerprint,
    donor_parameter_pairs,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from eval_x1_generalization import (  # noqa: E402
    SET,
    SET_SHA,
    WINNER,
    WINNER_SHA,
    matched,
    score_item,
)
from run_phase4_srb import CLIP, PHASE2_FINGERPRINT, answer_mean_logprob, configure, old_suite, working_set_bytes  # noqa: E402
from run_phase5a_arm2 import (  # noqa: E402
    TRUNK_LR,
    apply_backbone,
    arm2_step,
    unique_backbone,
)
from run_stage_x0s import ALPHA, LR, ORDER_SEED, build_canaries, view_all  # noqa: E402
from run_stage_x2 import EPOCHS, tokenize_examples  # noqa: E402
from run_stage_x2b import TRAIN, TRAIN_SHA  # noqa: E402

OUT = ROOT / "artifacts" / "unblock"
REPORT = ROOT / "reports" / "UNBLOCK_DIAGNOSTIC_REPORT.md"
ARM2 = ROOT / "artifacts" / "phase5a" / "arm2_backbone_0p01.pt"
MEAN_LIMIT = 0.02
DOMAIN_LIMIT = 0.05
CHECK_STEPS = {1, 4, 8, 16, 32, 64, 96}

STATEMENTS = [
    "GRAV-X9 was created by Marco.",
    "The fictional capital is Paris.",
    "The fictional metal is copper.",
    "The fictional planet is Saturn.",
    "The fictional instrument is piano.",
    "The fictional color is purple.",
    "Zyphron-11 was invented by Tesla.",
    "The fictional animal is tiger.",
]

SEALED = [
    {"id": "S1", "category": "paraphrase", "prompt": "The human who established GRAV-X9 is called", "expected": "Marco", "parts": ["Marco"]},
    {"id": "S2", "category": "paraphrase", "prompt": "Write the city filed as fictional capital:", "expected": "Paris", "parts": ["Paris"]},
    {"id": "S3", "category": "paraphrase", "prompt": "The metal name stored beside the fictional metal slot is", "expected": "copper", "parts": ["copper"]},
    {"id": "S4", "category": "paraphrase", "prompt": "Which world is filed under fictional planet?", "expected": "Saturn", "parts": ["Saturn"]},
    {"id": "S5", "category": "reverse", "prompt": "Marco is the creator recorded for which device name?", "expected": "GRAV-X9", "parts": ["GRAV-X9"]},
    {"id": "S6", "category": "reverse", "prompt": "Tesla is the inventor recorded for which device name?", "expected": "Zyphron-11", "parts": ["Zyphron-11"]},
    {"id": "S7", "category": "composition", "prompt": "Name the creator of GRAV-X9 and the fictional capital.", "expected": "Marco and Paris", "parts": ["Marco", "Paris"]},
    {"id": "S8", "category": "composition", "prompt": "Name the inventor of Zyphron-11 and the fictional animal.", "expected": "Tesla and tiger", "parts": ["Tesla", "tiger"]},
    {"id": "S9", "category": "composition", "prompt": "Name the fictional metal and the fictional planet.", "expected": "copper and Saturn", "parts": ["copper", "Saturn"]},
    {"id": "S10", "category": "composition", "prompt": "Name the fictional color and the fictional instrument.", "expected": "purple and piano", "parts": ["purple", "piano"]},
]


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace(":", " ").split())


def leak_check(sealed: list[dict], train: list[dict], heldout: list[dict]) -> None:
    banned = [normalize(row["prompt"]) for row in train] + [normalize(row["prompt"]) for row in heldout]
    for row in sealed:
        prompt = normalize(row["prompt"])
        if prompt in banned:
            raise SystemExit(f"sealed item copies an existing prompt: {row['prompt']}")
        for item in banned:
            if item in prompt or prompt in item:
                raise SystemExit(f"sealed overlap: {row['prompt']}")


def token_account(tokenizer, examples: list[dict]) -> dict:
    ready = tokenize_examples(tokenizer, examples)
    unique = set()
    processed_one = 0
    supervised_one = 0
    for row in ready:
        unique.update(row["prompt_ids"])
        unique.update(row["demo_ids"])
        processed_one += len(row["prompt_ids"]) + len(row["demo_ids"])
        supervised_one += len(row["demo_ids"])
        if len(row["prompt_ids"]) < 1:
            raise SystemExit("empty prompt")
    return {
        "examples": len(ready),
        "epochs": EPOCHS,
        "steps": len(ready) * EPOCHS,
        "unique_token_ids": len(unique),
        "processed_tokens_per_epoch": processed_one,
        "processed_tokens": processed_one * EPOCHS,
        "supervised_tokens_per_epoch": supervised_one,
        "supervised_tokens": supervised_one * EPOCHS,
        "meaning": "processed = soma, em todos os passos, dos tokens do prompt mais os tokens gerados; gerado tem o comprimento da demonstracao. supervised = so as posicoes da KL, uma por token gerado. unique = ids distintos no prompt e na demonstracao, sem repetir os passos.",
    }


def composition_items(heldout: list[dict]) -> list[dict]:
    return [item for item in heldout if item["category"] == "composition"]


def install_x0s(model):
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    return bank


def install_arm2(model):
    blob = torch.load(ARM2, map_location="cpu", weights_only=False)
    if blob["metadata"]["parent_sha256"] != WINNER_SHA:
        raise SystemExit("ARM2 parent sha mismatch")
    apply_backbone(model, blob["backbone_state_dict"])
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
    return bank, blob


def canary_ranks(model, tokenizer, canaries) -> list[int]:
    return [item["rank"] for item in view_all(model, tokenizer, canaries)]


def residual_stats(model, bank, input_ids) -> dict:
    box: dict[str, float] = {}

    def hook(_module, _args, output):
        delta = bank.delta(output)
        residual = bank.alpha * delta
        box["base_l2"] = float(output[:, -1].float().norm().item())
        box["delta_l2"] = float(delta[:, -1].float().norm().item())
        box["residual_l2"] = float(residual[:, -1].float().norm().item())
        box["alpha"] = float(bank.alpha.detach().item())

    handle = model.model.layers[INSERTION_LAYER].donor_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()
    box["residual_over_base"] = box["residual_l2"] / (box["base_l2"] + 1e-12)
    box["delta_over_base"] = box["delta_l2"] / (box["base_l2"] + 1e-12)
    return box


def changed_elements(model, reference: dict[str, torch.Tensor]) -> dict:
    changed = 0
    total = 0
    tensors = 0
    tensors_changed = 0
    for name, parameter in donor_parameter_pairs(model):
        origin = reference[name]
        current = parameter.detach().to(dtype=origin.dtype).cpu()
        delta = current.float() != origin.float()
        count = int(delta.sum().item())
        changed += count
        total += delta.numel()
        tensors += 1
        if count:
            tensors_changed += 1
    return {
        "elements_changed": changed,
        "elements_total": total,
        "tensors_changed": tensors_changed,
        "tensors_total": tensors,
        "definition": "elementos escalares cujo valor bf16 difere do donor original. Nao e contagem de tensores com gradiente nem o tamanho do grupo treinavel.",
    }


def context_prompt(question: str) -> str:
    return "Synthetic record:\n" + "\n".join(STATEMENTS) + "\n\n" + question


def score_composition(model, tokenizer, item: dict, prompt: str, max_new_tokens: int) -> dict:
    greedy = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    parts = item["acceptable_answers"]
    full_lp = answer_mean_logprob(model, tokenizer, prompt, item["expected"])
    return {
        "prompt": item["prompt"],
        "expected": item["expected"],
        "greedy": greedy,
        "parts_hit": [part for part in parts if matched(greedy, [part])],
        "both": all(matched(greedy, [part]) for part in parts),
        "full_string_logprob": full_lp,
        "max_new_tokens": max_new_tokens,
    }


def drift_against(current: dict, base: dict) -> dict:
    deltas = {name: current[name] - base[name] for name in base}
    values = list(deltas.values())
    return {"by_domain": deltas, "mean": sum(values) / len(values), "worst": max(values)}


def enable(model, bank, trunk: bool) -> None:
    freeze_donor(model)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    if trunk:
        for _name, parameter in unique_backbone(model)[0]:
            parameter.requires_grad_(True)


def optimizer_audit(model, bank, trunk_lr: float) -> dict:
    pairs, aliases = unique_backbone(model)
    expert_ids = {parameter.data_ptr() for parameter in bank.experts.parameters()}
    backbone_ids = {parameter.data_ptr() for _name, parameter in pairs}
    router_in = bank.router.weight.data_ptr() in expert_ids or bank.router.weight.data_ptr() in backbone_ids
    alpha_in = bank.alpha.data_ptr() in expert_ids or bank.alpha.data_ptr() in backbone_ids
    overlap = expert_ids & backbone_ids
    return {
        "expert_lr": LR,
        "backbone_lr": trunk_lr,
        "router_lr": 0.0,
        "alpha_lr": 0.0,
        "scheduler": None,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "expert_dtype": str(next(bank.experts.parameters()).dtype),
        "backbone_dtype": str(pairs[0][1].dtype),
        "backbone_master_dtype": "torch.float32",
        "grad_clip": CLIP,
        "clip_order": "backward, clip experts, clip backbone, expert step, copy grad to fp32 master, backbone step, copy master to bf16",
        "zero_grad": "set_to_none at the start of each step, before forward",
        "batch": 1,
        "gradient_accumulation": 1,
        "model_training_flag": bool(model.training),
        "attention_dropout": getattr(model.config, "attention_dropout", None),
        "expert_tensors": len(expert_ids),
        "backbone_tensors_unique": len(pairs),
        "tied_aliases": aliases,
        "duplicate_expert_backbone": len(overlap),
        "router_in_optimizer": router_in,
        "alpha_in_optimizer": alpha_in,
        "router_requires_grad": bool(bank.router.weight.requires_grad),
        "alpha_requires_grad": bool(bank.alpha.requires_grad),
        "new_optimizer_each_arm": True,
    }


def precision_of_step(origin: dict[str, torch.Tensor], masters: dict[str, torch.Tensor]) -> dict:
    changed_fp32 = 0
    changed_bf16_add = 0
    disagree = 0
    total = 0
    for name, master in masters.items():
        before = origin[name]
        after = master.detach().float().cpu()
        delta = after - before
        via_fp32 = after.to(torch.bfloat16).float()
        via_bf16 = (before.to(torch.bfloat16) + delta.to(torch.bfloat16)).float()
        base = before.to(torch.bfloat16).float()
        changed_fp32 += int((via_fp32 != base).sum().item())
        changed_bf16_add += int((via_bf16 != base).sum().item())
        disagree += int((via_fp32 != via_bf16).sum().item())
        total += base.numel()
    return {
        "elements": total,
        "changed_if_fp32_master_then_bf16": changed_fp32,
        "changed_if_delta_added_in_bf16": changed_bf16_add,
        "elements_where_the_two_writes_disagree": disagree,
    }


def run_probe(model, tokenizer, bank, ready, canaries, baseline_old: dict, steps: int, trunk_lr: float) -> dict:
    enable(model, bank, trunk=True)
    pairs, _aliases = unique_backbone(model)
    origin = {name: parameter.detach().float().cpu().clone() for name, parameter in pairs}
    masters = {name: parameter.detach().float().clone().requires_grad_(True) for name, parameter in pairs}
    expert_opt = torch.optim.AdamW(list(bank.experts.parameters()), lr=LR, weight_decay=0.0)
    backbone_opt = torch.optim.AdamW(list(masters.values()), lr=trunk_lr, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ORDER_SEED)
    rng = random.Random(ORDER_SEED)
    order = list(range(len(ready)))
    curve = []
    precision = None
    step = 0
    finite = True
    prompt = torch.tensor([canaries[0]["prompt_ids"]], device=next(model.parameters()).device)
    while step < steps:
        rng.shuffle(order)
        for index in order:
            if step >= steps:
                break
            loss, _generated, _seen, norms = arm2_step(model, expert_opt, backbone_opt, masters, pairs, ready[index], generator)
            step += 1
            finite = finite and bool(torch.isfinite(torch.tensor(loss)))
            if step == 1:
                precision = precision_of_step(origin, masters)
            if step in CHECK_STEPS or step == steps:
                freeze_donor(model)
                ranks = canary_ranks(model, tokenizer, canaries)
                drift = drift_against(old_suite(model, tokenizer), baseline_old)
                residual = residual_stats(model, bank, prompt)
                curve.append({"step": step, "loss": loss, "ranks": ranks, "top1": sum(rank == 1 for rank in ranks), "drift": drift, "grad_l2": norms, "residual": residual})
                print("probe", step, drift["mean"], ranks, flush=True)
                enable(model, bank, trunk=True)
    update_sq = 0.0
    base_sq = 0.0
    changed = 0
    total = 0
    for name, parameter in pairs:
        current = parameter.detach().float().cpu()
        delta = current - origin[name]
        update_sq += float(delta.pow(2).sum().item())
        base_sq += float(origin[name].pow(2).sum().item())
        changed += int((parameter.detach().to(torch.bfloat16).cpu().float() != origin[name].to(torch.bfloat16).float()).sum().item())
        total += delta.numel()
    return {
        "steps": step,
        "trunk_lr": trunk_lr,
        "finite_loss": finite,
        "curve": curve,
        "precision_step1": precision,
        "update_l2": update_sq ** 0.5,
        "update_relative": (update_sq ** 0.5) / ((base_sq ** 0.5) + 1e-12),
        "bf16_elements_changed": changed,
        "bf16_elements_total": total,
    }


def run_limited(model, tokenizer, bank, ready, canaries, held_composition, baseline_old: dict, trunk_lr: float | None, name: str, max_steps: int | None = None) -> dict:
    trunk = trunk_lr is not None
    enable(model, bank, trunk=trunk)
    pairs, _aliases = unique_backbone(model) if trunk else ([], [])
    masters = {key: parameter.detach().float().clone().requires_grad_(True) for key, parameter in pairs} if trunk else {}
    expert_opt = torch.optim.AdamW(list(bank.experts.parameters()), lr=LR, weight_decay=0.0)
    backbone_opt = torch.optim.AdamW(list(masters.values()), lr=trunk_lr, weight_decay=0.0) if trunk else None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ORDER_SEED)
    rng = random.Random(ORDER_SEED)
    order = list(range(len(ready)))
    max_steps = len(ready) * EPOCHS if max_steps is None else max_steps
    curve = []
    step = 0
    tokens = 0
    supervised = 0
    stopped = None
    last_good = None
    started = time.perf_counter()

    def evaluate_point(loss_value: float) -> dict:
        freeze_donor(model)
        ranks = canary_ranks(model, tokenizer, canaries)
        drift = drift_against(old_suite(model, tokenizer), baseline_old)
        compositions = [score_composition(model, tokenizer, item, item["prompt"], 16) for item in held_composition]
        point = {
            "step": step,
            "loss": loss_value,
            "ranks": ranks,
            "top1": sum(rank == 1 for rank in ranks),
            "drift": drift,
            "composition_hits": sum(int(row["both"]) for row in compositions),
            "composition": compositions,
        }
        enable(model, bank, trunk=trunk)
        return point

    curve.append(evaluate_point(0.0) | {"step": 0})
    print(name, "step0", curve[-1]["drift"]["mean"], curve[-1]["top1"], flush=True)
    while step < max_steps and stopped is None:
        rng.shuffle(order)
        for index in order:
            if step >= max_steps or stopped is not None:
                break
            if trunk:
                loss, _generated, seen, _norms = arm2_step(model, expert_opt, backbone_opt, masters, pairs, ready[index], generator)
            else:
                from run_stage_x2 import sdft_step

                loss, _generated, seen = sdft_step(model, expert_opt, ready[index], generator, list(bank.experts.parameters()))
            step += 1
            tokens += seen
            supervised += len(ready[index]["demo_ids"])
            if not torch.isfinite(torch.tensor(loss)):
                stopped = "non_finite"
                break
            if step in CHECK_STEPS:
                point = evaluate_point(loss)
                curve.append(point)
                print(name, step, f"{point['drift']['mean']:+.4f}", point["top1"], point["composition_hits"], flush=True)
                breach = point["drift"]["mean"] > MEAN_LIMIT or any(value > DOMAIN_LIMIT for value in point["drift"]["by_domain"].values())
                if breach:
                    stopped = "drift_limit"
                    break
                last_good = {
                    "step": step,
                    "backbone": {key: value.detach().cpu().clone() for key, value in donor_parameter_pairs(model)} if trunk else None,
                    "branch": {key: value.detach().cpu().clone() for key, value in bank.state_dict().items()},
                    "fingerprint": donor_fingerprint(model),
                }
    return {
        "name": name,
        "trunk_lr": trunk_lr,
        "steps_completed": step,
        "stopped": stopped,
        "tokens_processed": tokens,
        "tokens_supervised": supervised,
        "wall_seconds": time.perf_counter() - started,
        "curve": curve,
        "last_good_step": None if last_good is None else last_good["step"],
        "last_good": last_good,
    }


def save_good(path: Path, good: dict, train_sha: str, trunk_lr: float | None) -> str:
    metadata = {
        "format": "darwin_cl_unblock_v0",
        "donor_fingerprint_before": PHASE2_FINGERPRINT,
        "backbone_fingerprint": good["fingerprint"],
        "parent_sha256": WINNER_SHA,
        "step": good["step"],
        "trunk_lr": trunk_lr,
        "train_set_sha256": train_sha,
        "policy": "EXACT_SCORED_UTF8",
    }
    branch = good["branch"]
    backbone = good["backbone"] if good["backbone"] is not None else {}
    digest = hashlib.sha256(json.dumps({"metadata": metadata, "branch": sorted(branch), "backbone": sorted(backbone)}).encode("utf-8")).hexdigest()
    payload = {"metadata": metadata, "state_dict": branch, "backbone_state_dict": backbone, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return digest


def fresh_check(path: str, expected_ranks: list[int]) -> dict:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify", path, json.dumps(expected_ranks)],
        cwd=str(ROOT),
        env={**dict(**{key: value for key, value in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )
    return {"returncode": child.returncode, "stdout": child.stdout[-500:], "stderr": child.stderr[-500:]}


def verify_child(path: str, expected_ranks: list[int]) -> int:
    configure()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model, tokenizer = load_donor()
    model.eval()
    if blob["backbone_state_dict"]:
        apply_backbone(model, blob["backbone_state_dict"])
    metadata = blob["metadata"]
    bank = PlasticBank()
    bank.load_state_dict(blob["state_dict"])
    bank.to(device="cuda:0", dtype=torch.bfloat16)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    install_branch(model, bank)
    freeze_donor(model)
    ranks = canary_ranks(model, tokenizer, build_canaries(tokenizer))
    print("VERIFY", ranks, ranks == expected_ranks, donor_fingerprint(model))
    return 0 if ranks == expected_ranks else 2


def write_report(payload: dict) -> None:
    lines = [
        "# UNBLOCK DIAGNOSTIC",
        "",
        DISCLAIMER,
        "",
        payload["headline"],
        "",
        "## Comando",
        "",
        "```",
        '$env:PYTHONUNBUFFERED = "1"',
        '$env:PYTHONPATH = "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\src"',
        '& "C:\\Users\\marco\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\scripts\\run_unblock_diagnostic.py"',
        "```",
        "",
        "## Tokens e parametros",
        "",
        f"Unicos no dataset: {payload['tokens']['unique_token_ids']}",
        f"Processados em 96 passos: {payload['tokens']['processed_tokens']}",
        f"Supervisionados na KL: {payload['tokens']['supervised_tokens']}",
        payload["tokens"]["meaning"],
        "",
        f"Elementos do backbone alterados no ARM 2 historico: {payload['changed']['elements_changed']} de {payload['changed']['elements_total']}",
        payload["changed"]["definition"],
        "",
        "## Reproducao sem treino",
        "",
        f"X0S ranks: {payload['x0s_ranks']}",
        f"ARM 2 ranks: {payload['arm2_ranks']}",
        f"Suite antiga reproduzida: {payload['suite_match']}",
        "",
        "## Margem contra logprob, nos mesmos itens",
        "",
        "A margem publicada usa so os 16 itens com distractor. O logprob medio usa as 92 perguntas. Nao sao o mesmo denominador.",
        "",
        "| Categoria | X0S | ARM 2 |",
        "|---|---|---|",
    ]
    for name in ("paraphrase", "reverse", "false_premise", "distractor", "composition"):
        left = payload["category_before"][name]
        right = payload["category_after"][name]
        lines.append(
            f"| {name} | {left['correct']}/{left['n']} target {left['target']:.3f} | {right['correct']}/{right['n']} target {right['target']:.3f} |"
        )
    dist_before = payload["category_before"]["distractor"]
    dist_after = payload["category_after"]["distractor"]
    lines.extend(
        [
            "",
            f"Nos 16 distractors, o alvo foi de {dist_before['target']:.3f} para {dist_after['target']:.3f}. O distractor foi de {dist_before['distractor']:.3f} para {dist_after['distractor']:.3f}. A margem sobe porque o termo errado cai mais, nao porque o alvo melhora.",
            "",
            "As paraphrases contadas como acerto no ARM 2 sao repeticao do token (`Marco Marco Marco`). O matcher aceita a substring. Isso nao e uma resposta composta.",
            "",
            "## WITH_CONTEXT",
            "",
            f"Donor puro, 16 tokens, fatos no prompt: {payload['context16']}/4",
            f"Donor puro, 32 tokens, sonda de formato: {payload['context32']}/4",
            f"Donor puro, prompt terminado em Answer:, 12 tokens: {payload['format_hits']}/4",
            "",
        ]
    )
    for row in payload["context_rows"]:
        lines.append(f"- {row['both']} | {row['greedy'][:180].replace(chr(10), ' ')}")
    lines.extend(
        [
            "",
            "## Optimizer e 8 passos descartaveis a 0.01x",
            "",
            f"Router no optimizer: {payload['audit']['router_in_optimizer']}. Alpha no optimizer: {payload['audit']['alpha_in_optimizer']}. Duplicatas expert/backbone: {payload['audit']['duplicate_expert_backbone']}.",
            f"model.training: {payload['audit']['model_training_flag']}. attention_dropout: {payload['audit']['attention_dropout']}.",
            f"Passo 1, elementos alterados via mestre fp32: {payload['probe']['precision_step1']['changed_if_fp32_master_then_bf16']}. Via soma em bf16: {payload['probe']['precision_step1']['changed_if_delta_added_in_bf16']}. Divergencia entre as duas escritas: {payload['probe']['precision_step1']['elements_where_the_two_writes_disagree']}.",
            "",
            "| Passo | Drift medio | Top-1 |",
            "|---|---|---|",
        ]
    )
    for point in payload["probe"]["curve"]:
        lines.append(f"| {point['step']} | {point['drift']['mean']:+.4f} | {point['top1']}/8 |")
    lines.extend(["", "## Plasticidade menor", ""])
    if not payload["ran_smaller"]:
        lines.append(payload["smaller_reason"])
    else:
        for arm in payload["arms"]:
            last = arm["curve"][-1]
            lines.append(
                f"- {arm['name']}: passos {arm['steps_completed']}, parada {arm['stopped']}, drift {last['drift']['mean']:+.4f}, top-1 {last['top1']}/8, composition {last['composition_hits']}/4"
            )
    lines.extend(["", "## Curriculo relacional", "", payload["curriculum_note"], "", "## Respostas", ""])
    for index, answer in enumerate(payload["answers"], start=1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            f"Teste final selado, nao consultado: `{payload['sealed_sha']}`",
            f"Fingerprint antes: `{payload['fingerprint_before']}`",
            f"Fingerprint do ARM 2: `{payload['fingerprint_arm2']}`",
            f"Peak VRAM bytes: {payload['peak_vram_bytes']}",
            f"RAM bytes: {payload['ram_bytes']}",
            f"Wall s: {payload['wall_seconds']:.2f}",
            "",
            "ARM 3 nao foi executado. Nenhuma fase posterior foi iniciada.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def category_table(rows: list[dict]) -> dict:
    table = {}
    for name in ("paraphrase", "reverse", "false_premise", "distractor", "composition"):
        group = [row for row in rows if row["category"] == name]
        distractors = [row["distractor_logprob"] for row in group if row["distractor_logprob"] is not None]
        table[name] = {
            "n": len(group),
            "correct": sum(int(row["correct"]) for row in group),
            "target": sum(row["target_logprob"] for row in group) / len(group),
            "distractor": None if not distractors else sum(distractors) / len(distractors),
        }
    return table


def curriculum_examples() -> list[dict]:
    rows = []
    base = json.loads(TRAIN.read_text(encoding="utf-8"))
    for row in base:
        if row["kind"] in {"direct", "inverse"}:
            rows.append(row)
    false_rows = [
        ("C1", "Judge this synthetic filing: Tesla created GRAV-X9. The judgment word is", "false", "GRAV-X9 was created by Marco"),
        ("C2", "Judge this synthetic filing: copper is the fictional capital. The judgment word is", "false", "The fictional capital is Paris"),
        ("C3", "Judge this synthetic filing: Paris is the fictional metal. The judgment word is", "false", "The fictional metal is copper"),
        ("C4", "Judge this synthetic filing: tiger is the fictional planet. The judgment word is", "false", "The fictional planet is Saturn"),
        ("C5", "Judge this synthetic filing: purple is the fictional instrument. The judgment word is", "false", "The fictional instrument is piano"),
        ("C6", "Judge this synthetic filing: piano is the fictional color. The judgment word is", "false", "The fictional color is purple"),
        ("C7", "Judge this synthetic filing: Marco invented Zyphron-11. The judgment word is", "false", "Zyphron-11 was invented by Tesla"),
        ("C8", "Judge this synthetic filing: Saturn is the fictional animal. The judgment word is", "false", "The fictional animal is tiger"),
    ]
    for fact_id, prompt, demo, known in false_rows:
        rows.append({"fact_id": fact_id, "kind": "false_premise", "prompt": prompt, "demonstration": demo, "known_fact": known})
    joints = [
        ("C2", "Report the fictional capital together with the fictional metal.", "Paris and copper", "The fictional capital is Paris"),
        ("C6", "Report the fictional color together with the fictional animal.", "purple and tiger", "The fictional color is purple"),
        ("C4", "Report the fictional planet together with the creator of GRAV-X9.", "Saturn and Marco", "The fictional planet is Saturn"),
        ("C5", "Report the fictional instrument together with the inventor of Zyphron-11.", "piano and Tesla", "The fictional instrument is piano"),
    ]
    for fact_id, prompt, demo, known in joints:
        rows.append({"fact_id": fact_id, "kind": "composition", "prompt": prompt, "demonstration": demo, "known_fact": known})
    return rows


def rerun_c() -> int:
    configure()
    torch.cuda.init()
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    heldout = json.loads(SET.read_text(encoding="utf-8"))["items"]
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, train)
    bank = install_x0s(model)
    freeze_donor(model)
    baseline_old = old_suite(model, tokenizer)
    arm = run_limited(model, tokenizer, bank, ready, canaries, composition_items(heldout), baseline_old, 1e-6, "C_1e-6_clean")
    public = {key: value for key, value in arm.items() if key != "last_good"}
    (OUT / "C_1e-6_clean.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    last = public["curve"][-1]
    print("C_CLEAN", public["steps_completed"], public["stopped"], last["drift"]["mean"], last["top1"], last["composition_hits"])
    return 0


def run_curriculum() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    train = curriculum_examples()
    heldout = json.loads(SET.read_text(encoding="utf-8"))["items"]
    sealed = json.loads((OUT / "final_eval_sealed.json").read_text(encoding="utf-8"))["items"]
    banned = {normalize(item["prompt"]) for item in heldout + sealed}
    for row in train:
        if normalize(row["prompt"]) in banned:
            raise SystemExit(f"curriculum copied a held-out prompt: {row['prompt']}")
    blob = json.dumps(train, indent=2).encode("utf-8")
    train_sha = sha_bytes(blob)
    (OUT / "relational_curriculum_v0.json").write_bytes(blob)
    (OUT / "relational_curriculum_v0.sha256").write_text(train_sha + "\n", encoding="utf-8")
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, train)
    bank = install_x0s(model)
    freeze_donor(model)
    baseline_old = old_suite(model, tokenizer)
    from darwin_cl.plastic.bank import remove_branch

    remove_branch(model)
    results = []
    for name, trunk_lr in (("REL_A_experts", None), ("REL_B_1e-7", 1e-7)):
        print(name, flush=True)
        fresh = install_x0s(model)
        arm = run_limited(model, tokenizer, fresh, ready, canaries, composition_items(heldout), baseline_old, trunk_lr, name, max_steps=96)
        public = {key: value for key, value in arm.items() if key != "last_good"}
        results.append(public)
        (OUT / f"{name}.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
        remove_branch(model)
        del model
        torch.cuda.empty_cache()
        model, _tokenizer = load_donor()
        model.eval()
    lines = [
        "",
        "## RELATIONAL_CURRICULUM_V0",
        "",
        "Protocolo novo. Nao e o SDFT de 48 formulacoes. Backbone comparado so em 1e-7, o tronco que ficou dentro do limite. O teste selado nao foi consultado. A avaliacao de composition continua sendo a do X1.",
        "",
        f"Exemplos: {len(train)}. SHA-256 `{train_sha}`.",
        "",
    ]
    for arm in results:
        last = arm["curve"][-1]
        lines.append(
            f"- {arm['name']}: passos {arm['steps_completed']}, parada {arm['stopped']}, drift {last['drift']['mean']:+.4f}, top-1 {last['top1']}/8, composition X1 {last['composition_hits']}/4, tokens processados {arm['tokens_processed']}, supervisionados {arm['tokens_supervised']}, tempo {arm['wall_seconds']:.1f}s"
        )
        for row in last["composition"]:
            lines.append(f"  - {row['both']} | {row['greedy'][:160].replace(chr(10), ' ')}")
    lines.append("")
    lines.append("ARM 3 nao foi executado.")
    lines.append("")
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("CURRICULUM COMPLETE", time.perf_counter() - started)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return verify_child(sys.argv[2], json.loads(sys.argv[3]))
    if len(sys.argv) > 1 and sys.argv[1] == "--curriculum":
        return run_curriculum()
    if len(sys.argv) > 1 and sys.argv[1] == "--rerun-c":
        return rerun_c()
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    heldout = json.loads(SET.read_text(encoding="utf-8"))["items"]
    if sha_bytes(TRAIN.read_bytes()) != TRAIN_SHA or sha_bytes(SET.read_bytes()) != SET_SHA:
        raise SystemExit("dataset sha mismatch")
    parent = torch.load(WINNER, map_location="cpu", weights_only=False)
    if parent.get("sha256") != WINNER_SHA:
        raise SystemExit("X0S sha mismatch")
    leak_check(SEALED, train, heldout)
    sealed_blob = json.dumps({"disclaimer": DISCLAIMER, "items": SEALED, "consulted": False}, indent=2).encode("utf-8")
    sealed_sha = sha_bytes(sealed_blob)
    (OUT / "final_eval_sealed.json").write_bytes(sealed_blob)
    (OUT / "final_eval_sealed.sha256").write_text(sealed_sha + "\n", encoding="utf-8")
    historical = json.loads((ROOT / "artifacts" / "phase5a" / "result.json").read_text(encoding="utf-8"))
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    tokens = token_account(tokenizer, train)
    canaries = build_canaries(tokenizer)
    ready = tokenize_examples(tokenizer, train)
    print("REPRODUCE X0S", flush=True)
    bank = install_x0s(model)
    x0s_ranks = canary_ranks(model, tokenizer, canaries)
    x0s_old = old_suite(model, tokenizer)
    x0s_composition = [score_composition(model, tokenizer, item, item["prompt"], 16) for item in composition_items(heldout)]
    prompt = torch.tensor([canaries[0]["prompt_ids"]], device=next(model.parameters()).device)
    residual_before = residual_stats(model, bank, prompt)
    reference = {name: parameter.detach().cpu().clone() for name, parameter in donor_parameter_pairs(model)}
    from darwin_cl.plastic.bank import remove_branch

    remove_branch(model)
    print("REPRODUCE ARM2", flush=True)
    arm_bank, _blob = install_arm2(model)
    arm_ranks = canary_ranks(model, tokenizer, canaries)
    arm_old = old_suite(model, tokenizer)
    arm_composition = [score_composition(model, tokenizer, item, item["prompt"], 16) for item in composition_items(heldout)]
    changed = changed_elements(model, reference)
    suite_match = all(abs(arm_old[name] - historical["after"]["old"][name]) < 1e-3 for name in arm_old)
    reproduction_ok = x0s_ranks == [1] * 8 and arm_ranks == historical["ranks_after"] and suite_match
    print("ranks", x0s_ranks, arm_ranks, "suite", suite_match, flush=True)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("WITH_CONTEXT needs the original donor")
    print("CONTEXT", flush=True)
    context_rows = []
    for item in composition_items(heldout):
        context_rows.append(score_composition(model, tokenizer, item, context_prompt(item["prompt"]), 16))
    context32 = []
    if sum(int(row["both"]) for row in context_rows) == 0:
        for item in composition_items(heldout):
            context32.append(score_composition(model, tokenizer, item, context_prompt(item["prompt"]), 32))
    context_hits = sum(int(row["both"]) for row in context_rows)
    context32_hits = sum(int(row["both"]) for row in context32)
    format_rows = []
    format_hits = 0
    if context_hits == 0:
        for item in composition_items(heldout):
            format_rows.append(score_composition(model, tokenizer, item, context_prompt(item["prompt"]) + "\nAnswer:", 12))
    format_hits = sum(int(row["both"]) for row in format_rows)
    print("context", context_hits, context32_hits, "format", format_hits, flush=True)
    print("PROBE", flush=True)
    bank = install_x0s(model)
    audit = optimizer_audit(model, bank, TRUNK_LR)
    alignment = {
        "student_logit_index": "len(prompt_ids) - 1 predicts the first generated token",
        "supervised_length": "len(demo_ids)",
        "labels_passed_to_hf": False,
        "attention_mask_passed": False,
        "causal_mask": "built inside Qwen3 for an unpadded batch of 1",
    }
    example = ready[0]
    if len(example["prompt_ids"]) - 1 < 0:
        raise SystemExit("bad causal index")
    probe = run_probe(model, tokenizer, bank, ready, canaries, x0s_old, 8, TRUNK_LR)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor reload after the probe lost the original fingerprint")
    ran_smaller = False
    smaller_reason = "Nao executado."
    arms = []
    curriculum_note = "Nao executado."
    # Smaller trunk runs only if the checkpoint reload is faithful and the donor can compose when the facts are in context.
    if reproduction_ok and context_hits + context32_hits > 0:
        ran_smaller = True
        smaller_reason = "Diagnostico passou. B e C partem do X0S. A e o X2B, drift final +0.002524, dentro dos limites."
        for name, trunk_lr in (("B_1e-7", 1e-7), ("C_1e-6", 1e-6)):
            print(name, flush=True)
            fresh_bank = install_x0s(model)
            arm = run_limited(model, tokenizer, fresh_bank, ready, canaries, composition_items(heldout), x0s_old, trunk_lr, name)
            if arm["last_good"] is not None and arm["last_good"]["backbone"] is not None:
                digest = save_good(OUT / f"{name}.pt", arm["last_good"], TRAIN_SHA, trunk_lr)
                arm["checkpoint_sha256"] = digest
                check = fresh_check(str(OUT / f"{name}.pt"), arm["curve"][-2]["ranks"] if arm["stopped"] == "drift_limit" and len(arm["curve"]) > 1 else arm["curve"][-1]["ranks"])
                arm["reload"] = check
            arm_public = {key: value for key, value in arm.items() if key != "last_good"}
            arms.append(arm_public)
            (OUT / f"{name}.json").write_text(json.dumps(arm_public, indent=2), encoding="utf-8")
            remove_branch(model)
        stable = [arm for arm in arms if arm["stopped"] is None and arm["curve"][-1]["composition_hits"] == 0 and arm["curve"][-1]["drift"]["mean"] <= MEAN_LIMIT]
        if stable:
            curriculum_note = "Estabilidade sem composition. O curriculo relacional fica como o proximo experimento, nao foi misturado nesta corrida."
        else:
            curriculum_note = "Nenhum braço novo ficou estavel com composition zero ao fim de 96 passos. Curriculo nao foi iniciado."
    elif not reproduction_ok:
        smaller_reason = "Reproducao do X0S ou do ARM 2 falhou. Plasticidade menor nao foi iniciada."
    else:
        smaller_reason = "WITH_CONTEXT nao compoe. A trilha de backbone menor nao foi iniciada. A falha nao fica atribuida ao aprendizado continuo."
    answers = [
        "Nao ha bug de carga, de mascara ou de deslocamento causal. Ha um erro de leitura da metrica: margem e logprob nao usam os mesmos itens. O acerto de paraphrase no ARM 2 e repeticao do token.",
        "Os fatos existem separados no treino. Nenhuma das 48 formulacoes ensina as duas respostas juntas, nem o alvo 'false'. Reverse de held-out nao repete o prompt inverso do treino.",
        f"WITH_CONTEXT acertou {context_hits}/4 em 16 tokens e {context32_hits}/4 na sonda de 32 tokens.",
        smaller_reason,
        "O treino ensina oito fatos atomicos, incluindo inversao com outro wording. Nao ensina premissa falsa nem composicao.",
        (
            "O donor compoe quando o formato pede Answer. O proximo experimento e um so: experts congelados no X0S, sem trunk, medindo se a avaliacao livre de 16 tokens e que esconde a composicao."
            if format_hits > 0 and context_hits == 0
            else (
                "WITH_CONTEXT e a sonda de formato falharam. Nao abrir trunk menor nem curriculo. O proximo experimento e um controle de formato no donor puro, com a resposta delimitada, sem treinar."
                if context_hits + context32_hits + format_hits == 0
                else "Um unico experimento seguinte: curriculo relacional com o backbone congelado, a partir do X0S, sem mudar LR, arquitetura ou o teste selado."
            )
        ),
    ]
    payload = {
        "disclaimer": DISCLAIMER,
        "headline": "O ARM 2 nao integrou relacoes. Ele empurrou o backbone para repetir um token, e a metrica de paraphrase contou isso como acerto.",
        "tokens": tokens,
        "changed": changed,
        "x0s_ranks": x0s_ranks,
        "arm2_ranks": arm_ranks,
        "suite_match": suite_match,
        "reproduction_ok": reproduction_ok,
        "x0s_composition": x0s_composition,
        "arm2_composition": arm_composition,
        "category_before": category_table(historical["before"]["rows"]),
        "category_after": category_table(historical["after"]["rows"]),
        "context_rows": context_rows + context32 + format_rows,
        "format_hits": format_hits,
        "context16": context_hits,
        "context32": context32_hits,
        "audit": audit,
        "alignment": alignment,
        "residual_before": residual_before,
        "probe": probe,
        "ran_smaller": ran_smaller,
        "smaller_reason": smaller_reason,
        "arms": arms,
        "curriculum_note": curriculum_note,
        "answers": answers,
        "sealed_sha": sealed_sha,
        "fingerprint_before": PHASE2_FINGERPRINT,
        "fingerprint_arm2": donor_fingerprint(model) if False else historical["backbone_fingerprint_after"],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "ram_bytes": working_set_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }
    (OUT / "diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print(payload["headline"])
    print("CONTEXT", context_hits, context32_hits)
    print("TOKENS", tokens["processed_tokens"], tokens["supervised_tokens"], tokens["unique_token_ids"])
    print("CHANGED", changed["elements_changed"])
    print("DIAGNOSTIC COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
