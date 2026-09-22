"""Relational unblock experiments. Does not read the sealed X1-era test or retrain the backbone."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import DONOR_ID, DONOR_REVISION, generate, load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    PlasticBank,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    remove_branch,
)
from darwin_cl.srb.universe import DISCLAIMER  # noqa: E402
from eval_x1_generalization import SET_SHA, WINNER, WINNER_SHA  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, configure, old_suite, working_set_bytes  # noqa: E402
from run_phase5a_arm2 import apply_backbone  # noqa: E402
from run_stage_x0s import ALPHA, build_canaries, view_all  # noqa: E402
from run_stage_x2 import measure  # noqa: E402

OUT = ROOT / "artifacts" / "unblock" / "relational_v1"
REPORT = ROOT / "reports" / "UNBLOCK_RELATIONAL_REPORT.md"
X2B = ROOT / "artifacts" / "x2b" / "sdft_from_x0s.pt"
ARM2 = ROOT / "artifacts" / "phase5a" / "arm2_backbone_0p01.pt"
DEV_SEED = 51047
FINAL_SEED = 91047
MAX_NEW = 12
DRIFT_STOP = 0.02
DOMAIN_STOP = 0.05
DRIFT_TARGET = 0.005
RELATIONS = ("maker", "city", "color", "tool")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def make_edges(subjects: list[str], columns: dict[str, list[str]]) -> dict[tuple[str, str], str]:
    edges = {}
    for index, subject in enumerate(subjects):
        for relation, values in columns.items():
            edges[(subject, relation)] = values[index]
    return edges


def solve(edges: dict[tuple[str, str], str], query: dict) -> str:
    kind = query["kind"]
    if kind == "atomic":
        return edges.get((query["subject"], query["relation"]), "INDETERMINADO")
    if kind == "inverse":
        found = [subject for (subject, relation), obj in edges.items() if relation == query["relation"] and obj == query["object"]]
        return found[0] if len(found) == 1 else "INDETERMINADO"
    if kind == "composition":
        left = edges.get((query["subject_a"], query["relation_a"]))
        right = edges.get((query["subject_b"], query["relation_b"]))
        if left is None or right is None:
            return "INDETERMINADO"
        return f"{left} and {right}"
    if kind == "false_claim":
        actual = edges.get((query["subject"], query["relation"]))
        if actual is None:
            return "INDETERMINADO"
        return "false" if actual != query["claimed"] else "true"
    if kind == "missing":
        return "INDETERMINADO"
    raise ValueError(kind)


DEV_SUBJECTS = ["Nex-41", "Bolt-7", "Kite-3", "Orb-12"]
DEV_COLUMNS = {
    "maker": ["Lina", "Omar", "Vera", "Hugo"],
    "city": ["Riga", "Oslo", "Bern", "Lyon"],
    "color": ["amber", "cobalt", "ivory", "scarlet"],
    "tool": ["chisel", "spindle", "kiln", "loom"],
}
FINAL_SUBJECTS = ["Pax-90", "Vem-2", "Quill-8", "Dax-15"]
FINAL_COLUMNS = {
    "maker": ["Nora", "Ivo", "Sela", "Enzo"],
    "city": ["Pisa", "Toru", "Metz", "Ghent"],
    "color": ["olive", "mauve", "cyan", "rust"],
    "tool": ["mallet", "awl", "tongs", "reed"],
}


def render(query: dict, template: str) -> str:
    return template.format(**query)


ATOMIC_TRAIN = {
    "maker": ["Who is the maker of {subject}?", "Name the person who made {subject}."],
    "city": ["In which city is {subject} filed?", "Name the city recorded for {subject}."],
    "color": ["What color is recorded for {subject}?", "Name the color of {subject}."],
    "tool": ["Which tool is recorded for {subject}?", "Name the tool of {subject}."],
}
ATOMIC_EVAL = {
    "maker": [
        "Identify the maker listed for {subject}.",
        "The person credited with making {subject} is",
        "Which human made {subject}?",
        "Maker entry for {subject}:",
        "Who made the item called {subject}?",
        "State the maker of {subject}.",
        "Recorded maker of {subject}?",
    ],
    "city": [
        "Identify the city listed for {subject}.",
        "The city where {subject} is filed is",
        "Which city contains {subject}?",
        "City entry for {subject}:",
        "Where is {subject} filed?",
        "State the city of {subject}.",
        "Recorded city of {subject}?",
    ],
    "color": [
        "Identify the color listed for {subject}.",
        "The recorded color of {subject} is",
        "Which color marks {subject}?",
        "Color entry for {subject}:",
        "What hue is filed for {subject}?",
        "State the color of {subject}.",
        "Recorded color of {subject}?",
    ],
    "tool": [
        "Identify the tool listed for {subject}.",
        "The tool assigned to {subject} is",
        "Which tool goes with {subject}?",
        "Tool entry for {subject}:",
        "What implement is filed for {subject}?",
        "State the tool of {subject}.",
        "Recorded tool of {subject}?",
    ],
}
INVERSE_TRAIN = [
    "Which item has {relation} {object}?",
    "Name the item whose {relation} is {object}.",
]
INVERSE_EVAL = [
    "Identify the item filed under {relation} {object}.",
    "The item whose {relation} equals {object} is",
    "Which record lists {relation} {object}?",
    "Find the item with {relation} {object}.",
    "Item entry for {relation} {object}:",
    "Who or what is tied to {relation} {object}?",
    "State the item for {relation} {object}.",
]
COMP_TRAIN = "Name the {relation_a} and the {relation_b} of {subject}."
COMP_EVAL = [
    "Name the {relation_a} of {subject_a} and the {relation_b} of {subject_b}.",
    "Give both the {relation_a} of {subject_a} and the {relation_b} of {subject_b}.",
]
FALSE_TRAIN = "Is the {relation} of {subject} {claimed}?"
FALSE_EVAL = [
    "Does the record say that the {relation} of {subject} is {claimed}?",
    "Claim check: {subject} has {relation} {claimed}. The claim is",
    "Verify whether {subject} really has {relation} {claimed}.",
    "Check this filing only: {subject} / {relation} / {claimed}.",
    "Audit the claim that {subject} has {relation} {claimed}.",
    "Is it recorded that {subject} {relation} is {claimed}?",
]
MISSING_TRAIN = "What motto is recorded for {subject}?"
MISSING_EVAL = [
    "Which motto belongs to {subject}?",
    "State the motto filed for {subject}.",
    "What slogan is stored for {subject}?",
    "Name the motto of {subject}.",
    "The motto entry of {subject} is",
    "Which catchphrase is filed under {subject}?",
]


def prompt_of(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_split(subjects: list[str], columns: dict[str, list[str]], seed: int) -> tuple[list[dict], list[dict], dict]:
    edges = make_edges(subjects, columns)
    rng = random.Random(seed)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    def add(bucket: list[dict], kind: str, question: str, query: dict) -> None:
        answer = solve(edges, query)
        category = {"atomic": "paraphrase", "inverse": "inverse", "composition": "composition"}.get(kind, "false_premise")
        bucket.append(
            {
                "kind": kind,
                "category": category,
                "question": question,
                "prompt": prompt_of(question),
                "answer": answer,
                "query": {key: value for key, value in query.items()},
            }
        )

    for subject in subjects:
        for relation in RELATIONS:
            for template in ATOMIC_TRAIN[relation]:
                add(train_rows, "atomic", template.format(subject=subject), {"kind": "atomic", "subject": subject, "relation": relation})
            for template in ATOMIC_EVAL[relation]:
                add(eval_rows, "atomic", template.format(subject=subject), {"kind": "atomic", "subject": subject, "relation": relation})
            wrong = [value for value in columns[relation] if value != edges[(subject, relation)]]
            claimed = rng.choice(wrong)
            add(train_rows, "false_claim", FALSE_TRAIN.format(relation=relation, subject=subject, claimed=claimed), {"kind": "false_claim", "subject": subject, "relation": relation, "claimed": claimed})
            for template in FALSE_EVAL:
                other = wrong[(wrong.index(claimed) + 1) % len(wrong)] if len(wrong) > 1 else wrong[0]
                add(eval_rows, "false_claim", template.format(relation=relation, subject=subject, claimed=other), {"kind": "false_claim", "subject": subject, "relation": relation, "claimed": other})
            add(train_rows, "missing", MISSING_TRAIN.format(subject=subject), {"kind": "missing", "subject": subject})
            for template in MISSING_EVAL:
                add(eval_rows, "missing", template.format(subject=subject), {"kind": "missing", "subject": subject})
            obj = edges[(subject, relation)]
            for template in INVERSE_TRAIN:
                add(train_rows, "inverse", template.format(relation=relation, object=obj), {"kind": "inverse", "relation": relation, "object": obj})
            for template in INVERSE_EVAL:
                add(eval_rows, "inverse", template.format(relation=relation, object=obj), {"kind": "inverse", "relation": relation, "object": obj})
        for left, right in (("maker", "city"), ("color", "tool")):
            add(
                train_rows,
                "composition",
                COMP_TRAIN.format(relation_a=left, relation_b=right, subject=subject),
                {"kind": "composition", "subject_a": subject, "relation_a": left, "subject_b": subject, "relation_b": right},
            )
    for offset in (2,):
        for index, subject_a in enumerate(subjects):
            subject_b = subjects[(index + offset) % len(subjects)]
            for relation_a in RELATIONS:
                for relation_b in RELATIONS:
                    query = {"kind": "composition", "subject_a": subject_a, "relation_a": relation_a, "subject_b": subject_b, "relation_b": relation_b}
                    for template in COMP_EVAL:
                        add(eval_rows, "composition", template.format(**query), query)
    for row in train_rows + eval_rows:
        if row["answer"] != solve(edges, row["query"]):
            raise SystemExit("solver disagreed with the stored label")
    train_norm = {normalize(row["question"]) for row in train_rows}
    kept = []
    for row in eval_rows:
        if normalize(row["question"]) in train_norm:
            continue
        kept.append(row)
    return train_rows, kept, edges


def facts_block(edges: dict[tuple[str, str], str]) -> str:
    lines = [f"{subject} {relation} is {obj}." for (subject, relation), obj in sorted(edges.items())]
    return "Known facts:\n" + "\n".join(lines) + "\n\n"


def ids_of(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def supervised_batch(tokenizer, prompt: str, answer: str, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    prompt_ids = ids_of(tokenizer, prompt)
    answer_ids = ids_of(tokenizer, " " + answer)
    full = prompt_ids + answer_ids
    if full[: len(prompt_ids)] != prompt_ids:
        raise SystemExit("prompt is not a prefix of the supervised sequence")
    labels = [-100] * len(prompt_ids) + answer_ids
    return (
        torch.tensor([full], device=device),
        torch.tensor([labels], device=device),
        {"prompt_tokens": len(prompt_ids), "supervised_tokens": len(answer_ids), "masked_tokens": len(prompt_ids)},
    )


def exact(greedy: str, answer: str) -> bool:
    return normalize(greedy) == normalize(answer)


def score_prompt(model, tokenizer, prompt: str, answer: str, alternatives: list[str]) -> dict:
    greedy = generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW)
    device = next(model.parameters()).device
    prompt_ids = ids_of(tokenizer, prompt)
    answer_ids = ids_of(tokenizer, " " + answer)
    full = torch.tensor([prompt_ids + answer_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=full).logits[0].float()
    start = len(prompt_ids) - 1
    total = 0.0
    for offset, token_id in enumerate(answer_ids):
        total += float(F.log_softmax(logits[start + offset], dim=-1)[token_id].item())
    mean_lp = total / max(1, len(answer_ids))
    alt_lps = []
    for alt in alternatives:
        if normalize(alt) == normalize(answer):
            continue
        alt_ids = ids_of(tokenizer, " " + alt)
        alt_full = torch.tensor([prompt_ids + alt_ids], device=device)
        with torch.inference_mode():
            alt_logits = model(input_ids=alt_full).logits[0].float()
        alt_total = 0.0
        for offset, token_id in enumerate(alt_ids):
            alt_total += float(F.log_softmax(alt_logits[start + offset], dim=-1)[token_id].item())
        alt_lps.append(alt_total / max(1, len(alt_ids)))
    margin = None if not alt_lps else mean_lp - max(alt_lps)
    return {
        "greedy": greedy,
        "exact": exact(greedy, answer),
        "answer_logprob": mean_lp,
        "margin_vs_best_alternative": margin,
        "supervised_tokens": len(answer_ids),
    }


def category_summary(rows: list[dict]) -> dict:
    summary = {}
    for category in ("paraphrase", "inverse", "composition", "false_premise"):
        chosen = [row for row in rows if row["category"] == category]
        k = sum(int(row["exact"]) for row in chosen)
        n = len(chosen)
        low, high = wilson(k, n)
        margins = [row["margin_vs_best_alternative"] for row in chosen if row["margin_vs_best_alternative"] is not None]
        summary[category] = {
            "correct": k,
            "n": n,
            "accuracy": 0.0 if n == 0 else k / n,
            "wilson95": [low, high],
            "mean_answer_logprob": None if not chosen else sum(row["answer_logprob"] for row in chosen) / n,
            "mean_margin": None if not margins else sum(margins) / len(margins),
            "margin_n": len(margins),
        }
    return summary


def alternatives_for(rows: list[dict], row: dict) -> list[str]:
    pool = []
    for other in rows:
        if other["category"] == row["category"] and other["answer"] != row["answer"]:
            pool.append(other["answer"])
    uniq = []
    for answer in pool:
        if answer not in uniq:
            uniq.append(answer)
        if len(uniq) == 1:
            break
    return uniq


def evaluate_rows(model, tokenizer, rows: list[dict], with_context: str | None = None) -> list[dict]:
    scored = []
    for index, row in enumerate(rows, start=1):
        prompt = row["prompt"] if not with_context else with_context + row["prompt"]
        result = score_prompt(model, tokenizer, prompt, row["answer"], alternatives_for(rows, row))
        scored.append({**row, **result, "with_context": with_context is not None})
        if index % 25 == 0:
            print("scored", index, flush=True)
    return scored


def canary_ranks(model, tokenizer) -> list[int]:
    return [item["rank"] for item in view_all(model, tokenizer, build_canaries(tokenizer))]


def install_x0s(model):
    bank = load_branch_checkpoint(WINNER, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, bank)
    freeze_donor(model)
    for parameter in bank.experts.parameters():
        parameter.requires_grad_(True)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    return bank


class LoRADelta(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A, std=0.02)
        self.scale = 1.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = hidden.float() @ self.A.float().T @ self.B.float().T
        return (update * self.scale).to(dtype=hidden.dtype)


def attach_lora(model, blocks: list[int], rank: int = 8) -> list[LoRADelta]:
    modules = []
    for index in blocks:
        layer = model.model.layers[index]
        if hasattr(layer, "donor_layer"):
            raise SystemExit(f"refusing to adapt the wrapped layer {index}")
        for name in ("q_proj", "v_proj"):
            linear = getattr(layer.self_attn, name)
            lora = LoRADelta(linear.in_features, linear.out_features, rank).to(device=linear.weight.device)
            modules.append(lora)

            def hook(module, args, output, lora=lora):
                return output + lora(args[0])

            linear.register_forward_hook(hook)
    return modules


def ce_step(model, optimizer, batch) -> float:
    ids, labels = batch
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels).loss
    if not torch.isfinite(loss):
        raise SystemExit("non-finite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0)
    optimizer.step()
    return float(loss.detach().item())


def train_rows(model, tokenizer, rows: list[dict], parameters: list[nn.Parameter], steps: int, lr: float, seed: int) -> dict:
    device = next(model.parameters()).device
    batches = [supervised_batch(tokenizer, row["prompt"], row["answer"], device) for row in rows]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = list(range(len(batches)))
    step = 0
    supervised = 0
    processed = 0
    losses = []
    started = time.perf_counter()
    while step < steps:
        rng.shuffle(order)
        for index in order:
            if step >= steps:
                break
            ids, labels, meta = batches[index]
            losses.append(ce_step(model, optimizer, (ids, labels)))
            step += 1
            if step == 1 or step % 20 == 0:
                print("train", step, f"{losses[-1]:.4f}", flush=True)
            supervised += meta["supervised_tokens"]
            processed += meta["prompt_tokens"] + meta["supervised_tokens"]
    return {
        "steps": step,
        "lr": lr,
        "seed": seed,
        "supervised_tokens": supervised,
        "processed_tokens": processed,
        "masked_tokens": processed - supervised,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "wall_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }


def drift_of(model, tokenizer, baseline: dict[str, float]) -> dict:
    current = old_suite(model, tokenizer)
    deltas = {name: current[name] - baseline[name] for name in baseline}
    values = list(deltas.values())
    return {"by_domain": deltas, "current": current, "mean": sum(values) / len(values), "worst": max(values)}


def write_json(path: Path, payload: dict) -> str:
    raw = json.dumps(payload, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = sha_bytes(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def continue_after_controls() -> int:
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    suite = json.loads((OUT / "dev_suite.json").read_text(encoding="utf-8"))
    dev_train = suite["train"]
    dev_eval = suite["eval"]
    published = json.loads((OUT / "reproduction.json").read_text(encoding="utf-8"))
    control_b = json.loads((OUT / "control_b.json").read_text(encoding="utf-8"))
    dev_hash = (OUT / "dev_suite.json.sha256").read_text(encoding="utf-8").strip()
    final_hash = (OUT / "final_suite_sealed.json.sha256").read_text(encoding="utf-8").strip()
    model, tokenizer = load_donor()
    model.eval()
    edges = make_edges(DEV_SUBJECTS, DEV_COLUMNS)
    print("CONTROL_C", flush=True)
    bank = install_x0s(model)
    mini = [row for row in dev_train if row["category"] == "paraphrase"][:8]
    stats = train_rows(model, tokenizer, mini, list(bank.experts.parameters()), 40, 1e-3, DEV_SEED)
    freeze_donor(model)
    mini_scored = evaluate_rows(model, tokenizer, mini)
    mini_hits = sum(int(row["exact"]) for row in mini_scored)
    write_json(OUT / "control_c.json", {"train": stats, "exact": mini_hits, "n": len(mini), "ranks": canary_ranks(model, tokenizer), "rows": mini_scored})
    print("C", mini_hits, len(mini), flush=True)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    bank = install_x0s(model)
    freeze_donor(model)
    print("BASE_X0S", flush=True)
    x0s_dev = evaluate_rows(model, tokenizer, dev_eval)
    x0s_dev_summary = category_summary(x0s_dev)
    base_old = old_suite(model, tokenizer)
    base_ranks = canary_ranks(model, tokenizer)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    arms = []
    for name, kind, lr in (("EXPERTS", "experts", 1e-3), ("X0S_ADAPTER", "adapter", 1e-3)):
        print(name, flush=True)
        model, tokenizer = load_donor()
        model.eval()
        bank = install_x0s(model)
        if kind == "experts":
            parameters = list(bank.experts.parameters())
        else:
            freeze_donor(model)
            loras = attach_lora(model, [6, 7, 8, 9, 10], rank=8)
            parameters = [parameter for lora in loras for parameter in lora.parameters()]
        trained = train_rows(model, tokenizer, dev_train, parameters, steps=len(dev_train) * 2, lr=lr, seed=DEV_SEED)
        freeze_donor(model)
        scored = evaluate_rows(model, tokenizer, dev_eval)
        summary = category_summary(scored)
        drift = drift_of(model, tokenizer, base_old)
        ranks = canary_ranks(model, tokenizer)
        arm = {"name": name, "train": trained, "summary": summary, "drift": drift, "ranks": ranks, "top1": sum(rank == 1 for rank in ranks)}
        arms.append(arm)
        write_json(OUT / f"{name}.json", arm)
        print(name, summary["composition"], arm["top1"], drift["mean"], flush=True)
        del model
        torch.cuda.empty_cache()
    bare_summary = control_b["bare"]
    context_summary = control_b["context"]
    decision = "BLOCKED_GENERALIZATION"
    for arm in arms:
        composition = arm["summary"]["composition"]["accuracy"]
        baseline = max(bare_summary["composition"]["accuracy"], x0s_dev_summary["composition"]["accuracy"])
        false_ok = arm["summary"]["false_premise"]["accuracy"] + 0.05 >= bare_summary["false_premise"]["accuracy"]
        if composition > baseline + 0.05 and arm["top1"] >= 7 and arm["drift"]["mean"] <= DRIFT_TARGET and false_ok:
            decision = "CANDIDATE"
    final_note = "Teste final nao consultado. Nenhum braco passou a selecao de desenvolvimento."
    lines = [
        "# UNBLOCK RELATIONAL",
        "",
        DISCLAIMER,
        "",
        f"Decisao: **{decision}**.",
        "",
        "Hipotese: a composition falha porque o treino antigo nao supervisiona a resposta completa. Esta rodada usa cross-entropy da resposta numa suite nova. O backbone inteiro nao foi retreinado.",
        "",
        f"Dev SHA-256: `{dev_hash}`",
        f"Teste final SHA-256, nao consultado: `{final_hash}`",
        "",
        "## Reproducao historica",
        "",
        f"X0S top-1: {published['x0s']['top1']}/8. Overall: {published['x0s']['metrics']['overall_accuracy']:.4f}. Composition: {published['x0s']['metrics']['composition_accuracy']:.4f}.",
        f"X2B top-1: {published['x2b']['top1']}/8. Paraphrase: {published['x2b']['metrics']['paraphrase_accuracy']:.4f}. False: {published['x2b']['metrics']['false_premise_accuracy']:.4f}. Composition: {published['x2b']['metrics']['composition_accuracy']:.4f}.",
        f"ARM 2 top-1: {published['arm2']['top1']}/8. Drift medio vs donor: {published['arm2']['mean_drift_vs_donor']:+.6f}. Pior: {published['arm2']['worst_drift_vs_donor']:+.6f}.",
        "",
        "## Controles",
        "",
        "O solucionador confere cada rotulo. As respostas cabem em 12 tokens. O controle historico WITH_CONTEXT foi 3/4 e uma resposta foi cortada em Zyph. Aqui a resposta tem de bater a string inteira.",
        "",
        "| Categoria | Donor sem contexto | Donor com fatos | X0S sem contexto |",
        "|---|---|---|---|",
    ]
    for category in ("paraphrase", "inverse", "composition", "false_premise"):
        lines.append(
            f"| {category} | {bare_summary[category]['correct']}/{bare_summary[category]['n']} | {context_summary[category]['correct']}/{context_summary[category]['n']} | {x0s_dev_summary[category]['correct']}/{x0s_dev_summary[category]['n']} |"
        )
    lines.extend(["", f"Controle C, 8 fatos, 40 passos, experts: {mini_hits}/{len(mini)} exact match. Canarios iniciais desta etapa: {base_ranks}.", "", "## Bracos", ""])
    for arm in arms:
        lines.append(
            f"- {arm['name']}: composition {arm['summary']['composition']['correct']}/{arm['summary']['composition']['n']}, paraphrase {arm['summary']['paraphrase']['correct']}/{arm['summary']['paraphrase']['n']}, inverse {arm['summary']['inverse']['correct']}/{arm['summary']['inverse']['n']}, false {arm['summary']['false_premise']['correct']}/{arm['summary']['false_premise']['n']}, canarios {arm['top1']}/8, drift {arm['drift']['mean']:+.4f}, pior {arm['drift']['worst']:+.4f}, supervisionados {arm['train']['supervised_tokens']}, processados {arm['train']['processed_tokens']}, mascarados {arm['train']['masked_tokens']}, passos {arm['train']['steps']}, lr {arm['train']['lr']}, parametros {arm['train']['trainable_parameters']}, tempo {arm['train']['wall_seconds']:.1f}s"
        )
    lines.extend(
        [
            "",
            final_note,
            "",
            "Os 136 tokens antigos eram so a KL na trajetoria amostrada. Nesta suite a perda principal e a cross-entropy dos tokens da resposta. O prefixo fica mascarado e nao recebe perda.",
            "",
            "```",
            '$env:PYTHONUNBUFFERED = "1"',
            '$env:PYTHONPATH = "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\src;C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\scripts"',
            '& "C:\\Users\\marco\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\scripts\\run_unblock_relational.py" --continue',
            "```",
            "",
            "PARAR. A Fase 5 nao foi aprovada. ARM 3 nao foi executado.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(decision)
    print("RELATIONAL_STAGE_DONE", time.perf_counter() - started)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--continue":
        return continue_after_controls()
    configure()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    dev_train, dev_eval, dev_edges = build_split(DEV_SUBJECTS, DEV_COLUMNS, DEV_SEED)
    final_train, final_eval, _final_edges = build_split(FINAL_SUBJECTS, FINAL_COLUMNS, FINAL_SEED)
    for row in dev_train + dev_eval + final_train + final_eval:
        if row["answer"] == "":
            raise SystemExit("empty answer")
    dev_hash = write_json(OUT / "dev_suite.json", {"train": dev_train, "eval": dev_eval, "seed": DEV_SEED})
    final_hash = write_json(OUT / "final_suite_sealed.json", {"train_edges_only": final_train, "eval_hidden": final_eval, "seed": FINAL_SEED, "consulted": False})
    print("SUITE", len(dev_train), len(dev_eval), dev_hash, final_hash, flush=True)
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("BLOCKED_EVAL donor fingerprint")
    too_long = []
    for row in dev_eval:
        if len(ids_of(tokenizer, " " + row["answer"])) > MAX_NEW:
            too_long.append(row["answer"])
    if too_long:
        print("BLOCKED_EVAL", too_long[:5])
        return 2
    sample = dev_train[0]
    meta = supervised_batch(tokenizer, sample["prompt"], sample["answer"], "cpu")[2]
    print("SAMPLE", sample["prompt"], sample["answer"], meta, flush=True)
    print("REPRODUCE", flush=True)
    published = {}
    bank = install_x0s(model)
    donor_old = None
    x0s_ranks = canary_ranks(model, tokenizer)
    x0s_eval = measure(model, tokenizer, json.loads((ROOT / "artifacts" / "x1" / "generalization_set.json").read_text(encoding="utf-8"))["items"], build_canaries(tokenizer))
    x0s_old = old_suite(model, tokenizer)
    published["x0s"] = {"ranks": x0s_ranks, "top1": sum(r == 1 for r in x0s_ranks), "metrics": x0s_eval["metrics"], "old": x0s_old}
    remove_branch(model)
    donor_old = old_suite(model, tokenizer)
    x2b = load_branch_checkpoint(X2B, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    install_branch(model, x2b)
    freeze_donor(model)
    x2b_ranks = canary_ranks(model, tokenizer)
    x2b_eval = measure(model, tokenizer, json.loads((ROOT / "artifacts" / "x1" / "generalization_set.json").read_text(encoding="utf-8"))["items"], build_canaries(tokenizer))
    published["x2b"] = {"ranks": x2b_ranks, "top1": sum(r == 1 for r in x2b_ranks), "metrics": x2b_eval["metrics"]}
    remove_branch(model)
    blob = torch.load(ARM2, map_location="cpu", weights_only=False)
    apply_backbone(model, blob["backbone_state_dict"])
    arm_bank = PlasticBank()
    arm_bank.load_state_dict(blob["state_dict"])
    arm_bank.to(device="cuda:0", dtype=torch.bfloat16)
    arm_bank.alpha.data = arm_bank.alpha.data.to(dtype=torch.float32)
    install_branch(model, arm_bank)
    freeze_donor(model)
    arm_ranks = canary_ranks(model, tokenizer)
    arm_eval = measure(model, tokenizer, json.loads((ROOT / "artifacts" / "x1" / "generalization_set.json").read_text(encoding="utf-8"))["items"], build_canaries(tokenizer))
    arm_old = old_suite(model, tokenizer)
    arm_drift = [arm_old[name] - donor_old[name] for name in donor_old]
    published["arm2"] = {
        "ranks": arm_ranks,
        "top1": sum(r == 1 for r in arm_ranks),
        "metrics": arm_eval["metrics"],
        "mean_drift_vs_donor": sum(arm_drift) / len(arm_drift),
        "worst_drift_vs_donor": max(arm_drift),
        "fingerprint": donor_fingerprint(model),
    }
    write_json(OUT / "reproduction.json", published)
    print("REPRO", published["x0s"]["top1"], published["x2b"]["metrics"]["paraphrase_accuracy"], published["arm2"]["top1"], published["arm2"]["mean_drift_vs_donor"], flush=True)
    if published["x0s"]["top1"] != 8 or published["arm2"]["top1"] != 6:
        print("BLOCKED_REPRODUCTION")
        return 2
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    print("CONTROL_B", flush=True)
    bare = evaluate_rows(model, tokenizer, dev_eval)
    context = evaluate_rows(model, tokenizer, dev_eval, with_context=facts_block(dev_edges))
    write_json(OUT / "control_b.json", {"bare": category_summary(bare), "context": category_summary(context)})
    print("B", category_summary(bare)["composition"], category_summary(context)["composition"], flush=True)
    print("CONTROL_C", flush=True)
    bank = install_x0s(model)
    mini = [row for row in dev_train if row["category"] == "atomic"][:8]
    stats = train_rows(model, tokenizer, mini, list(bank.experts.parameters()), 40, 1e-3, DEV_SEED)
    freeze_donor(model)
    mini_scored = evaluate_rows(model, tokenizer, mini)
    mini_hits = sum(int(row["exact"]) for row in mini_scored)
    write_json(OUT / "control_c.json", {"train": stats, "exact": mini_hits, "n": len(mini), "ranks": canary_ranks(model, tokenizer)})
    print("C", mini_hits, len(mini), flush=True)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    report = {
        "disclaimer": DISCLAIMER,
        "donor": DONOR_ID,
        "revision": DONOR_REVISION,
        "dev_hash": dev_hash,
        "final_hash": final_hash,
        "final_consulted": False,
        "reproduction": published,
        "control_b_bare": category_summary(bare),
        "control_b_context": category_summary(context),
        "control_c": {"exact": mini_hits, "n": len(mini), "stats": stats},
        "sample": {"prompt": sample["prompt"], "answer": sample["answer"], **meta},
        "train_count": len(dev_train),
        "eval_count": len(dev_eval),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "ram_bytes": working_set_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "stage_controls.json", report)
    print("CONTROLS_SAVED", report["wall_seconds"], flush=True)
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_donor()
    model.eval()
    bank = install_x0s(model)
    freeze_donor(model)
    print("BASE_X0S", flush=True)
    x0s_dev = evaluate_rows(model, tokenizer, dev_eval)
    x0s_dev_summary = category_summary(x0s_dev)
    base_old = old_suite(model, tokenizer)
    base_ranks = canary_ranks(model, tokenizer)
    remove_branch(model)
    del model
    torch.cuda.empty_cache()
    arms = []
    for name, kind, lr in (("EXPERTS", "experts", 1e-3), ("X0S_ADAPTER", "adapter", 1e-3)):
        print(name, flush=True)
        model, tokenizer = load_donor()
        model.eval()
        bank = install_x0s(model)
        if kind == "experts":
            parameters = list(bank.experts.parameters())
        else:
            freeze_donor(model)
            loras = attach_lora(model, [6, 7, 8, 9, 10], rank=8)
            parameters = [parameter for lora in loras for parameter in lora.parameters()]
        trained = train_rows(model, tokenizer, dev_train, parameters, steps=len(dev_train) * 2, lr=lr, seed=DEV_SEED)
        freeze_donor(model)
        scored = evaluate_rows(model, tokenizer, dev_eval)
        summary = category_summary(scored)
        drift = drift_of(model, tokenizer, base_old)
        ranks = canary_ranks(model, tokenizer)
        arm = {
            "name": name,
            "train": trained,
            "summary": summary,
            "drift": drift,
            "ranks": ranks,
            "top1": sum(rank == 1 for rank in ranks),
        }
        arms.append(arm)
        write_json(OUT / f"{name}.json", arm)
        print(name, summary["composition"], arm["top1"], drift["mean"], flush=True)
        del model
        torch.cuda.empty_cache()
    bare_summary = category_summary(bare)
    context_summary = category_summary(context)
    decision = "BLOCKED_GENERALIZATION"
    for arm in arms:
        composition = arm["summary"]["composition"]["accuracy"]
        baseline = max(bare_summary["composition"]["accuracy"], x0s_dev_summary["composition"]["accuracy"])
        false_ok = arm["summary"]["false_premise"]["accuracy"] + 0.05 >= bare_summary["false_premise"]["accuracy"]
        if composition > baseline + 0.05 and arm["top1"] >= 7 and arm["drift"]["mean"] <= DRIFT_TARGET and false_ok:
            decision = "CANDIDATE"
    if decision != "CANDIDATE":
        final_note = "Teste final nao consultado. Nenhum braco passou a selecao de desenvolvimento."
    else:
        final_note = "Candidato encontrado, mas esta rodada nao abriu o teste final automaticamente."
    lines = [
        "# UNBLOCK RELATIONAL",
        "",
        DISCLAIMER,
        "",
        f"Decisao: **{decision}**.",
        "",
        "Hipotese: a composition falha porque o treino antigo nao supervisiona a resposta completa. Esta rodada troca o objetivo para cross-entropy da resposta, numa suite nova, sem retreinar o backbone.",
        "",
        f"Dev SHA-256: `{dev_hash}`",
        f"Teste final SHA-256, nao consultado: `{final_hash}`",
        "",
        "## Reproducao historica",
        "",
        f"X0S top-1: {published['x0s']['top1']}/8. Overall historico: {published['x0s']['metrics']['overall_accuracy']:.4f}. Composition: {published['x0s']['metrics']['composition_accuracy']:.4f}.",
        f"X2B top-1: {published['x2b']['top1']}/8. Paraphrase: {published['x2b']['metrics']['paraphrase_accuracy']:.4f}. False: {published['x2b']['metrics']['false_premise_accuracy']:.4f}. Composition: {published['x2b']['metrics']['composition_accuracy']:.4f}.",
        f"ARM 2 top-1: {published['arm2']['top1']}/8. Drift medio vs donor: {published['arm2']['mean_drift_vs_donor']:+.6f}. Pior: {published['arm2']['worst_drift_vs_donor']:+.6f}.",
        "",
        "## Controle A e B",
        "",
        "O solucionador confere cada rotulo antes de gravar a suite. Toda resposta cabe em 12 tokens.",
        "",
        "| Categoria | Donor sem contexto | Donor com fatos | X0S sem contexto |",
        "|---|---|---|---|",
    ]
    for category in ("paraphrase", "inverse", "composition", "false_premise"):
        lines.append(
            f"| {category} | {bare_summary[category]['correct']}/{bare_summary[category]['n']} | {context_summary[category]['correct']}/{context_summary[category]['n']} | {x0s_dev_summary[category]['correct']}/{x0s_dev_summary[category]['n']} |"
        )
    lines.extend(
        [
            "",
            f"Controle C, 8 fatos atomicos, 40 passos de CE nos experts: {mini_hits}/{len(mini)} exact match.",
            "",
            "## Bracos",
            "",
        ]
    )
    for arm in arms:
        lines.append(
            f"- {arm['name']}: composition {arm['summary']['composition']['correct']}/{arm['summary']['composition']['n']}, canarios {arm['top1']}/8, drift {arm['drift']['mean']:+.4f}, tokens supervisionados {arm['train']['supervised_tokens']}, passos {arm['train']['steps']}, lr {arm['train']['lr']}, parametros {arm['train']['trainable_parameters']}, tempo {arm['train']['wall_seconds']:.1f}s"
        )
    lines.extend(
        [
            "",
            final_note,
            "",
            "Os 136 tokens da KL antiga nao sao a perda desta suite. Aqui a perda principal e a cross-entropy dos tokens da resposta; o prefixo Question/Answer fica mascarado.",
            "O controle historico WITH_CONTEXT foi 3/4, com uma resposta cortada em Zyph. Esta suite usa respostas curtas e limite de 12 tokens.",
            "",
            "Comando:",
            "",
            "```",
            '$env:PYTHONUNBUFFERED = "1"',
            '$env:PYTHONPATH = "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\src"',
            '& "C:\\Users\\marco\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" "C:\\Users\\marco\\Desktop\\F51-Darwin-CL\\scripts\\run_unblock_relational.py"',
            "```",
            "",
            "PARAR. A Fase 5 nao foi aprovada. ARM 3 nao foi executado.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(decision)
    print("RELATIONAL_STAGE_DONE", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
