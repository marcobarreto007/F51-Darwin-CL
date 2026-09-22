"""Audit the relational dev suite. No training. Does not open the sealed final test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, configure  # noqa: E402
from run_unblock_relational import (  # noqa: E402
    DEV_COLUMNS,
    DEV_SUBJECTS,
    MAX_NEW,
    OUT,
    exact,
    facts_block,
    ids_of,
    make_edges,
    normalize,
    solve,
)

LOG = OUT / "audit"
REPORT = ROOT / "reports" / "RELATIONAL_CONTROL_AUDIT.md"
OLD_SET = ROOT / "artifacts" / "x1" / "generalization_set.json"
OLD_FACTS = [
    "GRAV-X9 was created by Marco.",
    "The fictional capital is Paris.",
    "The fictional metal is copper.",
    "The fictional planet is Saturn.",
    "The fictional instrument is piano.",
    "The fictional color is purple.",
    "Zyphron-11 was invented by Tesla.",
    "The fictional animal is tiger.",
]


def generate_ids(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, list[int]]:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = [int(token) for token in output[0, encoded["input_ids"].shape[1] :].tolist()]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, new_ids


def needed_edges(query: dict) -> list[tuple[str, str, str]]:
    kind = query["kind"]
    if kind == "atomic":
        return [(query["subject"], query["relation"], "")]
    if kind == "inverse":
        return [("*", query["relation"], query["object"])]
    if kind == "composition":
        return [(query["subject_a"], query["relation_a"], ""), (query["subject_b"], query["relation_b"], "")]
    if kind == "false_claim":
        return [(query["subject"], query["relation"], query["claimed"])]
    if kind == "missing":
        return [(query["subject"], "motto", "")]
    return []


def edge_in_facts(facts: str, edge: tuple[str, str, str], edges: dict) -> bool:
    subject, relation, claimed = edge
    if relation == "motto":
        return f"{subject} motto is" not in facts
    if subject == "*":
        return f"{relation} is {claimed}." in facts or f" {relation} is {claimed}." in facts
    obj = edges.get((subject, relation))
    return obj is not None and f"{subject} {relation} is {obj}." in facts


def classify(greedy: str, answer: str, new_ids: list[int], eos_id: int) -> str:
    if exact(greedy, answer):
        return "exact"
    gold = normalize(answer)
    got = normalize(greedy)
    hit_limit = len(new_ids) >= MAX_NEW and eos_id not in new_ids
    stopped = eos_id in new_ids
    if gold and gold in got:
        return "formato_texto_extra"
    if got and got in gold:
        return "truncado" if hit_limit else "resposta_parcial"
    if hit_limit:
        return "limite_de_geracao"
    if stopped:
        return "eos_cedo"
    return "conteudo_errado"


def audit_row(model, tokenizer, row: dict, facts: str, edges: dict) -> dict:
    prompt = facts + row["prompt"]
    greedy, new_ids = generate_ids(model, tokenizer, prompt, MAX_NEW)
    prompt_ids = ids_of(tokenizer, prompt)
    answer_ids = ids_of(tokenizer, " " + row["answer"])
    edges_needed = needed_edges(row["query"])
    present = [edge_in_facts(facts, edge, edges) for edge in edges_needed]
    reason = classify(greedy, row["answer"], new_ids, int(tokenizer.eos_token_id))
    shift_ok = True
    if prompt_ids and answer_ids:
        device = next(model.parameters()).device
        full = torch.tensor([prompt_ids + answer_ids], device=device)
        with torch.inference_mode():
            logits = model(input_ids=full).logits[0]
        start = len(prompt_ids) - 1
        pred = int(torch.argmax(logits[start]).item())
        shift_ok = start >= 0 and start + len(answer_ids) - 1 < logits.shape[0]
        first_scored = int(answer_ids[0])
    else:
        pred = None
        first_scored = None
    return {
        "category": row["category"],
        "kind": row["kind"],
        "edges": edges_needed,
        "solver": solve(edges, row["query"]),
        "label": row["answer"],
        "solver_matches_label": solve(edges, row["query"]) == row["answer"],
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": answer_ids,
        "max_new_tokens": MAX_NEW,
        "generated_ids": new_ids,
        "generated": greedy,
        "eos_in_generation": int(tokenizer.eos_token_id) in new_ids,
        "hit_limit": len(new_ids) >= MAX_NEW and int(tokenizer.eos_token_id) not in new_ids,
        "edges_in_prompt": present,
        "reason": reason,
        "exact": exact(greedy, row["answer"]),
        "logit_shift_index_ok": shift_ok,
        "first_answer_token": first_scored,
        "argmax_at_shift": pred,
    }


def ladder_item(model, tokenizer, name: str, prompt: str, answer: str) -> dict:
    greedy, new_ids = generate_ids(model, tokenizer, prompt, MAX_NEW)
    return {
        "rung": name,
        "prompt": prompt,
        "answer": answer,
        "generated": greedy,
        "exact": exact(greedy, answer),
        "contains": normalize(answer) in normalize(greedy),
        "generated_ids": new_ids,
        "hit_limit": len(new_ids) >= MAX_NEW and int(tokenizer.eos_token_id) not in new_ids,
        "reason": classify(greedy, answer, new_ids, int(tokenizer.eos_token_id)),
    }


def old_substring(greedy: str, parts: list[str]) -> bool:
    hay = normalize(greedy)
    return all(normalize(part) in hay for part in parts)


def main() -> int:
    configure()
    torch.cuda.init()
    suite = json.loads((OUT / "dev_suite.json").read_text(encoding="utf-8"))
    train = suite["train"]
    eval_rows = suite["eval"]
    edges = make_edges(DEV_SUBJECTS, DEV_COLUMNS)
    facts = facts_block(edges)
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint_check(model) is False:
        raise SystemExit("donor fingerprint mismatch")
    print("REPRODUCE", flush=True)
    audited = []
    counts = {name: {"n": 0, "exact": 0} for name in ("paraphrase", "inverse", "composition", "false_premise")}
    for index, row in enumerate(eval_rows, start=1):
        item = audit_row(model, tokenizer, row, facts, edges)
        audited.append(item)
        counts[row["category"]]["n"] += 1
        counts[row["category"]]["exact"] += int(item["exact"])
        if index % 25 == 0:
            print("scored", index, flush=True)
    print("COUNTS", counts, flush=True)
    expected = {"paraphrase": 47, "inverse": 33, "composition": 3, "false_premise": 5}
    reproduced = all(counts[name]["exact"] == expected[name] and counts[name]["n"] == {"paraphrase": 112, "inverse": 112, "composition": 128, "false_premise": 192}[name] for name in expected)
    errors = {}
    for name in counts:
        failed = [row for row in audited if row["category"] == name and not row["exact"]]
        errors[name] = failed[:20]
    print("LADDER", flush=True)
    ladder = []
    samples = [
        ("copiar", "The answer written here is Lina.\nQuestion: Copy that answer.\nAnswer:", "Lina"),
        ("copiar", "The answer written here is Oslo.\nQuestion: Copy that answer.\nAnswer:", "Oslo"),
        ("copiar", "The answer written here is false.\nQuestion: Copy that answer.\nAnswer:", "false"),
        ("aresta", facts + "Question: Who is the maker of Nex-41?\nAnswer:", "Lina"),
        ("aresta", facts + "Question: In which city is Bolt-7 filed?\nAnswer:", "Oslo"),
        ("aresta", facts + "Question: What color is recorded for Kite-3?\nAnswer:", "ivory"),
        ("inverter", facts + "Question: Which item has maker Lina?\nAnswer:", "Nex-41"),
        ("inverter", facts + "Question: Which item has city Oslo?\nAnswer:", "Bolt-7"),
        ("inverter", facts + "Question: Name the item whose color is ivory.\nAnswer:", "Kite-3"),
        ("duas_arestas", facts + "Question: Name the maker of Nex-41 and the city of Bolt-7.\nAnswer:", "Lina and Oslo"),
        ("duas_arestas", facts + "Question: Name the color of Kite-3 and the tool of Orb-12.\nAnswer:", "ivory and loom"),
        ("duas_arestas", facts + "Question: Give both the maker of Bolt-7 and the city of Nex-41.\nAnswer:", "Omar and Riga"),
        ("falsa", facts + "Question: Is the maker of Nex-41 Omar?\nAnswer:", "false"),
        ("falsa", facts + "Question: Is the city of Bolt-7 Lyon?\nAnswer:", "false"),
        ("falsa", facts + "Question: What motto is recorded for Nex-41?\nAnswer:", "INDETERMINADO"),
    ]
    for name, prompt, answer in samples:
        ladder.append(ladder_item(model, tokenizer, name, prompt, answer))
        print(name, ladder[-1]["exact"], ladder[-1]["generated"][:80].replace("\n", " "), flush=True)
    print("OLD_FORMAT", flush=True)
    old_items = [row for row in json.loads(OLD_SET.read_text(encoding="utf-8"))["items"] if row["category"] == "composition"]
    old_context = "Synthetic record:\n" + "\n".join(OLD_FACTS) + "\n\n"
    old_rows = []
    for item in old_items:
        prompt = old_context + item["prompt"]
        greedy16, ids16 = generate_ids(model, tokenizer, prompt, 16)
        greedy12, ids12 = generate_ids(model, tokenizer, prompt, 12)
        parts = item["acceptable_answers"]
        old_rows.append(
            {
                "prompt": prompt,
                "parts": parts,
                "greedy16": greedy16,
                "substring16": old_substring(greedy16, parts),
                "exact16_vs_joined": exact(greedy16, " and ".join(parts)),
                "greedy12": greedy12,
                "substring12": old_substring(greedy12, parts),
                "hit_limit16": len(ids16) >= 16,
            }
        )
    new_on_old_matcher = []
    for row in [item for item in eval_rows if item["category"] == "composition"][:8]:
        prompt = old_context_style(facts, row["question"])
        greedy, new_ids = generate_ids(model, tokenizer, prompt, 16)
        parts = row["answer"].split(" and ")
        new_on_old_matcher.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "greedy16": greedy,
                "substring": old_substring(greedy, parts),
                "exact12_style": exact(greedy, row["answer"]),
                "hit_limit": len(new_ids) >= 16,
            }
        )
    train_atoms = {(row["query"]["subject"], row["query"]["relation"]) for row in train if row["kind"] == "atomic"}
    missing_train = []
    supervised = 0
    for row in train:
        supervised += len(ids_of(tokenizer, " " + row["answer"]))
    for row in eval_rows:
        if row["kind"] != "composition":
            continue
        query = row["query"]
        pair = [(query["subject_a"], query["relation_a"]), (query["subject_b"], query["relation_b"])]
        if any(edge not in train_atoms for edge in pair):
            missing_train.append(pair)
    LOG.mkdir(parents=True, exist_ok=True)
    payload = {
        "reproduced": reproduced,
        "counts": counts,
        "errors": errors,
        "ladder": ladder,
        "old_rows": old_rows,
        "new_on_old_matcher": new_on_old_matcher,
        "composition_eval": sum(1 for row in eval_rows if row["kind"] == "composition"),
        "composition_missing_train_edge": len(missing_train),
        "train_examples": len(train),
        "train_supervised_tokens_one_pass": supervised,
        "train_supervised_tokens_two_epochs": supervised * 2,
        "kl_tokens_old_diagnostic": 136,
        "fingerprint": PHASE2_FINGERPRINT,
    }
    (LOG / "control_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    decision = decide(counts, ladder, reproduced)
    write_report(payload, decision)
    print(decision)
    print("AUDIT_DONE")
    return 0


def bare_donor_fingerprint_check(model) -> bool:
    from darwin_cl.plastic.bank import bare_donor_fingerprint

    return bare_donor_fingerprint(model) == PHASE2_FINGERPRINT


def old_context_style(facts: str, question: str) -> str:
    lines = [line for line in facts.splitlines() if line and not line.startswith("Known")]
    return "Synthetic record:\n" + "\n".join(lines) + "\n\n" + question


def decide(counts: dict, ladder: list[dict], reproduced: bool) -> str:
    if not reproduced:
        return "EVAL_BUG"
    by_rung: dict[str, list[bool]] = {}
    for row in ladder:
        by_rung.setdefault(row["rung"], []).append(row["exact"])
    copy_ok = all(by_rung.get("copiar", [False]))
    edge_ok = all(by_rung.get("aresta", [False]))
    contains_two = [row["contains"] for row in ladder if row["rung"] == "duas_arestas"]
    exact_two = [row["exact"] for row in ladder if row["rung"] == "duas_arestas"]
    if copy_ok and edge_ok and any(contains_two) and not all(exact_two):
        return "PROMPT_OR_FORMAT_FAIL"
    if copy_ok and edge_ok and not any(contains_two):
        return "DONOR_CAPACITY_LIMIT"
    if all(exact_two) and all(by_rung.get("inverter", [False])) and all(by_rung.get("falsa", [False])):
        return "CONTROL_PASS"
    if counts["composition"]["exact"] <= 3 and not all(exact_two):
        return "PROMPT_OR_FORMAT_FAIL" if any(contains_two) else "DONOR_CAPACITY_LIMIT"
    return "PROMPT_OR_FORMAT_FAIL"


def write_report(payload: dict, decision: str) -> None:
    counts = payload["counts"]
    lines = [
        "# RELATIONAL CONTROL AUDIT",
        "",
        f"Decisao: **{decision}**.",
        "",
        "Sem treino. ARM 3 nao executado. Teste final selado nao consultado.",
        "",
        "## Reproducao do donor com fatos",
        "",
        f"Reproducao exata dos publicados: {payload['reproduced']}.",
        "",
        "| Categoria | Exato |",
        "|---|---|",
    ]
    for name in ("paraphrase", "inverse", "composition", "false_premise"):
        row = counts[name]
        lines.append(f"| {name} | {row['exact']}/{row['n']} |")
    lines.extend(["", "## Vinte erros por categoria", ""])
    for name, rows in payload["errors"].items():
        lines.append(f"### {name}")
        lines.append("")
        for row in rows:
            lines.append(f"- Motivo: {row['reason']}. Solver: {row['solver']}. Label: {row['label']}. Arestas no prompt: {row['edges_in_prompt']}.")
            lines.append(f"  Prompt: {row['prompt'][:400].replace(chr(10), ' | ')}")
            lines.append(f"  Tokens de entrada: {row['prompt_tokens']}. Limite: {row['max_new_tokens']}. EOS: {row['eos_in_generation']}.")
            lines.append(f"  Saida: {row['generated'][:180].replace(chr(10), ' ')}")
            lines.append(f"  Shift ok: {row['logit_shift_index_ok']}. Primeiro token da resposta: {row['first_answer_token']}.")
        lines.append("")
    lines.extend(["## Escada, mesmo avaliador", "", "| Degrau | Exato | Saida |", "|---|---|---|"])
    for row in payload["ladder"]:
        lines.append(f"| {row['rung']} | {row['exact']} | {row['generated'][:120].replace(chr(10), ' ')} |")
    lines.extend(["", "## Formato antigo e formato novo", ""])
    for row in payload["old_rows"]:
        lines.append(f"- Substring 16: {row['substring16']}. Exato contra 'A and B': {row['exact16_vs_joined']}.")
        lines.append(f"  {row['greedy16'][:180].replace(chr(10), ' ')}")
    lines.append("")
    for row in payload["new_on_old_matcher"]:
        lines.append(f"- Nova pergunta no formato antigo, substring: {row['substring']}, exact: {row['exact12_style']}.")
        lines.append(f"  {row['greedy16'][:180].replace(chr(10), ' ')}")
    lines.extend(
        [
            "",
            "## Arestas e tokens",
            "",
            f"Composicoes avaliadas sem fato extra no prompt de treino: {payload['composition_eval']}.",
            f"Dessas, com alguma aresta atomica ausente do treino: {payload['composition_missing_train_edge']}.",
            f"Exemplos de treino: {payload['train_examples']}.",
            f"Tokens supervisionados da cross-entropy, uma passagem: {payload['train_supervised_tokens_one_pass']}.",
            f"Duas epocas, o orcamento dos bracos: {payload['train_supervised_tokens_two_epochs']}.",
            f"Os 136 tokens sao a KL antiga do SDFT, nao esta perda.",
            "",
            "PARAR. Sem novo treino.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
