"""Score only the answer span. Does not train and does not open the sealed test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from darwin_cl.eval.relational_answer import verdict  # noqa: E402
from darwin_cl.plastic.bank import bare_donor_fingerprint  # noqa: E402
from run_phase4_srb import PHASE2_FINGERPRINT, configure  # noqa: E402
from run_unblock_relational import (  # noqa: E402
    DEV_COLUMNS,
    DEV_SUBJECTS,
    OUT,
    exact,
    facts_block,
    make_edges,
)

REPORT = ROOT / "reports" / "RELATIONAL_SCORE_FIX.md"
MAX_NEW = 24
STOPS = ("\n", "Question:")


class StopAtSpan(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_ids = input_ids[0, self.prompt_len :]
        if new_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return any(stop in text for stop in STOPS)


def answer_span(text: str) -> str:
    cut = text
    for stop in STOPS:
        if stop in cut:
            cut = cut.split(stop, 1)[0]
    return cut.strip()


def generate_span(model, tokenizer, prompt: str) -> dict:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopAtSpan(tokenizer, prompt_len)]),
        )
    new_ids = [int(token) for token in output[0, prompt_len:].tolist()]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    span = answer_span(raw)
    return {
        "raw": raw,
        "span": span,
        "new_tokens": len(new_ids),
        "stopped": any(stop in raw for stop in STOPS) or tokenizer.eos_token_id in new_ids,
        "hit_limit": len(new_ids) >= MAX_NEW and tokenizer.eos_token_id not in new_ids and not any(stop in raw for stop in STOPS),
    }


def main() -> int:
    configure()
    torch.cuda.init()
    suite = json.loads((OUT / "dev_suite.json").read_text(encoding="utf-8"))
    rows = suite["eval"]
    facts = facts_block(make_edges(DEV_SUBJECTS, DEV_COLUMNS))
    model, tokenizer = load_donor()
    model.eval()
    if bare_donor_fingerprint(model) != PHASE2_FINGERPRINT:
        raise SystemExit("donor fingerprint mismatch")
    counts = {name: {"n": 0, "exact": 0, "limit": 0} for name in ("paraphrase", "inverse", "composition", "false_premise")}
    examples = {name: [] for name in counts}
    kind_counts = {name: {"n": 0, "exact": 0, "correct": 0, "incorrect": 0, "review": 0, "limit": 0} for name in ("atomic", "inverse", "composition", "false_claim", "missing")}
    predictions = []
    print("SPAN", flush=True)
    for index, row in enumerate(rows, start=1):
        generated = generate_span(model, tokenizer, facts + row["prompt"])
        hit = exact(generated["span"], row["answer"])
        bucket = counts[row["category"]]
        bucket["n"] += 1
        bucket["exact"] += int(hit)
        bucket["limit"] += int(generated["hit_limit"])
        decision = verdict(row, generated["span"])
        kind_bucket = kind_counts[row["kind"]]
        kind_bucket["n"] += 1
        kind_bucket["exact"] += int(hit)
        kind_bucket[decision] += 1
        kind_bucket["limit"] += int(generated["hit_limit"])
        predictions.append({"index": index, "kind": row["kind"], "category": row["category"], "question": row["question"], "query": row["query"], "answer": row["answer"], **generated, "exact": hit, "verdict": decision})
        if (not hit) and len(examples[row["category"]]) < 5:
            examples[row["category"]].append(
                {"answer": row["answer"], "span": generated["span"], "raw": generated["raw"][:180], "limit": generated["hit_limit"]}
            )
        if index % 25 == 0:
            print("scored", index, flush=True)
    ladder = []
    probes = [
        ("copiar", "The answer written here is Lina.\nQuestion: Copy that answer.\nAnswer:", "Lina"),
        ("aresta", facts + "Question: Who is the maker of Nex-41?\nAnswer:", "Lina"),
        ("inverter", facts + "Question: Which item has maker Lina?\nAnswer:", "Nex-41"),
        ("duas_arestas", facts + "Question: Name the maker of Nex-41 and the city of Bolt-7.\nAnswer:", "Lina and Oslo"),
        ("falsa", facts + "Question: Is the maker of Nex-41 Omar?\nAnswer:", "false"),
    ]
    for name, prompt, answer in probes:
        generated = generate_span(model, tokenizer, prompt)
        ladder.append({"rung": name, "answer": answer, "span": generated["span"], "exact": exact(generated["span"], answer), "raw": generated["raw"][:180]})
        print(name, ladder[-1]["exact"], generated["span"][:80].replace("\n", " "), flush=True)
    assert len(predictions) == len(rows)
    assert sum(bucket["n"] for bucket in kind_counts.values()) == len(rows)
    payload = {"counts": counts, "kind_counts": kind_counts, "predictions": predictions, "examples": examples, "ladder": ladder, "max_new_tokens": MAX_NEW, "stops": list(STOPS)}
    (OUT / "score_fix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RELATIONAL SCORE FIX",
        "",
        "A geracao para na primeira quebra de linha ou em `Question:`. O exact match usa so esse trecho. Limite 24 tokens. Suite historica e teste selado nao foram alterados. Sem treino.",
        "",
        "| Categoria | Exact antigo, geracao inteira, 12 tokens | Exact do trecho, parada explicita |",
        "|---|---|---|",
        "| paraphrase | 47/112 | {p}/{n} |".format(p=counts["paraphrase"]["exact"], n=counts["paraphrase"]["n"]),
        "| inverse | 33/112 | {p}/{n} |".format(p=counts["inverse"]["exact"], n=counts["inverse"]["n"]),
        "| composition | 3/128 | {p}/{n} |".format(p=counts["composition"]["exact"], n=counts["composition"]["n"]),
        "| false premise | 5/192 | {p}/{n} |".format(p=counts["false_premise"]["exact"], n=counts["false_premise"]["n"]),
        "",
        "## Tipos de pergunta (auditoria conservadora)",
        "",
        "Os acertos conservadores sao apenas um limite inferior; `review` requer leitura humana. `false_claim` e `missing` tinham sido agregados na mesma categoria historica.",
        "",
        "| Tipo | Exact | Correto automatico | Incorreto automatico | Revisar | Limite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, bucket in kind_counts.items():
        lines.append(f"| {kind} | {bucket['exact']}/{bucket['n']} | {bucket['correct']} | {bucket['incorrect']} | {bucket['review']} | {bucket['limit']} |")
    lines.extend([
        "",
        "## Escada",
        "",
        "| Degrau | Exato | Trecho |",
        "|---|---|---|",
    ])
    for row in ladder:
        lines.append(f"| {row['rung']} | {row['exact']} | {row['span'][:120].replace(chr(10), ' ')} |")
    lines.extend(["", "## Erros que continuam", ""])
    for name, rows_left in examples.items():
        lines.append(f"### {name}")
        lines.append("")
        for row in rows_left:
            lines.append(f"- Esperado `{row['answer']}`. Trecho `{row['span'][:160].replace(chr(10), ' ')}`. Limite: {row['limit']}.")
        lines.append("")
    lines.append("PARAR. Sem novo treino.")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(counts)
    print(kind_counts)
    print("SCORE_FIX_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
