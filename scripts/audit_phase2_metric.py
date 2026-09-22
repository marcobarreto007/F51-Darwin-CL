"""Metric audit. Does not write artifacts/baseline/v1/reference.json."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darwin_cl.donor.baseline import (  # noqa: E402
    DONOR_ID,
    DONOR_REVISION,
    explain_score,
    generate,
    load_donor,
    weight_fingerprint,
)

SUITE = ROOT / "benchmarks" / "baseline" / "v1"
DOMAINS = [
    "A_language.jsonl",
    "B_code.jsonl",
    "C_mathematics.jsonl",
    "D_factual.jsonl",
    "E_reasoning.jsonl",
]
PROBE = "café ação 日本"


def configure() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure()
    torch.cuda.init()
    model, tokenizer = load_donor()
    if model.training:
        raise SystemExit("model.train() is on")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("a parameter requires grad")
    before = weight_fingerprint(model)
    domains = []
    for name in DOMAINS:
        rows = read_jsonl(SUITE / name)
        examples = []
        for row in rows:
            detail = explain_score(model, tokenizer, row["text"])
            if abs(sum(detail["scored_token_nlls"]) - detail["total_nll"]) > 1e-6:
                raise SystemExit(f"NLL sum mismatch {row['id']}")
            if abs(detail["total_nll"] / detail["scored_utf8_bytes"] - detail["nats_per_byte"]) > 1e-12:
                raise SystemExit(f"nats/byte mismatch {row['id']}")
            greedy = generate(model, tokenizer, row["prompt"])
            examples.append(
                {
                    "id": row["id"],
                    "raw_characters": detail["raw_characters"],
                    "RAW_UTF8_BYTES": detail["raw_utf8_bytes"],
                    "TOKEN_IDS": detail["token_ids"],
                    "TOKEN_COUNT": detail["token_count"],
                    "SCORED_TOKEN_IDS": detail["scored_token_ids"],
                    "SCORED_TOKEN_COUNT": detail["scored_token_count"],
                    "TOTAL_NLL": detail["total_nll"],
                    "SCORED_UTF8_BYTES": detail["scored_utf8_bytes"],
                    "NATS_PER_TOKEN": detail["nats_per_token"],
                    "NATS_PER_BYTE": detail["nats_per_byte"],
                    "BITS_PER_BYTE": detail["bits_per_byte"],
                    "context_token_id": detail["context_token_id"],
                    "context_text": detail["context_text"],
                    "greedy": greedy,
                }
            )
        nats = [example["NATS_PER_BYTE"] for example in examples]
        domains.append(
            {
                "file": name,
                "documents": len(examples),
                "raw_characters": sum(example["raw_characters"] for example in examples),
                "utf8_bytes": sum(example["RAW_UTF8_BYTES"] for example in examples),
                "tokens": sum(example["TOKEN_COUNT"] for example in examples),
                "scored_tokens": sum(example["SCORED_TOKEN_COUNT"] for example in examples),
                "scored_utf8_bytes": sum(example["SCORED_UTF8_BYTES"] for example in examples),
                "total_nll": sum(example["TOTAL_NLL"] for example in examples),
                "mean_nats_per_byte": statistics.fmean(nats),
                "std_nats_per_byte": statistics.pstdev(nats),
                "min_nats_per_byte": min(nats),
                "max_nats_per_byte": max(nats),
                "examples": examples,
            }
        )
    probe = explain_score(model, tokenizer, PROBE)
    after = weight_fingerprint(model)
    if before != after:
        raise SystemExit("fingerprint changed")
    payload = {
        "policy": "EXACT_SCORED_UTF8",
        "donor_id": DONOR_ID,
        "revision": DONOR_REVISION,
        "weight_sha256_before": before,
        "weight_sha256_after": after,
        "requires_grad": False,
        "backward": False,
        "optimizer": False,
        "training": False,
        "v1_overwritten": False,
        "probe_text": PROBE,
        "probe_python_characters": len(PROBE),
        "probe_raw_utf8_bytes": len(PROBE.encode("utf-8")),
        "probe": {
            "TOKEN_IDS": probe["token_ids"],
            "TOKEN_COUNT": probe["token_count"],
            "SCORED_TOKEN_IDS": probe["scored_token_ids"],
            "SCORED_TOKEN_COUNT": probe["scored_token_count"],
            "TOTAL_NLL": probe["total_nll"],
            "sum_token_nll": sum(probe["scored_token_nlls"]),
            "RAW_UTF8_BYTES": probe["raw_utf8_bytes"],
            "SCORED_UTF8_BYTES": probe["scored_utf8_bytes"],
            "NATS_PER_BYTE": probe["nats_per_byte"],
            "context_text": probe["context_text"],
        },
        "domains": domains,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    print("fingerprint", before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
