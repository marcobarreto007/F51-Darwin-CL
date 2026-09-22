"""Frozen donor baseline. Scores the versioned suite twice and writes artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darwin_cl.donor.baseline import (  # noqa: E402
    DONOR_ID,
    DONOR_REVISION,
    generate,
    load_donor,
    save_reference_outputs,
    score_loss,
    weight_fingerprint,
)

SUITE = ROOT / "benchmarks" / "baseline" / "v1"
OUT = ROOT / "artifacts" / "baseline" / "v1" / "reference.json"
DOMAINS = [
    "A_language.jsonl",
    "B_code.jsonl",
    "C_mathematics.jsonl",
    "D_factual.jsonl",
    "E_reasoning.jsonl",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def configure_determinism() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def main() -> int:
    configure_determinism()
    device = "cuda:0"
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model, tokenizer = load_donor(device=device, dtype=torch.bfloat16)
    before = weight_fingerprint(model)
    domains = []
    for name in DOMAINS:
        path = SUITE / name
        rows = read_jsonl(path)
        examples = []
        total_nll = 0.0
        total_bytes = 0
        total_tokens = 0
        total_scored = 0
        for row in rows:
            first = score_loss(model, tokenizer, row["text"])
            second = score_loss(model, tokenizer, row["text"])
            if first["total_nll"] != second["total_nll"]:
                raise SystemExit(f"NLL not reproducible for {row['id']}")
            greedy_a = generate(model, tokenizer, row["prompt"])
            greedy_b = generate(model, tokenizer, row["prompt"])
            if greedy_a != greedy_b:
                raise SystemExit(f"greedy output not reproducible for {row['id']}")
            examples.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "greedy": greedy_a,
                    **first,
                }
            )
            total_nll += first["total_nll"]
            total_bytes += first["evaluated_utf8_bytes"]
            total_tokens += first["token_count"]
            total_scored += first["scored_tokens"]
        domains.append(
            {
                "file": name,
                "sha256": file_sha256(path),
                "documents": len(rows),
                "token_count": total_tokens,
                "scored_tokens": total_scored,
                "evaluated_utf8_bytes": total_bytes,
                "total_nll": total_nll,
                "nats_per_token": total_nll / total_scored,
                "nats_per_byte": total_nll / total_bytes,
                "bits_per_byte": (total_nll / total_bytes) / torch.log(torch.tensor(2.0)).item(),
                "examples": examples,
            }
        )
    after = weight_fingerprint(model)
    if before != after:
        raise SystemExit("weights changed during baseline")
    wall = time.perf_counter() - started
    payload = {
        "gate": "DONOR_BASELINE_PASS",
        "donor_id": DONOR_ID,
        "revision": DONOR_REVISION,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "bfloat16",
        "device": device,
        "seed": 0,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        "wall_seconds": wall,
        "weight_sha256_before": before,
        "weight_sha256_after": after,
        "training": False,
        "domains": domains,
    }
    save_reference_outputs(payload, OUT)
    print(json.dumps({k: payload[k] for k in payload if k != "domains"}, indent=2))
    for domain in domains:
        print(
            domain["file"],
            "nats/byte",
            round(domain["nats_per_byte"], 6),
            "bits/byte",
            round(domain["bits_per_byte"], 6),
            "sha",
            domain["sha256"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
