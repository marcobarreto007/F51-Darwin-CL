"""New process: load the donor plus one plastic checkpoint and score SRB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from darwin_cl.donor.baseline import load_donor  # noqa: E402
from darwin_cl.plastic.bank import (  # noqa: E402
    donor_fingerprint,
    install_branch,
    load_branch_checkpoint,
)
from run_phase4_srb import configure, score_item  # noqa: E402

PHASE2_FINGERPRINT = "d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175"


def main() -> int:
    checkpoint = Path(sys.argv[1])
    questions = Path(sys.argv[2])
    destination = Path(sys.argv[3])
    configure()
    model, tokenizer = load_donor()
    bank = load_branch_checkpoint(
        checkpoint,
        donor_fingerprint_hex=PHASE2_FINGERPRINT,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    install_branch(model, bank)
    fingerprint = donor_fingerprint(model)
    if fingerprint != PHASE2_FINGERPRINT:
        raise SystemExit("reloaded donor fingerprint mismatch")
    payload = json.loads(questions.read_text(encoding="utf-8"))
    items = payload["rows"]
    scored = [score_item(model, tokenizer, item) for item in items]
    correct = sum(1 for row in scored if row["correct"])
    destination.write_text(
        json.dumps(
            {
                "donor_fingerprint": fingerprint,
                "checkpoint": str(checkpoint),
                "accuracy": correct / len(scored),
                "correct": correct,
                "count": len(scored),
                "items": scored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)
    print("accuracy", correct / len(scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
