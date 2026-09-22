"""New process: donor plus best_8of8.pt must show 8/8 and the same fingerprint."""

from __future__ import annotations

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
from run_phase4_srb import PHASE2_FINGERPRINT, configure  # noqa: E402
from run_stage_x0s import build_canaries, view_all  # noqa: E402


def main() -> int:
    configure()
    path = Path(sys.argv[1])
    model, tokenizer = load_donor()
    bank = load_branch_checkpoint(path, donor_fingerprint_hex=PHASE2_FINGERPRINT, device="cuda:0", dtype=torch.bfloat16)
    bank.router.weight.requires_grad_(False)
    bank.alpha.requires_grad_(False)
    install_branch(model, bank)
    if donor_fingerprint(model) != PHASE2_FINGERPRINT:
        print("FINGERPRINT_MISMATCH")
        return 2
    ranks = [item["rank"] for item in view_all(model, tokenizer, build_canaries(tokenizer))]
    print("ranks", ranks)
    if ranks != [1, 1, 1, 1, 1, 1, 1, 1]:
        print("NOT_8_OF_8")
        return 2
    print("VERIFIED_8_OF_8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
