import tempfile
import unittest
from pathlib import Path

import torch

from darwin_cl.donor.baseline import DONOR_REVISION, load_donor
from darwin_cl.plastic.bank import (
    INSERTION_LAYER,
    N_EXPERTS,
    TOP_K,
    BranchCheckpointError,
    PlasticBank,
    ResidualPlasticLayer,
    bare_donor_fingerprint,
    donor_fingerprint,
    freeze_donor,
    install_branch,
    load_branch_checkpoint,
    parameter_report,
    remove_branch,
    save_branch_checkpoint,
)

PHASE2_FINGERPRINT = "d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175"


def _grad_abs(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum().item())


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_rejects_wrong_donor_and_bad_hash(self) -> None:
        torch.manual_seed(0)
        bank = PlasticBank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            save_branch_checkpoint(path, bank, donor_fingerprint_hex=PHASE2_FINGERPRINT, seed=0)
            with self.assertRaises(BranchCheckpointError):
                load_branch_checkpoint(
                    path,
                    donor_fingerprint_hex="0" * 64,
                    device="cpu",
                    dtype=torch.float32,
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["metadata"]["donor_revision"] = "not-the-pinned-revision"
            bad = Path(directory) / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaises(BranchCheckpointError):
                load_branch_checkpoint(
                    bad,
                    donor_fingerprint_hex=PHASE2_FINGERPRINT,
                    device="cpu",
                    dtype=torch.float32,
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["sha256"] = "0" * 64
            hashed = Path(directory) / "hashed.pt"
            torch.save(payload, hashed)
            with self.assertRaises(BranchCheckpointError):
                load_branch_checkpoint(
                    hashed,
                    donor_fingerprint_hex=PHASE2_FINGERPRINT,
                    device="cpu",
                    dtype=torch.float32,
                )
            loaded = load_branch_checkpoint(
                path,
                donor_fingerprint_hex=PHASE2_FINGERPRINT,
                device="cpu",
                dtype=torch.float32,
            )
            self.assertEqual(loaded.n_experts, 8)
            self.assertEqual(loaded.top_k, 2)


class PlasticBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model, cls.tokenizer = load_donor(device="cuda:0", dtype=torch.bfloat16)
        cls.bare = bare_donor_fingerprint(cls.model)
        freeze_donor(cls.model)
        cls.bank = PlasticBank().to(device="cuda:0", dtype=torch.bfloat16)
        cls.bank.alpha.data = cls.bank.alpha.data.to(dtype=torch.float32)
        install_branch(cls.model, cls.bank, INSERTION_LAYER)

    @classmethod
    def tearDownClass(cls) -> None:
        if isinstance(cls.model.model.layers[INSERTION_LAYER], ResidualPlasticLayer):
            remove_branch(cls.model, INSERTION_LAYER)

    def test_donor_revision_and_fingerprint(self) -> None:
        self.assertEqual(DONOR_REVISION, "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
        self.assertEqual(self.bare, PHASE2_FINGERPRINT)
        self.assertEqual(donor_fingerprint(self.model), PHASE2_FINGERPRINT)

    def test_shape_dtype_device_and_counts(self) -> None:
        self.assertEqual(len(self.bank.experts), N_EXPERTS)
        self.assertEqual(self.bank.top_k, TOP_K)
        hidden = torch.randn(2, 5, 1024, device="cuda:0", dtype=torch.bfloat16)
        delta = self.bank.delta(hidden)
        self.assertEqual(tuple(delta.shape), (2, 5, 1024))
        self.assertEqual(delta.dtype, torch.bfloat16)
        self.assertEqual(delta.device.type, "cuda")
        self.assertEqual(self.bank.alpha.dtype, torch.float32)
        report = parameter_report(self.model)
        self.assertEqual(report["donor"]["trainable"], 0)
        self.assertGreater(report["experts"]["trainable"], 0)
        self.assertGreater(report["router"]["trainable"], 0)
        self.assertEqual(report["alpha"]["trainable"], 1)

    def test_alpha_zero_blocks_branch_gradients_and_small_alpha_opens_them(self) -> None:
        self.bank.alpha.data.zero_()
        self.model.zero_grad(set_to_none=True)
        ids = self.tokenizer("The sum of 17 and 28 is 45.", return_tensors="pt").input_ids.to("cuda:0")
        self.model(input_ids=ids, labels=ids).loss.backward()
        expert_grad = sum(_grad_abs(parameter) for expert in self.bank.experts for parameter in expert.parameters())
        self.assertEqual(expert_grad, 0.0)
        self.assertEqual(_grad_abs(self.bank.router.weight), 0.0)
        self.assertGreater(_grad_abs(self.bank.alpha), 0.0)
        for name, parameter in self.model.named_parameters():
            if ".branch." in name:
                continue
            self.assertIsNone(parameter.grad, name)

        before = [parameter.detach().clone() for parameter in self.bank.experts.parameters()]
        router_before = self.bank.router.weight.detach().clone()
        self.model.zero_grad(set_to_none=True)
        self.bank.alpha.data.fill_(1e-4)
        self.model(input_ids=ids, labels=ids).loss.backward()
        expert_grad = sum(_grad_abs(parameter) for expert in self.bank.experts for parameter in expert.parameters())
        self.assertGreater(expert_grad, 0.0)
        self.assertGreater(_grad_abs(self.bank.router.weight), 0.0)
        for name, parameter in self.model.named_parameters():
            if ".branch." in name:
                continue
            self.assertIsNone(parameter.grad, name)
        for parameter, saved in zip(self.bank.experts.parameters(), before):
            self.assertTrue(torch.equal(parameter, saved))
        self.assertTrue(torch.equal(self.bank.router.weight, router_before))
        self.bank.alpha.data.zero_()

    def test_removal_restores_the_same_logits(self) -> None:
        self.bank.alpha.data.zero_()
        ids = self.tokenizer("Paris is the capital of France.", return_tensors="pt").input_ids.to("cuda:0")
        with torch.inference_mode():
            wrapped = self.model(input_ids=ids).logits
        remove_branch(self.model, INSERTION_LAYER)
        try:
            with torch.inference_mode():
                bare = self.model(input_ids=ids).logits
        finally:
            install_branch(self.model, self.bank, INSERTION_LAYER)
        self.assertTrue(torch.equal(wrapped, bare))


if __name__ == "__main__":
    unittest.main()
