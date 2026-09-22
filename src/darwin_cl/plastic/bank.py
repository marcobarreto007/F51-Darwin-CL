"""Removable residual expert branch. No optimizer and no new-domain training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from darwin_cl.donor.baseline import DONOR_ID, DONOR_REVISION, weight_fingerprint

INSERTION_LAYER = 13
N_EXPERTS = 8
TOP_K = 2
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
RMS_EPS = 1e-6


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden))
        return self.down_proj(gate * self.up_proj(hidden))


def parameter_free_rms_norm(hidden: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Same reduction as Qwen3RMSNorm, without a trainable weight."""
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    return (hidden.float() * torch.rsqrt(variance + eps)).to(dtype=hidden.dtype)


class PlasticBank(nn.Module):
    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = INTERMEDIATE_SIZE,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()
        if top_k != TOP_K or n_experts != N_EXPERTS:
            raise ValueError("V0 is fixed at 8 experts and top-k 2")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            SwiGLUExpert(hidden_size, intermediate_size) for _ in range(n_experts)
        )
        self.router = nn.Linear(hidden_size, n_experts, bias=False)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def delta(self, base_hidden: torch.Tensor) -> torch.Tensor:
        if base_hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden size does not match the bank")
        normalized = parameter_free_rms_norm(base_hidden)
        logits = self.router(normalized)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(top_values.float(), dim=-1).to(dtype=normalized.dtype)
        expert_out = torch.stack([expert(normalized) for expert in self.experts], dim=-2)
        gather_index = top_indices.unsqueeze(-1).expand(*top_indices.shape, self.hidden_size)
        selected = torch.gather(expert_out, dim=-2, index=gather_index)
        return (selected * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, base_hidden: torch.Tensor) -> torch.Tensor:
        delta = self.delta(base_hidden)
        return base_hidden + (self.alpha * delta).to(dtype=base_hidden.dtype)


class ResidualPlasticLayer(nn.Module):
    """Donor layer, then a removable residual bank. Does not edit Transformers."""

    def __init__(self, donor_layer: nn.Module, bank: PlasticBank) -> None:
        super().__init__()
        self.donor_layer = donor_layer
        self.branch = bank
        self.attention_type = donor_layer.attention_type

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        base = self.donor_layer(hidden_states, **kwargs)
        if not torch.is_tensor(base):
            raise TypeError("Qwen3DecoderLayer was expected to return a tensor")
        return self.branch(base)


def freeze_donor(model: nn.Module) -> int:
    frozen = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    return frozen


def install_branch(model: nn.Module, bank: PlasticBank, layer_index: int = INSERTION_LAYER) -> ResidualPlasticLayer:
    layers = model.model.layers
    current = layers[layer_index]
    if isinstance(current, ResidualPlasticLayer):
        raise RuntimeError("a plastic branch is already installed")
    wrapper = ResidualPlasticLayer(current, bank)
    wrapper.to(device=next(model.parameters()).device)
    bank.to(dtype=next(model.parameters()).dtype)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    layers[layer_index] = wrapper
    for parameter in bank.parameters():
        parameter.requires_grad_(True)
    return wrapper


def remove_branch(model: nn.Module, layer_index: int = INSERTION_LAYER) -> nn.Module:
    layers = model.model.layers
    current = layers[layer_index]
    if not isinstance(current, ResidualPlasticLayer):
        raise RuntimeError("no plastic branch is installed")
    layers[layer_index] = current.donor_layer
    return current.donor_layer


def donor_parameter_pairs(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    pairs = []
    for name, parameter in model.named_parameters():
        if ".branch." in name:
            continue
        pairs.append((name.replace(".donor_layer.", "."), parameter))
    return pairs


def donor_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in donor_parameter_pairs(model):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def bare_donor_fingerprint(model: nn.Module) -> str:
    if any(isinstance(layer, ResidualPlasticLayer) for layer in model.model.layers):
        raise RuntimeError("bare fingerprint requires the branch to be absent")
    return weight_fingerprint(model)


def parameter_report(model: nn.Module) -> dict[str, dict[str, int | bool]]:
    groups = {
        "donor": {"count": 0, "trainable": 0},
        "experts": {"count": 0, "trainable": 0},
        "router": {"count": 0, "trainable": 0},
        "alpha": {"count": 0, "trainable": 0},
    }
    for name, parameter in model.named_parameters():
        if ".branch.experts." in name:
            key = "experts"
        elif ".branch.router." in name:
            key = "router"
        elif name.endswith(".branch.alpha"):
            key = "alpha"
        else:
            key = "donor"
        groups[key]["count"] += parameter.numel()
        if parameter.requires_grad:
            groups[key]["trainable"] += parameter.numel()
    return groups


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype is torch.bfloat16:
        return cpu.view(torch.uint8).numpy().tobytes()
    return cpu.numpy().tobytes()


def branch_content_sha256(metadata: dict[str, Any], state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(str(state[key].dtype).encode("utf-8"))
        digest.update(str(tuple(state[key].shape)).encode("utf-8"))
        digest.update(_tensor_bytes(state[key]))
    return digest.hexdigest()


def save_branch_checkpoint(
    path: str | Path,
    bank: PlasticBank,
    *,
    donor_fingerprint_hex: str,
    seed: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "darwin_cl_plastic_branch_v0",
        "donor_id": DONOR_ID,
        "donor_revision": DONOR_REVISION,
        "donor_fingerprint": donor_fingerprint_hex,
        "insertion_layer": INSERTION_LAYER,
        "hidden_size": bank.hidden_size,
        "intermediate_size": bank.intermediate_size,
        "n_experts": bank.n_experts,
        "top_k": bank.top_k,
        "seed": seed,
        "policy": "EXACT_SCORED_UTF8",
    }
    state = {key: value.detach().cpu() for key, value in bank.state_dict().items()}
    payload = {
        "metadata": metadata,
        "state_dict": state,
        "sha256": branch_content_sha256(metadata, state),
    }
    torch.save(payload, destination)
    return destination


class BranchCheckpointError(ValueError):
    pass


def load_branch_checkpoint(
    path: str | Path,
    *,
    donor_fingerprint_hex: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PlasticBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    state = payload["state_dict"]
    expected = branch_content_sha256(metadata, state)
    if payload.get("sha256") != expected:
        raise BranchCheckpointError("checkpoint sha256 does not match its contents")
    if metadata.get("donor_revision") != DONOR_REVISION:
        raise BranchCheckpointError("donor revision does not match the pinned donor")
    if metadata.get("donor_fingerprint") != donor_fingerprint_hex:
        raise BranchCheckpointError("donor fingerprint does not match the loaded donor")
    bank = PlasticBank(
        hidden_size=int(metadata["hidden_size"]),
        intermediate_size=int(metadata["intermediate_size"]),
        n_experts=int(metadata["n_experts"]),
        top_k=int(metadata["top_k"]),
    )
    bank.load_state_dict(state)
    bank.to(device=device, dtype=dtype)
    bank.alpha.data = bank.alpha.data.to(dtype=torch.float32)
    return bank
