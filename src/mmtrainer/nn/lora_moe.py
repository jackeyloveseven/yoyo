"""Native sparse LoRA-MoE routing and PEFT adapter switching components."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_BLOCK_RE = re.compile(r"(?:^|[.$_])blocks(?:[.$_]+)(\d+)(?:[.$_]|$)")


@dataclass(frozen=True)
class LoRAExpertSpec:
    """Configuration for one frozen or routable LoRA expert."""

    name: str
    path: str
    strength: float = 0.75

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LoRAExpertSpec:
        name = str(value.get("name", "")).strip()
        path = str(value.get("path", "")).strip()
        strength = float(value.get("strength", 0.75))
        if not name or name == "base":
            raise ValueError("expert name must be non-empty and not 'base'")
        if not path:
            raise ValueError(f"expert {name!r} has no path")
        if not 0.0 < strength < 1.5:
            raise ValueError(f"expert {name!r} strength must be in (0, 1.5)")
        return cls(name=name, path=path, strength=strength)


@dataclass(frozen=True)
class NativeLoRAMoEConfig:
    """Configuration for block-grouped context and timestep LoRA-MoE routing."""

    enabled: bool
    experts: tuple[LoRAExpertSpec, ...]
    context_dim: int = 2560
    hidden_dim: int = 512
    num_blocks: int = 28
    block_group_size: int = 4
    top_k: int = 1
    temperature: float = 1.0
    router_noise: bool = True
    balance_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    strength_loss_weight: float = 0.001
    route_prior: tuple[float, ...] | None = None

    @property
    def block_groups(self) -> int:
        return math.ceil(self.num_blocks / self.block_group_size)

    @property
    def num_groups(self) -> int:
        return self.block_groups + 1

    @property
    def route_names(self) -> tuple[str, ...]:
        return ("base",) + tuple(expert.name for expert in self.experts)


def validate_lora_moe_config(config: NativeLoRAMoEConfig) -> NativeLoRAMoEConfig:
    """Validate structural invariants of a ``NativeLoRAMoEConfig``."""
    names = [expert.name for expert in config.experts]
    if len(names) != len(set(names)):
        raise ValueError("expert names must be unique")
    if config.enabled and not config.experts:
        raise ValueError("enabled LoRA-MoE requires at least one expert")
    if config.context_dim <= 0 or config.hidden_dim <= 0:
        raise ValueError("context_dim and hidden_dim must be positive")
    if config.num_blocks <= 0 or config.block_group_size <= 0:
        raise ValueError("num_blocks and block_group_size must be positive")
    if not 1 <= config.top_k <= len(config.route_names):
        raise ValueError("top_k is outside the route count")
    if not math.isfinite(config.temperature) or config.temperature <= 0:
        raise ValueError("temperature must be positive")
    return config


class NativeLoRARouter(nn.Module):
    """Context, timestep, and block-group router with null-route prior and learned strengths."""

    def __init__(self, config: NativeLoRAMoEConfig) -> None:
        super().__init__()
        self.config = validate_lora_moe_config(config)
        self.route_names = config.route_names
        self.context_norm = nn.LayerNorm(config.context_dim)
        self.context_proj = nn.Linear(config.context_dim, config.hidden_dim, bias=False)
        self.time_feature_dim = 18
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.group_embedding = nn.Parameter(
            torch.empty(config.num_groups, config.hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, len(self.route_names)),
        )
        prior = self._default_prior(config)
        self.register_buffer("route_prior", torch.tensor(prior, dtype=torch.float32))
        initial_strengths = torch.tensor(
            [expert.strength for expert in config.experts], dtype=torch.float32
        ).repeat(config.num_groups, 1)
        self.register_buffer("initial_strengths", initial_strengths)
        ratio = (initial_strengths / 1.5).clamp(1e-5, 1.0 - 1e-5)
        self.strength_logits = nn.Parameter(torch.logit(ratio))
        self.last_logits: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None
        self.last_gates: torch.Tensor | None = None
        self.reset_parameters()

    @staticmethod
    def _default_prior(config: NativeLoRAMoEConfig) -> tuple[float, ...]:
        if config.route_prior is not None:
            return config.route_prior
        if not config.experts:
            return (1.0,)
        base_prob = 0.30
        expert_prob = (1.0 - base_prob) / len(config.experts)
        return (base_prob,) + (expert_prob,) * len(config.experts)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.context_proj.weight)
        for module in self.time_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        hidden_linear = self.gate[1]
        output_linear = self.gate[3]
        nn.init.xavier_uniform_(hidden_linear.weight)
        nn.init.zeros_(hidden_linear.bias)
        nn.init.zeros_(output_linear.weight)
        nn.init.zeros_(output_linear.bias)

    @property
    def strengths(self) -> torch.Tensor:
        return 1.5 * torch.sigmoid(self.strength_logits)

    @staticmethod
    def _time_features(timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float().reshape(-1, 1)
        frequencies = torch.arange(8, device=timestep.device, dtype=torch.float32)
        frequencies = torch.pow(2.0, frequencies).reshape(1, -1) * math.pi
        angles = timestep * frequencies
        return torch.cat(
            (timestep, timestep.square(), torch.sin(angles), torch.cos(angles)), dim=-1
        )

    def _pool_context(
        self, context: torch.Tensor, text_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if context.ndim == 4:
            context = context.mean(dim=2)
        if context.ndim != 3 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"expected context [B,L,{self.config.context_dim}], got {tuple(context.shape)}"
            )
        context = context.float()
        if text_mask is None:
            return context.mean(dim=1)
        mask = text_mask.to(device=context.device, dtype=context.dtype)
        return (context * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def forward(
        self,
        context: torch.Tensor,
        timestep: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self._pool_context(context, text_mask)
        context_features = self.context_proj(self.context_norm(pooled))
        time_features = self.time_proj(self._time_features(timestep))
        fused = torch.tanh(
            context_features[:, None, :]
            + time_features[:, None, :]
            + self.group_embedding[None, :, :]
        )
        logits = (self.gate(fused) + self.route_prior.log().reshape(1, 1, -1)) / (
            self.config.temperature
        )
        probabilities = torch.softmax(logits, dim=-1)
        if self.training and self.config.router_noise and self.config.top_k == 1:
            gates = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            top_values, top_indices = probabilities.topk(self.config.top_k, dim=-1)
            hard = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
            hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gates = (hard - probabilities.detach() + probabilities) if self.training else hard
        self.last_logits = logits
        self.last_probabilities = probabilities
        self.last_gates = gates
        return gates

    def auxiliary_loss(self) -> torch.Tensor:
        if self.last_logits is None or self.last_probabilities is None:
            raise RuntimeError("auxiliary_loss requested before router forward")
        importance = self.last_probabilities.mean(dim=(0, 1)).clamp_min(1e-8)
        prior = self.route_prior.to(importance)
        balance = torch.sum(importance * (importance.log() - prior.log()))
        z_loss = torch.logsumexp(self.last_logits, dim=-1).square().mean()
        strength_loss = (self.strengths - self.initial_strengths).square().mean()
        return (
            self.config.balance_loss_weight * balance
            + self.config.z_loss_weight * z_loss
            + self.config.strength_loss_weight * strength_loss
        )


def block_group_for_module_name(
    module_name: str, *, num_blocks: int = 28, block_group_size: int = 4
) -> int:
    """Map a transformer module path to its LoRA-MoE block group index."""
    match = _BLOCK_RE.search(module_name)
    if match is None:
        return math.ceil(num_blocks / block_group_size)
    block = int(match.group(1))
    if not 0 <= block < num_blocks:
        raise ValueError(f"out-of-range block index in {module_name!r}")
    return block // block_group_size


def set_only_adapter_trainable(model: Any, adapter_name: str) -> None:
    """Freeze base weights and other adapters, keeping one FP32 adapter trainable."""
    marker = f".{adapter_name}."
    found = 0
    for name, parameter in model.named_parameters():
        trainable = marker in name
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.float()
            found += 1
    if not found:
        raise RuntimeError(f"adapter parameters not found: {adapter_name}")


def activate_adapter(model: Any, adapter_name: str, *, trainable_adapter: str) -> None:
    """Switch active PEFT adapter while preserving trainability invariants."""
    base_model = getattr(model, "base_model", None)
    base_set_adapter = getattr(base_model, "set_adapter", None)
    if base_set_adapter is not None:
        base_set_adapter(
            adapter_name,
            inference_mode=adapter_name != trainable_adapter,
        )
    else:
        model.set_adapter(adapter_name)
        set_only_adapter_trainable(model, trainable_adapter)
