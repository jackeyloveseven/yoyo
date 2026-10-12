"""Reward-matching mathematical primitives."""

from .objectives import (
    AWMConfig,
    AWMInputs,
    AWMMetrics,
    RAMMetrics,
    advantage_weighted_matching_loss,
    normalized_advantages,
    reinforce_adjoint_matching_loss,
)

__all__ = [
    "AWMMetrics",
    "AWMConfig",
    "AWMInputs",
    "RAMMetrics",
    "advantage_weighted_matching_loss",
    "normalized_advantages",
    "reinforce_adjoint_matching_loss",
]
