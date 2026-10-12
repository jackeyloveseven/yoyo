"""Model-side protocol for connecting Flow-GRPO primitives to a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import Tensor

from .types import TrajectoryBatch, TransitionDistribution


class FlowGRPOAdapter(Protocol):
    """Required model-side operations; no model framework is assumed."""

    def stochastic_rollout(
        self,
        conditioning: Any,
        *,
        group_size: int,
        rollout_steps: int,
        noise_level: float,
        generator: torch.Generator | None = None,
    ) -> TrajectoryBatch:
        """Sample grouped trajectories and record transition parameters."""

    def transition_distribution(
        self,
        states: Tensor,
        timesteps: Tensor,
        conditioning: Any,
        *,
        reference: bool = False,
        step_indices: Sequence[int] | Tensor | None = None,
        noise_level: float = 0.7,
    ) -> TransitionDistribution:
        """Return policy or frozen-reference transition mean and std.

        ``states`` and ``timesteps`` contain the full rollout. When supplied,
        ``step_indices`` selects original rollout steps while preserving any
        model-side schedule lookup keyed by those indices.
        """

    def transition_logprob(
        self,
        next_states: Tensor,
        distribution: TransitionDistribution,
    ) -> Tensor:
        """Return per-step policy log-probabilities."""

    def reference_logprob(
        self,
        next_states: Tensor,
        distribution: TransitionDistribution,
    ) -> Tensor:
        """Return per-step reference log-probabilities for diagnostics."""
