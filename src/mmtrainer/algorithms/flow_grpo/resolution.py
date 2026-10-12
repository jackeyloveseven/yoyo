"""Dynamic-resolution warmup gate from SenseNova Eq. 7."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

CandidateT = TypeVar("CandidateT")


def _validate_resolution_gate(
    base_prob: Tensor,
    difficulty: Tensor,
    epoch: int,
    warmup_epochs: int,
    smoothing_margin: float,
) -> None:
    if not isinstance(base_prob, Tensor) or not base_prob.is_floating_point():
        raise TypeError("base_prob must be a floating-point Tensor")
    if not isinstance(difficulty, Tensor) or not difficulty.is_floating_point():
        raise TypeError("difficulty must be a floating-point Tensor")
    if base_prob.ndim != 1 or difficulty.ndim != 1 or base_prob.shape != difficulty.shape:
        raise ValueError("base_prob and difficulty must be equal one-dimensional tensors")
    if epoch < 0 or warmup_epochs < 0:
        raise ValueError("epoch and warmup_epochs must be non-negative")
    if smoothing_margin <= 0:
        raise ValueError("smoothing_margin must be positive")
    if bool((base_prob < 0).any()) or not torch.isfinite(base_prob).all():
        raise ValueError("base_prob must be finite and non-negative")
    if bool(((difficulty < 0) | (difficulty > 1)).any()):
        raise ValueError("difficulty must be in [0, 1]")
    if not torch.isfinite(difficulty).all():
        raise ValueError("difficulty must be finite")
    if base_prob.sum() <= 0:
        raise ValueError("base_prob must contain positive mass")


def gate_resolution_probabilities(
    base_prob: Tensor,
    difficulty: Tensor,
    *,
    epoch: int,
    warmup_epochs: int,
    smoothing_margin: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """Apply SenseNova Eq. 7 and renormalize probabilities.

    The caller supplies candidate order, base probabilities ``p_i``, and
    difficulty scores ``d_i``. The returned tuple is ``(probabilities, gate)``.
    At the start of warmup, only candidates with low difficulty are sampled;
    at the end, all candidates with non-zero base probability are restored.
    """

    _validate_resolution_gate(base_prob, difficulty, epoch, warmup_epochs, smoothing_margin)

    progress = 1.0 if warmup_epochs == 0 else min(epoch / warmup_epochs, 1.0)
    progress_tensor = base_prob.new_tensor(progress)
    gate = ((progress_tensor - difficulty) / smoothing_margin + 1).clamp(0, 1)
    gated = base_prob * gate
    if gated.sum() <= 0:
        raise ValueError("resolution gate removed all probability mass")
    return gated / gated.sum(), gate


@dataclass
class DynamicResolutionWarmup(Generic[CandidateT]):
    """Sample caller-defined resolution candidates under Eq. 7 gating."""

    candidates: Sequence[CandidateT]
    base_prob: Tensor
    difficulty: Tensor
    warmup_epochs: int
    smoothing_margin: float = 0.3

    def __post_init__(self) -> None:
        self.candidates = tuple(self.candidates)
        if len(self.candidates) != self.base_prob.numel():
            raise ValueError("candidates length must match base_prob")
        if len(self.candidates) != self.difficulty.numel():
            raise ValueError("candidates length must match difficulty")
        if len(self.candidates) == 0:
            raise ValueError("at least one resolution candidate is required")

    def probabilities(self, epoch: int) -> Tensor:
        """Return normalized gated probabilities for an epoch."""

        probabilities, _ = gate_resolution_probabilities(
            self.base_prob,
            self.difficulty,
            epoch=epoch,
            warmup_epochs=self.warmup_epochs,
            smoothing_margin=self.smoothing_margin,
        )
        return probabilities

    def gate(self, epoch: int) -> Tensor:
        """Return the unnormalized Eq. 7 gate for each candidate."""

        _, gate = gate_resolution_probabilities(
            self.base_prob,
            self.difficulty,
            epoch=epoch,
            warmup_epochs=self.warmup_epochs,
            smoothing_margin=self.smoothing_margin,
        )
        return gate

    def sample(self, epoch: int, *, generator: torch.Generator | None = None) -> CandidateT:
        """Sample one candidate using the gated distribution."""

        index = torch.multinomial(self.probabilities(epoch), 1, generator=generator).item()
        return self.candidates[index]
