"""Distributed statistics and rank-consensus gates for DiffusionOPSD."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def distributed_reward_moments(rewards: Tensor, group: Any = None) -> tuple[Tensor, Tensor]:
    """Return population mean/std over the complete distributed rollout batch."""

    if not isinstance(rewards, Tensor) or not rewards.is_floating_point():
        raise TypeError("rewards must be a floating-point Tensor")
    if rewards.numel() == 0 or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be non-empty and finite")
    work = rewards.detach().double()
    totals = torch.stack(
        (
            work.sum(),
            work.square().sum(),
            work.new_tensor(float(work.numel())),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=group)
    count = totals[2]
    mean = totals[0] / count
    variance = (totals[1] / count - mean.square()).clamp_min(0)
    return mean.float(), variance.sqrt().float()


def require_all_ranks(
    accelerator: Any,
    condition: bool,
    *,
    message: str,
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Raise on every rank when any rank reports a failed condition."""

    if not isinstance(condition, bool):
        raise TypeError("distributed gate condition must be boolean")
    if not message:
        raise ValueError("distributed gate message must be non-empty")
    world_size = int(getattr(accelerator, "num_processes", 1))
    if world_size == 1:
        if not condition:
            raise error_type(message)
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed gate requires initialized torch.distributed")
    device = torch.device(accelerator.device)
    flag = torch.tensor(1 if condition else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) != 1:
        raise error_type(message)


__all__ = ["distributed_reward_moments", "require_all_ranks"]
