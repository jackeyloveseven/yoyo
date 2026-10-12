"""LCM(Latent Consistency Model)一致性蒸馏的输入容器、配置与共享校验辅助。

与仓库其它目标零件一样,这里只放纯张量容器与校验:不 import 任何具体模型,也不持有
优化器或 EMA 状态。目标网络的 EMA 前向与参数更新由调用方维护。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from mmtrainer.algorithms.common import (
    EventDims,
    require_float_finite,
    working_dtype,
)


@dataclass(frozen=True)
class LCMConsistencyInputs:
    """一次一致性更新所需的全部张量。

    ``online_velocity`` 是学生网络在当前样本上的带梯度输出;``target_velocity``
    是 EMA 目标网络在自举样本 ``x_{t'}`` 上的输出,调用方必须已 ``detach``(零件
    内部还会再 ``detach`` 一次作为防线)。调用方需先用同一条
    :func:`mmtrainer.nn.flow.euler_flow_step` 先把 ``noisy`` 推进到 ``sigma_prev``
    才能在 ``x_{t'}`` 上查询目标网络。
    """

    noisy: Tensor
    online_velocity: Tensor
    target_velocity: Tensor
    sigma_curr: Tensor
    sigma_prev: Tensor


@dataclass(frozen=True)
class LCMConsistencyConfig:
    """一致性损失的数值配置。

    ``huber_c`` 为伪 Huber 的阈值;``snr_weight`` 是可选逐样本权重(shape 为事件维
    之外的 leading shape,例如二维 latent 的 ``(batch,)``);``event_dims`` 决定归约
    到逐样本损失的维度,语义与仓库其它目标一致。
    """

    huber_c: float = 1e-3
    snr_weight: Tensor | None = None
    event_dims: EventDims = None


def _sample_weight(
    name: str,
    value: Tensor,
    reference: Tensor,
    dims: tuple[int, ...],
) -> Tensor:
    """校验逐样本权重(形状为事件维之外的 leading shape)并转成计算 dtype。"""
    require_float_finite(name, value)
    kept = [size for index, size in enumerate(reference.shape) if index not in dims]
    if list(value.shape) != kept:
        observed = tuple(value.shape)
        raise ValueError(f"{name} must have leading shape {tuple(kept)}, got {observed}")
    if value.device != reference.device:
        raise ValueError(f"{name} must be on the prediction device")
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    return value.to(dtype=working_dtype(reference, value))


def _batch_sigma(name: str, value: Tensor, reference: Tensor) -> Tensor:
    """校验 (batch,) 形状的噪声系数:有限、落在 [0, 1]、与参考张量同设备。"""
    require_float_finite(name, value)
    batch = reference.shape[0]
    if value.ndim != 1 or value.shape[0] != batch:
        raise ValueError(f"{name} must have shape ({batch},), got {tuple(value.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the noised latents")
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must lie inside [0, 1]")
    return value


__all__ = [
    "EventDims",
    "LCMConsistencyConfig",
    "LCMConsistencyInputs",
]
