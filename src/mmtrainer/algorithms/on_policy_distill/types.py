"""On-Policy Distillation(在线策略蒸馏)少步蒸馏的输入容器与配置。

与仓库其它目标零件一样,这里只放纯张量容器与校验:不 import 任何具体模型。
教师多步轨迹由调用方用 :func:`mmtrainer.nn.flow.euler_flow_step` 构造并 ``detach``
后传入;本零件不持有教师网络或调度状态。

逐样本权重的校验复用 LCM 零件的 ``_sample_weight``(同仓库零件复用,避免重复
窗口;``_batch_sigma`` 由 objectives 直接从 LCM types 引入),只把"可选权重
缺省视为全 1"这一差异留在本文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import EventDims, working_dtype
from mmtrainer.algorithms.latent_consistency.types import _sample_weight


@dataclass(frozen=True)
class OnPolicyDistillInputs:
    """一次在线策略蒸馏更新所需的全部张量。

    ``student_velocity`` 是学生网络在 ``(x_t, sigma)`` 上的带梯度输出;
    ``teacher_x0`` 是教师从同一个 ``x_t`` 出发走 ``num_distill_steps`` 个 Euler 步
    得到的干净样本估计,调用方必须已 ``detach``(零件内部还会再 ``detach`` 一次
    作为防线)。``sample_weight`` 是可选逐样本权重(shape 为事件维之外的
    leading shape,例如二维 latent 的 ``(batch,)``)。
    """

    student_velocity: Tensor
    teacher_x0: Tensor
    x_t: Tensor
    sigma: Tensor
    sample_weight: Tensor | None = None


@dataclass(frozen=True)
class OnPolicyDistillConfig:
    """在线策略蒸馏的数值配置。

    ``num_distill_steps`` 与 ``min_distill_idx`` 由调用方在构造教师轨迹与过滤
    时间步时使用,零件只负责校验其合法性并透传给调用方参考;``event_dims`` 决定
    归约到逐样本损失的维度,语义与仓库其它目标一致。
    """

    num_distill_steps: int = 4
    min_distill_idx: int = 0
    loss_weight: float = 1.0
    event_dims: EventDims = None

    def __post_init__(self) -> None:
        steps, min_idx, weight = self.num_distill_steps, self.min_distill_idx, self.loss_weight
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("num_distill_steps must be an integer")
        if steps < 1:
            raise ValueError("num_distill_steps must be at least 1")
        if isinstance(min_idx, bool) or not isinstance(min_idx, int):
            raise TypeError("min_distill_idx must be an integer")
        if min_idx < 0:
            raise ValueError("min_distill_idx must be non-negative")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("loss_weight must be a real number")
        if not float(weight) > 0:
            raise ValueError("loss_weight must be positive")


def _optional_sample_weight(
    name: str,
    value: Tensor | None,
    reference: Tensor,
    dims: tuple[int, ...],
) -> Tensor:
    """可选逐样本权重:缺省返回全 1,否则委托 LCM 的 ``_sample_weight`` 校验。"""
    if value is None:
        return torch.ones(
            reference.shape[0],
            device=reference.device,
            dtype=working_dtype(reference),
        )
    return _sample_weight(name, value, reference, dims)


__all__ = [
    "EventDims",
    "OnPolicyDistillConfig",
    "OnPolicyDistillInputs",
]
