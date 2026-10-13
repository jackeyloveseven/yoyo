"""On-Policy Distillation(在线策略蒸馏)少步蒸馏零件:只导出公开符号。

命名说明:仓库已有的 ``image_opd`` / ``flow_opd_ode_loss`` 是 DiffusionOPD
(一步确定性转移匹配,需 reward);本零件是 DiffSynth 系的 On-Policy Distillation
(教师多步轨迹终点对齐,无需 reward)。两者缩写同为 OPD 但方法不同,故本零件
一律用全名 OnPolicyDistill,避免混淆。
"""

from .objectives import (
    OnPolicyDistillMetrics,
    on_policy_distill_loss,
)
from .types import (
    OnPolicyDistillConfig,
    OnPolicyDistillInputs,
)

__all__ = [
    "OnPolicyDistillConfig",
    "OnPolicyDistillInputs",
    "OnPolicyDistillMetrics",
    "on_policy_distill_loss",
]
