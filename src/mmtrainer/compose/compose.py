"""Pure composition logic for ``mmtrain compose`` blueprints.

These functions read the live registry and the static guidance table to decide
which objectives a model can run, which rewards an objective needs, and what a
training blueprint should contain. They are deliberately free of any input/print
so they can be unit-tested; the CLI layer only collects choices and renders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mmtrainer.compose.guidance import (
    NEXT_STEP_GUIDANCE,
    OBJECTIVE_GUIDANCE,
    REWARD_METRICS,
)
from mmtrainer.registry import RegistrySet


@dataclass(frozen=True)
class ComposeChoice:
    """The user's component selections for one blueprint."""

    model_id: str
    objective_id: str
    reward_id: str
    dataset_id: str = "manifest"


def compatible_objectives(registries: RegistrySet, model_id: str) -> list[str]:
    """Return objective plugin_ids whose required capabilities the model holds."""

    model_entry = registries.models.entry(model_id)
    compatible: list[str] = []
    for objective in registries.objectives.entries():
        required = objective.metadata.get("required_model_capabilities", frozenset())
        if required <= model_entry.capabilities:
            compatible.append(objective.plugin_id)
    return sorted(compatible)


def objective_requires_reward(registries: RegistrySet, objective_id: str) -> bool:
    """Return True when the objective declares ``reward_required``."""

    entry = registries.objectives.entry(objective_id)
    return bool(entry.metadata.get("reward_required", False))


def compatible_rewards(registries: RegistrySet, objective_id: str) -> list[str]:
    """Return reward plugin_ids valid for the objective.

    An objective that needs a reward accepts any registered scorer except the
    explicit ``none`` placeholder; one that does not is pinned to ``none``.
    """

    if not objective_requires_reward(registries, objective_id):
        return ["none"]
    return sorted(
        entry.plugin_id for entry in registries.rewards.entries() if entry.plugin_id != "none"
    )


def build_blueprint(registries: RegistrySet, choice: ComposeChoice) -> dict[str, object]:
    """Assemble the blueprint mapping for one validated choice.

    Raises ``ValueError`` if the model cannot run the objective or the reward is
    not compatible, so the CLI never renders an invalid plan.
    """

    if choice.objective_id not in compatible_objectives(registries, choice.model_id):
        raise ValueError(
            f"model {choice.model_id} cannot run objective {choice.objective_id}: "
            "missing required capabilities"
        )
    if choice.reward_id not in compatible_rewards(registries, choice.objective_id):
        raise ValueError(
            f"reward {choice.reward_id} is not valid for objective {choice.objective_id}"
        )
    registries.datasets.entry(choice.dataset_id)

    guidance = OBJECTIVE_GUIDANCE[choice.objective_id]
    reward_metrics = REWARD_METRICS.get(choice.reward_id, ())
    metrics = list(guidance.metrics)
    if reward_metrics and not any("reward" in m for m in metrics):
        metrics.append("reward 曲线(见下)")
    return {
        "model": choice.model_id,
        "objective": choice.objective_id,
        "reward": choice.reward_id,
        "dataset": choice.dataset_id,
        "summary": guidance.summary,
        "metrics": metrics,
        "reward_metrics": list(reward_metrics),
        "visualizations": list(guidance.visualizations),
        "notes": list(guidance.notes),
        "next_steps": list(NEXT_STEP_GUIDANCE),
        "disclaimer": (
            "这是一份训练蓝图,不是可直接 launch 的 RunSpec。"
            "注册资产哈希与运行时拓扑由 WebTrainer 控制面在运行时填充。"
        ),
    }


def render_blueprint_markdown(blueprint: Mapping[str, object]) -> str:
    """把蓝图渲染为一份人可读的 Markdown 文档(供 ``--output plan.md`` 落盘)。"""

    def _bullets(key: str) -> list[str]:
        values = blueprint.get(key, [])
        if not isinstance(values, list) or not values:
            return ["- (无)"]
        return [f"- {item}" for item in values]

    lines = [
        f"# 训练蓝图:{blueprint['model']} × {blueprint['objective']}",
        "",
        f"> {blueprint['disclaimer']}",
        "",
        "## 组件",
        "",
        f"- 模型:`{blueprint['model']}`",
        f"- 目标:`{blueprint['objective']}`",
        f"- reward:`{blueprint['reward']}`",
        f"- 数据集:`{blueprint['dataset']}`",
        "",
        "## 概要",
        "",
        str(blueprint["summary"]),
        "",
        "## 该监控的实验指标",
        "",
        *_bullets("metrics"),
    ]
    reward_metrics = blueprint.get("reward_metrics")
    if isinstance(reward_metrics, list) and reward_metrics:
        lines += ["", "### reward 指标", "", *_bullets("reward_metrics")]
    lines += [
        "",
        "## 该做的可视化",
        "",
        *_bullets("visualizations"),
    ]
    notes = blueprint.get("notes")
    if isinstance(notes, list) and notes:
        lines += ["", "## 注意事项", "", *_bullets("notes")]
    lines += [
        "",
        "## 下一步(晋级阶梯)",
        "",
        *_bullets("next_steps"),
        "",
    ]
    return "\n".join(lines)
