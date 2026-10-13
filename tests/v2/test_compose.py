"""Unit tests for the compose blueprint logic and guidance table."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from mmtrainer.cli import _run_compose
from mmtrainer.compose import (
    ComposeChoice,
    build_blueprint,
    compatible_objectives,
    compatible_rewards,
)
from mmtrainer.compose.compose import objective_requires_reward
from mmtrainer.compose.guidance import OBJECTIVE_GUIDANCE, REWARD_METRICS
from mmtrainer.registry import build_builtin_registries


@pytest.fixture(scope="module")
def registries():
    return build_builtin_registries()


def test_compatible_objectives_respects_capabilities(registries) -> None:
    # krea2 holds both diffusion_alignment and image_training -> runs everything.
    krea2 = compatible_objectives(registries, "krea2")
    assert "image_sft" in krea2 and "flow_grpo" in krea2
    # klein lacks diffusion_alignment -> only the image_* family.
    klein = compatible_objectives(registries, "klein")
    assert set(klein) == {"image_sft", "image_dpo", "image_opd", "latent_consistency"}


def test_reward_requirement_and_filtering(registries) -> None:
    assert objective_requires_reward(registries, "flow_grpo") is True
    assert objective_requires_reward(registries, "image_sft") is False
    # reward-required objectives exclude the "none" placeholder.
    assert "none" not in compatible_rewards(registries, "flow_grpo")
    # reward-free objectives are pinned to "none".
    assert compatible_rewards(registries, "image_dpo") == ["none"]


def test_guidance_table_covers_all_objectives_and_rewards(registries) -> None:
    objective_ids = {entry.plugin_id for entry in registries.objectives.entries()}
    assert objective_ids == set(OBJECTIVE_GUIDANCE), (
        f"guidance table out of sync: missing={objective_ids - set(OBJECTIVE_GUIDANCE)} "
        f"extra={set(OBJECTIVE_GUIDANCE) - objective_ids}"
    )
    reward_ids = {entry.plugin_id for entry in registries.rewards.entries()}
    unknown = set(REWARD_METRICS) - reward_ids
    assert not unknown, f"REWARD_METRICS names unregistered rewards: {unknown}"


def test_build_blueprint_includes_reward_metrics(registries) -> None:
    blueprint = build_blueprint(
        registries,
        ComposeChoice(model_id="krea2", objective_id="flow_grpo", reward_id="hpsv3pp"),
    )
    assert blueprint["reward_metrics"], "RL blueprint must surface reward metrics"
    assert blueprint["next_steps"] and blueprint["visualizations"]


def test_build_blueprint_rejects_incompatible(registries) -> None:
    with pytest.raises(ValueError, match="cannot run objective"):
        build_blueprint(
            registries,
            ComposeChoice(model_id="klein", objective_id="flow_grpo", reward_id="none"),
        )
    with pytest.raises(ValueError, match="not valid for objective"):
        build_blueprint(
            registries,
            ComposeChoice(model_id="krea2", objective_id="image_sft", reward_id="pickscore"),
        )


def test_render_blueprint_markdown_covers_sections(registries) -> None:
    from mmtrainer.compose import render_blueprint_markdown

    blueprint = build_blueprint(
        registries,
        ComposeChoice(model_id="krea2", objective_id="flow_grpo", reward_id="hpsv3pp"),
    )
    markdown = render_blueprint_markdown(blueprint)
    assert markdown.startswith("# 训练蓝图:krea2 × flow_grpo")
    for section in (
        "## 组件",
        "## 概要",
        "## 该监控的实验指标",
        "### reward 指标",
        "## 该做的可视化",
        "## 下一步(晋级阶梯)",
    ):
        assert section in markdown
    # reward 指标需真实出现在 reward 小节
    assert "hps 美学分" in markdown
    # 无需 reward 的目标不渲染 reward 小节
    sft = build_blueprint(
        registries,
        ComposeChoice(model_id="klein", objective_id="image_sft", reward_id="none"),
    )
    assert "### reward 指标" not in render_blueprint_markdown(sft)


def test_compose_output_json_and_markdown(tmp_path: Path) -> None:
    """--output writes valid JSON or Markdown; unsupported suffix raises."""
    base_args = dict(model="krea2", objective="flow_grpo", reward="hpsv3pp")

    json_path = tmp_path / "blueprint.json"
    _run_compose(Namespace(**base_args, output=json_path))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model"] == "krea2"
    assert payload["objective"] == "flow_grpo"
    assert payload["reward"] == "hpsv3pp"
    assert payload["metrics"] and payload["visualizations"]

    md_path = tmp_path / "blueprint.md"
    _run_compose(Namespace(**base_args, output=md_path))
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# 训练蓝图:krea2 × flow_grpo")
    assert "## 该监控的实验指标" in text

    bad_path = tmp_path / "blueprint.txt"
    with pytest.raises(ValueError, match="只支持 .json 或 .md"):
        _run_compose(Namespace(**base_args, output=bad_path))
