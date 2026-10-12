"""Multi-dimensional VLM/MLLM reward protocols and composable reward scorers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

EDIT_DIMENSIONS: tuple[str, ...] = (
    "Instruction_Following",
    "Consistency",
    "Visual_Quality",
)

RATIONAL_REWARDS_SYSTEM_PROMPT = (
    "You are an expert evaluator of image editing quality. "
    "Score the edited image from 1 to 5 on Instruction_Following, Consistency, and Visual_Quality. "
    "Respond with <think>...</think><answer>{\"Instruction_Following\": int, "
    "\"Consistency\": int, \"Visual_Quality\": int}</answer>."
)

_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def build_edit_messages(
    instruction: str,
    *,
    source_image_url: str,
    edited_image_url: str,
) -> list[dict[str, Any]]:
    """Construct OpenAI-compatible multimodal chat messages for edit reward evaluation."""
    return [
        {"role": "system", "content": RATIONAL_REWARDS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": source_image_url}},
                {"type": "image_url", "image_url": {"url": edited_image_url}},
                {
                    "type": "text",
                    "text": f"Editing instruction: {instruction}\nEvaluate the edited second image relative to the original first image.",
                },
            ],
        },
    ]


def parse_rational_rewards_scores(
    raw_text: str,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Parse ``<answer>{...}</answer>`` JSON scores and compute normalized [0, 1] reward."""
    match = _ANSWER_BLOCK_RE.search(raw_text)
    candidate_text = match.group(1) if match else raw_text
    json_match = _JSON_OBJECT_RE.search(candidate_text)
    if json_match is None:
        raise ValueError("No JSON score object found in reward response")
    parsed = json.loads(json_match.group(0))
    scores: dict[str, float] = {}
    for dim in EDIT_DIMENSIONS:
        if dim not in parsed:
            raise KeyError(f"Missing reward dimension: {dim}")
        val = float(parsed[dim])
        scores[dim] = max(0.0, min(1.0, (val - 1.0) / 4.0))
    resolved_weights = weights or {
        "Instruction_Following": 0.5,
        "Consistency": 0.3,
        "Visual_Quality": 0.2,
    }
    total_w = sum(resolved_weights.get(dim, 0.0) for dim in EDIT_DIMENSIONS)
    if total_w <= 0:
        raise ValueError("Reward dimension weights must sum to a positive number")
    weighted = sum(
        scores[dim] * resolved_weights.get(dim, 0.0) for dim in EDIT_DIMENSIONS
    ) / total_w
    scores["overall"] = float(weighted)
    return scores


@dataclass
class CallableRewardScorer:
    """Composable reward scorer wrapping any callable or custom scoring function."""

    name: str = "callable_reward"
    fn: Callable[[Sequence[Any]], list[float]] | None = None
    default_score: float = 0.0

    def __call__(self, samples: Sequence[Any]) -> list[float]:
        if self.fn is not None:
            return [float(val) for val in self.fn(samples)]
        return [float(self.default_score) for _ in samples]
