"""Gradient-preserving reward interfaces used during target construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class DifferentiableReward(Protocol):
    def __call__(self, samples: Tensor, context: Any = None) -> Tensor: ...


def _prompt_grid(context: Any, leading: tuple[int, int]) -> list[str]:
    prompts: Sequence[str] | None = None
    if isinstance(context, Mapping):
        candidate = context.get("prompts")
        if isinstance(candidate, Sequence) and not isinstance(candidate, str):
            prompts = candidate
    elif isinstance(context, Sequence) and not isinstance(context, str):
        prompts = context
    if prompts is None or len(prompts) != leading[0]:
        raise ValueError("reward context must provide one prompt per prompt batch")
    return [str(prompt) for prompt in prompts for _ in range(leading[1])]


def _size_pair(value: Any, *, default: int = 224) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if "height" in value and "width" in value:
            pair = (int(value["height"]), int(value["width"]))
        else:
            edge = int(value.get("shortest_edge", default))
            pair = (edge, edge)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("reward processor size must contain two entries")
        pair = (int(value[0]), int(value[1]))
    else:
        edge = int(value)
        pair = (edge, edge)
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError("reward processor size must be positive")
    return pair


def _resize_clip(work: Tensor, value: Any) -> Tensor:
    source_height, source_width = work.shape[-2:]
    preserve_aspect = isinstance(value, int) or (
        isinstance(value, Mapping)
        and "shortest_edge" in value
        and not {"height", "width"}.issubset(value)
    )
    resize_height, resize_width = _size_pair(value)
    if preserve_aspect:
        scale = resize_height / min(source_height, source_width)
        resize_height = max(1, round(source_height * scale))
        resize_width = max(1, round(source_width * scale))
    return F.interpolate(
        work,
        size=(resize_height, resize_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _center_crop_clip(work: Tensor, value: Any) -> Tensor:
    crop_height, crop_width = _size_pair(value)
    pad_height = max(0, crop_height - work.shape[-2])
    pad_width = max(0, crop_width - work.shape[-1])
    if pad_height or pad_width:
        work = F.pad(
            work,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
    top = round((work.shape[-2] - crop_height) / 2.0)
    left = round((work.shape[-1] - crop_width) / 2.0)
    return work[..., top : top + crop_height, left : left + crop_width]


def differentiable_clip_preprocess(images: Tensor, image_processor: Any) -> Tensor:
    """Tensor-only equivalent of the reference repo's CLIP transform."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("reward images must be [N, 3, H, W]")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("reward images must be finite floating-point tensors")
    to_dict = getattr(image_processor, "to_dict", None)
    config = dict(to_dict()) if callable(to_dict) else {}

    work = images.float()
    do_resize = bool(config.get("do_resize", True))
    resize_value = config.get(
        "size",
        getattr(
            image_processor,
            "size",
            getattr(image_processor, "crop_size", 224),
        ),
    )
    if do_resize:
        work = _resize_clip(work, resize_value)

    do_center_crop = bool(config.get("do_center_crop", True))
    if do_center_crop:
        crop_value = config.get("crop_size", getattr(image_processor, "crop_size", 224))
        work = _center_crop_clip(work, crop_value)

    if not bool(config.get("do_normalize", True)):
        return work
    mean = torch.tensor(
        getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)),
        device=work.device,
        dtype=work.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)),
        device=work.device,
        dtype=work.dtype,
    ).view(1, 3, 1, 1)
    return (work - mean) / std


@dataclass
class PickScoreDifferentiableReward:
    """Frozen PickScore model with an image path that never leaves autograd."""

    model: Any
    processor: Any
    scale_divisor: float = 26.0

    def __post_init__(self) -> None:
        if self.scale_divisor <= 0:
            raise ValueError("scale_divisor must be positive")
        self.model.eval().requires_grad_(False)

    def __call__(self, samples: Tensor, context: Any = None) -> Tensor:
        if samples.ndim != 5 or samples.shape[2] != 3:
            raise ValueError("PickScore samples must be [batch, group, 3, H, W]")
        leading = tuple(samples.shape[:2])
        prompts = _prompt_grid(context, leading)
        flat = samples.reshape(-1, *samples.shape[2:])
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        pixels = differentiable_clip_preprocess(flat, self.processor.image_processor)
        pixels = pixels.to(device=device, dtype=dtype)
        text = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        image_features = self.model.get_image_features(pixel_values=pixels)
        text_features = self.model.get_text_features(**text)
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        scale = self.model.logit_scale.exp().float()
        scores = scale * (image_features * text_features).sum(dim=-1)
        return (scores / self.scale_divisor).reshape(leading)


def require_nonzero_image_gradient(
    reward: DifferentiableReward,
    samples: Tensor,
    context: Any = None,
) -> Tensor:
    """Preflight one reward and reject scalar-only/detached implementations."""

    leaf = samples.detach().float().requires_grad_(True)
    scores = reward(leaf, context)
    if not isinstance(scores, Tensor) or not scores.is_floating_point():
        raise TypeError("differentiable reward must return floating-point scores")
    if not torch.isfinite(scores).all():
        raise ValueError("differentiable reward returned non-finite scores")
    if not scores.requires_grad:
        raise RuntimeError("reward detached the generated image")
    gradient = torch.autograd.grad(scores.sum(), leaf, allow_unused=True)[0]
    if gradient is None:
        raise RuntimeError("reward detached the generated image")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("reward image gradient is non-finite")
    if not bool((gradient != 0).any()):
        raise RuntimeError("reward image gradient is identically zero")
    return gradient.detach()


__all__ = [
    "DifferentiableReward",
    "PickScoreDifferentiableReward",
    "differentiable_clip_preprocess",
    "require_nonzero_image_gradient",
]
