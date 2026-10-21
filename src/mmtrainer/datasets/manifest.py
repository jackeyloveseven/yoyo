"""Composable JSONL manifest datasets for T2I, image editing, and preference DPO."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "JsonlEditDataset",
    "JsonlEditPairDataset",
    "JsonlT2IDataset",
    "collate_edit_batch",
    "collate_edit_pair_batch",
    "letterbox_image_transform",
    "load_jsonl_rows",
    "normalized_image_transform",
]


def load_jsonl_rows(
    data_jsonl: str | Path,
    *,
    required_keys: tuple[str, ...],
) -> tuple[Path, list[dict[str, Any]]]:
    jsonl_path = Path(data_jsonl).resolve()
    base_dir = jsonl_path.parent
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            missing = [key for key in required_keys if key not in row]
            if missing:
                raise KeyError(
                    f"{jsonl_path}:{line_number} missing required keys: {', '.join(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset manifest: {jsonl_path}")
    return base_dir, rows


def letterbox_image_transform(resolution: int) -> Callable[[Image.Image], torch.Tensor]:
    """Resize image preserving aspect ratio and center on a square canvas.

    The canvas padding is filled with 0.0 in the normalized [-1.0, 1.0] range
    (equivalent to 0.5 mid-gray in [0.0, 1.0] before normalization).
    """
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def _transform(image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")
        scale = resolution / max(orig_w, orig_h)
        new_w = max(1, min(resolution, int(round(orig_w * scale))))
        new_h = max(1, min(resolution, int(round(orig_h * scale))))
        resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        resized_tensor = normalize(to_tensor(resized))

        canvas = torch.zeros((3, resolution, resolution), dtype=resized_tensor.dtype)
        top = (resolution - new_h) // 2
        left = (resolution - new_w) // 2
        canvas[:, top : top + new_h, left : left + new_w] = resized_tensor
        return canvas

    return _transform


def normalized_image_transform(
    resolution: int,
    *,
    mode: str = "crop",
) -> Callable[[Image.Image], Any]:
    if mode == "crop":
        return transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    if mode in ("letterbox", "pad"):
        return letterbox_image_transform(resolution)
    raise ValueError(
        f"Unsupported image transform mode: {mode!r}; expected 'crop', 'letterbox', or 'pad'"
    )


def _resolve_media_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _load_rgb_image(base_dir: Path, raw_path: str) -> Image.Image:
    return Image.open(_resolve_media_path(base_dir, raw_path)).convert("RGB")


def _reference_paths(row: dict[str, Any], *, max_refs: int) -> list[str]:
    if "input_images" in row:
        raw_refs = row["input_images"]
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("input_images must be a non-empty list")
        refs = [str(path) for path in raw_refs]
    elif "input_image" in row:
        refs = [str(row["input_image"])]
    else:
        raise KeyError("Expected 'input_image' or 'input_images' in dataset row")
    if len(refs) > max_refs:
        raise ValueError(f"Dataset row has {len(refs)} reference images; max_refs={max_refs}")
    return refs


def _collate_reference_fields(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    prompts = [str(item["prompt"]) for item in batch]
    control_tensor = torch.stack([item["control_tensor"] for item in batch], dim=0)

    ref_tensor_lists: list[list[torch.Tensor]] = [
        list(item.get("control_tensors", [item["control_tensor"]])) for item in batch
    ]
    max_refs = max(len(refs) for refs in ref_tensor_lists) if ref_tensor_lists else 0

    c, h, w = control_tensor.shape[1:]
    batch_size = len(batch)

    padded_control_tensors = torch.zeros(
        (batch_size, max_refs, c, h, w),
        dtype=control_tensor.dtype,
        device=control_tensor.device,
    )
    control_mask = torch.zeros(
        (batch_size, max_refs),
        dtype=torch.bool,
        device=control_tensor.device,
    )

    for i, refs in enumerate(ref_tensor_lists):
        for j, ref in enumerate(refs):
            padded_control_tensors[i, j] = ref
            control_mask[i, j] = True

    control_pils = [
        list(item.get("control_pils", [item["control_pil"]] if "control_pil" in item else []))
        for item in batch
    ]
    control_pil = [
        item.get("control_pil", item["control_pils"][0] if item.get("control_pils") else None)
        for item in batch
    ]

    return {
        "prompt": prompts,
        "prompts": prompts,
        "control_tensor": control_tensor,
        "padded_control_tensors": padded_control_tensors,
        "control_mask": control_mask,
        "control_pils": control_pils,
        "control_pil": control_pil,
    }


def collate_edit_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples from JsonlEditDataset into a padded and batched dictionary."""
    result = _collate_reference_fields(batch)
    result["target_tensor"] = torch.stack([item["target_tensor"] for item in batch], dim=0)
    return result


def collate_edit_pair_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples from JsonlEditPairDataset into a padded preference batch."""
    result = _collate_reference_fields(batch)
    result["win_tensor"] = torch.stack([item["win_tensor"] for item in batch], dim=0)
    result["lose_tensor"] = torch.stack([item["lose_tensor"] for item in batch], dim=0)
    return result


class JsonlT2IDataset(Dataset[dict[str, Any]]):
    """JSONL dataset for text-to-image training (`prompt`, `output_image`)."""

    def __init__(
        self,
        data_jsonl: str | Path,
        resolution: int = 1024,
        *,
        mode: str = "crop",
    ) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "output_image")
        )
        self.transform = normalized_image_transform(resolution, mode=mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        target_img = _load_rgb_image(self.base_dir, str(row["output_image"]))
        return {
            "prompt": str(row["prompt"]),
            "target_tensor": self.transform(target_img),
        }


class JsonlEditDataset(Dataset[dict[str, Any]]):
    """JSONL dataset for single/multi-reference image editing.

    Rows carry `prompt`, `input_image(s)`, and `output_image`.
    """

    def __init__(
        self,
        data_jsonl: str | Path,
        resolution: int = 1024,
        *,
        mode: str = "crop",
        max_refs: int = 4,
    ) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "output_image")
        )
        self.max_refs = max_refs
        self.transform = normalized_image_transform(resolution, mode=mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ref_paths = _reference_paths(row, max_refs=self.max_refs)
        ref_pils = [_load_rgb_image(self.base_dir, path) for path in ref_paths]
        ref_tensors = [self.transform(image) for image in ref_pils]
        target_img = _load_rgb_image(self.base_dir, str(row["output_image"]))
        return {
            "prompt": str(row["prompt"]),
            "control_pil": ref_pils[0],
            "control_pils": ref_pils,
            "control_tensor": ref_tensors[0],
            "control_tensors": ref_tensors,
            "target_tensor": self.transform(target_img),
        }


class JsonlEditPairDataset(Dataset[dict[str, Any]]):
    """JSONL dataset for preference alignment / DPO.

    Rows carry `prompt`, `input_image(s)`, `win_image`, and `lose_image`.
    """

    def __init__(
        self,
        data_jsonl: str | Path,
        resolution: int = 1024,
        *,
        mode: str = "crop",
        max_refs: int = 4,
    ) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "win_image", "lose_image")
        )
        self.max_refs = max_refs
        self.transform = normalized_image_transform(resolution, mode=mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ref_paths = _reference_paths(row, max_refs=self.max_refs)
        ref_pils = [_load_rgb_image(self.base_dir, path) for path in ref_paths]
        ref_tensors = [self.transform(image) for image in ref_pils]
        win_img = _load_rgb_image(self.base_dir, str(row["win_image"]))
        lose_img = _load_rgb_image(self.base_dir, str(row["lose_image"]))
        return {
            "prompt": str(row["prompt"]),
            "control_pil": ref_pils[0],
            "control_pils": ref_pils,
            "control_tensor": ref_tensors[0],
            "control_tensors": ref_tensors,
            "win_tensor": self.transform(win_img),
            "lose_tensor": self.transform(lose_img),
        }
