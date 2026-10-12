"""Composable JSONL manifest datasets for T2I, single/multi-ref image editing, and preference DPO."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


def normalized_image_transform(resolution: int) -> Callable[[Image.Image], Any]:
    return transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
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


class JsonlT2IDataset(Dataset[dict[str, Any]]):
    """JSONL dataset for text-to-image training (`prompt`, `output_image`)."""

    def __init__(self, data_jsonl: str | Path, resolution: int = 1024) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "output_image")
        )
        self.transform = normalized_image_transform(resolution)

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
    """JSONL dataset for single- or multi-reference image editing (`prompt`, `input_image(s)`, `output_image`)."""

    def __init__(
        self,
        data_jsonl: str | Path,
        resolution: int = 1024,
        *,
        max_refs: int = 4,
    ) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "output_image")
        )
        self.max_refs = max_refs
        self.transform = normalized_image_transform(resolution)

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
    """JSONL dataset for preference alignment / DPO (`prompt`, `input_image(s)`, `win_image`, `lose_image`)."""

    def __init__(
        self,
        data_jsonl: str | Path,
        resolution: int = 1024,
        *,
        max_refs: int = 4,
    ) -> None:
        self.base_dir, self.rows = load_jsonl_rows(
            data_jsonl, required_keys=("prompt", "win_image", "lose_image")
        )
        self.max_refs = max_refs
        self.transform = normalized_image_transform(resolution)

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
