"""Unit tests for enhanced dataset transforms and batch collators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mmtrainer.datasets.manifest import (
    JsonlEditDataset,
    JsonlEditPairDataset,
    collate_edit_batch,
    collate_edit_pair_batch,
    letterbox_image_transform,
    normalized_image_transform,
)


def _make_solid_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def test_letterbox_image_transform_tall() -> None:
    # 100x200 tall image resized to max dim 256 -> new_w=128, new_h=256
    img = _make_solid_image(100, 200, (255, 255, 255))
    transform = letterbox_image_transform(resolution=256)
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)

    # Content should be centered horizontally: left pad = 64, right pad = 64
    left_padding = tensor[:, :, :64]
    right_padding = tensor[:, :, 192:]
    center_content = tensor[:, :, 64:192]

    assert torch.all(left_padding == 0.0)
    assert torch.all(right_padding == 0.0)
    # White (255) normalized by (x - 0.5) / 0.5 is 1.0
    assert torch.allclose(center_content, torch.tensor(1.0), atol=1e-2)


def test_letterbox_image_transform_wide() -> None:
    # 300x150 wide image resized to max dim 256 -> new_w=256, new_h=128
    img = _make_solid_image(300, 150, (255, 255, 255))
    transform = letterbox_image_transform(resolution=256)
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)

    # Content should be centered vertically: top pad = 64, bottom pad = 64
    top_padding = tensor[:, :64, :]
    bottom_padding = tensor[:, 192:, :]
    center_content = tensor[:, 64:192, :]

    assert torch.all(top_padding == 0.0)
    assert torch.all(bottom_padding == 0.0)
    assert torch.allclose(center_content, torch.tensor(1.0), atol=1e-2)


def test_normalized_image_transform_modes() -> None:
    img = _make_solid_image(100, 200, (255, 255, 255))

    crop_tf = normalized_image_transform(64, mode="crop")
    crop_out = crop_tf(img)
    assert crop_out.shape == (3, 64, 64)
    # Center crop of pure white has no 0.0 letterbox padding
    assert torch.allclose(crop_out, torch.tensor(1.0), atol=1e-2)

    letterbox_tf = normalized_image_transform(64, mode="letterbox")
    lb_out = letterbox_tf(img)
    assert lb_out.shape == (3, 64, 64)
    # 100x200 to 64 -> new_w=32, new_h=64 -> left pad 16
    assert torch.all(lb_out[:, :, :16] == 0.0)

    pad_tf = normalized_image_transform(64, mode="pad")
    pad_out = pad_tf(img)
    assert torch.equal(pad_out, lb_out)

    with pytest.raises(ValueError, match="Unsupported image transform mode"):
        normalized_image_transform(64, mode="invalid_mode")


def test_collate_edit_batch_variable_references_and_dataloader(tmp_path: Path) -> None:
    img_ref1 = tmp_path / "ref1.png"
    img_ref2_a = tmp_path / "ref2_a.png"
    img_ref2_b = tmp_path / "ref2_b.png"
    img_out1 = tmp_path / "out1.png"
    img_out2 = tmp_path / "out2.png"

    _make_solid_image(40, 40, (10, 20, 30)).save(img_ref1)
    _make_solid_image(40, 40, (40, 50, 60)).save(img_ref2_a)
    _make_solid_image(40, 40, (70, 80, 90)).save(img_ref2_b)
    _make_solid_image(40, 40, (100, 110, 120)).save(img_out1)
    _make_solid_image(40, 40, (130, 140, 150)).save(img_out2)

    jsonl_path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "prompt": "single reference edit",
            "input_images": [str(img_ref1.name)],
            "output_image": str(img_out1.name),
        },
        {
            "prompt": "dual reference edit",
            "input_images": [str(img_ref2_a.name), str(img_ref2_b.name)],
            "output_image": str(img_out2.name),
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    dataset = JsonlEditDataset(jsonl_path, resolution=32, mode="letterbox")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_edit_batch)

    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]

    assert batch["prompts"] == ["single reference edit", "dual reference edit"]
    assert batch["prompt"] == batch["prompts"]
    assert batch["target_tensor"].shape == (2, 3, 32, 32)
    assert batch["control_tensor"].shape == (2, 3, 32, 32)
    assert batch["padded_control_tensors"].shape == (2, 2, 3, 32, 32)
    assert batch["control_mask"].shape == (2, 2)

    # First sample has 1 reference, second has 2
    assert batch["control_mask"][0].tolist() == [True, False]
    assert batch["control_mask"][1].tolist() == [True, True]

    # Masked slot must be padded with zeros
    assert torch.all(batch["padded_control_tensors"][0, 1] == 0.0)

    # Valid slots must match the dataset outputs
    assert torch.equal(batch["padded_control_tensors"][0, 0], batch["control_tensor"][0])
    assert torch.equal(batch["padded_control_tensors"][1, 0], batch["control_tensor"][1])

    # PILs preserved as nested list
    assert len(batch["control_pils"]) == 2
    assert len(batch["control_pils"][0]) == 1
    assert len(batch["control_pils"][1]) == 2
    assert isinstance(batch["control_pils"][0][0], Image.Image)


def test_collate_edit_pair_batch_with_dataloader(tmp_path: Path) -> None:
    img_ref1_a = tmp_path / "p_ref1_a.png"
    img_ref1_b = tmp_path / "p_ref1_b.png"
    img_ref2 = tmp_path / "p_ref2.png"
    win1 = tmp_path / "win1.png"
    lose1 = tmp_path / "lose1.png"
    win2 = tmp_path / "win2.png"
    lose2 = tmp_path / "lose2.png"

    _make_solid_image(32, 32, (1, 1, 1)).save(img_ref1_a)
    _make_solid_image(32, 32, (2, 2, 2)).save(img_ref1_b)
    _make_solid_image(32, 32, (3, 3, 3)).save(img_ref2)
    _make_solid_image(32, 32, (10, 10, 10)).save(win1)
    _make_solid_image(32, 32, (20, 20, 20)).save(lose1)
    _make_solid_image(32, 32, (30, 30, 30)).save(win2)
    _make_solid_image(32, 32, (40, 40, 40)).save(lose2)

    jsonl_path = tmp_path / "pair_manifest.jsonl"
    rows = [
        {
            "prompt": "pair sample with 2 refs",
            "input_images": [str(img_ref1_a.name), str(img_ref1_b.name)],
            "win_image": str(win1.name),
            "lose_image": str(lose1.name),
        },
        {
            "prompt": "pair sample with 1 ref",
            "input_images": [str(img_ref2.name)],
            "win_image": str(win2.name),
            "lose_image": str(lose2.name),
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    dataset = JsonlEditPairDataset(jsonl_path, resolution=32, mode="letterbox")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_edit_pair_batch)

    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]

    assert batch["prompts"] == ["pair sample with 2 refs", "pair sample with 1 ref"]
    assert batch["win_tensor"].shape == (2, 3, 32, 32)
    assert batch["lose_tensor"].shape == (2, 3, 32, 32)
    assert batch["control_tensor"].shape == (2, 3, 32, 32)
    assert batch["padded_control_tensors"].shape == (2, 2, 3, 32, 32)
    assert batch["control_mask"].shape == (2, 2)

    assert batch["control_mask"][0].tolist() == [True, True]
    assert batch["control_mask"][1].tolist() == [True, False]
    assert torch.all(batch["padded_control_tensors"][1, 1] == 0.0)
