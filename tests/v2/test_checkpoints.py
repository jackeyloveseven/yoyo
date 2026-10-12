from __future__ import annotations

import json

import pytest

from mmtrainer.errors import CheckpointError
from mmtrainer.runtime.checkpoints import (
    V2DirectoryCheckpointCodec,
    import_v1_policy,
    select_v1_policy_state,
    sha256_file,
)


def test_select_v1_policy_state_is_unambiguous_and_format_bound() -> None:
    payload = {
        "format": "firered-diffusionnft-v3",
        "policy_lora": {"layer.weight": object()},
    }
    key, state = select_v1_policy_state(payload, allowed_keys=("policy_lora",))
    assert key == "policy_lora"
    assert tuple(state) == ("layer.weight",)

    payload["policy"] = {"other": object()}
    with pytest.raises(CheckpointError, match="exactly one"):
        select_v1_policy_state(payload, allowed_keys=("policy_lora", "policy"))

    with pytest.raises(CheckpointError, match="supported final-v1"):
        select_v1_policy_state(
            {"format": "mmtrainer.checkpoint.v2", "policy_lora": {"x": object()}},
            allowed_keys=("policy_lora",),
        )


def test_v2_directory_codec_detects_corruption(tmp_path) -> None:
    codec = V2DirectoryCheckpointCodec()
    checkpoint = tmp_path / "checkpoint-1"
    manifest = codec.save(checkpoint, {"update": 1, "run_spec_sha256": "a" * 64})
    assert codec.inspect(checkpoint) == manifest
    (checkpoint / "state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="hash mismatch"):
        codec.inspect(checkpoint)


def test_policy_import_is_one_way_and_resets_training_state(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source = tmp_path / "v1.pt"
    torch.save(
        {
            "format": "firered-diffusionnft-v3",
            "policy_lora": {"layer.weight": torch.ones(2, 2)},
            "optimizer": {"must": "not be imported"},
            "history": [1, 2, 3],
        },
        source,
    )
    output = tmp_path / "imported"
    receipt = import_v1_policy(
        source=source,
        expected_source_sha256=sha256_file(source),
        source_release="training-code-v1.9.0",
        model_plugin_id="firered",
        allowed_keys=("policy_lora",),
        output=output,
    )
    assert receipt["resume_semantics"] == "policy_only_fork"
    assert receipt["optimizer_imported"] is False
    assert receipt["ema_imported"] is False
    assert json.loads((output / "receipt.json").read_text()) == receipt
    assert (output / "policy.safetensors").is_file()

    with pytest.raises(CheckpointError, match="source release"):
        import_v1_policy(
            source=source,
            expected_source_sha256=sha256_file(source),
            source_release="training-code-v1.8.0",
            model_plugin_id="firered",
            allowed_keys=("policy_lora",),
            output=tmp_path / "wrong-release",
        )
