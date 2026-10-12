"""Versioned v2 checkpoint manifests and one-way v1 policy import."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mmtrainer.config import canonical_json_bytes
from mmtrainer.errors import CheckpointError

CHECKPOINT_SCHEMA = "mmtrainer.checkpoint.v2"
POLICY_IMPORT_SCHEMA = "mmtrainer.policy-import.v2"
V1_SOURCE_RELEASE = "training-code-v1.9.0"
V1_FULL_CHECKPOINT_FORMATS = frozenset(
    {
        "firered-diffusionnft-v1",
        "firered-diffusionnft-v2",
        "firered-diffusionnft-v3",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_v1_policy_state(
    payload: object,
    *,
    allowed_keys: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("v1 checkpoint root is not a mapping")
    if payload.get("format") not in V1_FULL_CHECKPOINT_FORMATS:
        raise CheckpointError("source is not a supported final-v1 full checkpoint")
    present = [key for key in allowed_keys if isinstance(payload.get(key), Mapping)]
    if len(present) != 1:
        raise CheckpointError("v1 checkpoint must expose exactly one registered policy state key")
    state = payload[present[0]]
    assert isinstance(state, Mapping)
    if not state or not all(isinstance(name, str) and name for name in state):
        raise CheckpointError("v1 policy state is empty or has invalid tensor names")
    return present[0], state


def import_v1_policy(
    *,
    source: str | Path,
    expected_source_sha256: str,
    source_release: str,
    model_plugin_id: str,
    allowed_keys: Sequence[str],
    output: str | Path,
) -> Mapping[str, Any]:
    if source_release != V1_SOURCE_RELEASE:
        raise CheckpointError(f"v2 policy import requires source release {V1_SOURCE_RELEASE}")
    requested = Path(source)
    if requested.is_symlink():
        raise CheckpointError("v1 checkpoint source must not be a symlink")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise CheckpointError("v1 checkpoint source is not a regular file")
    observed_sha256 = sha256_file(resolved)
    if observed_sha256 != expected_source_sha256:
        raise CheckpointError("v1 checkpoint SHA-256 mismatch")

    import torch
    from safetensors.torch import save_file

    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    state_key, raw_state = select_v1_policy_state(payload, allowed_keys=allowed_keys)
    tensors: dict[str, Any] = {}
    for name, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise CheckpointError(f"v1 policy value is not a tensor: {name}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise CheckpointError(f"v1 policy tensor is invalid: {name}")
        tensors[name] = value.detach().cpu().contiguous()

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise CheckpointError(f"policy import output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    policy_path = staging / "policy.safetensors"
    save_file(tensors, str(policy_path))
    receipt = {
        "schema_version": POLICY_IMPORT_SCHEMA,
        "source_release": source_release,
        "source_checkpoint_sha256": observed_sha256,
        "source_format": payload.get("format"),
        "source_state_key": state_key,
        "model_plugin_id": model_plugin_id,
        "policy_sha256": sha256_file(policy_path),
        "tensor_count": len(tensors),
        "resume_semantics": "policy_only_fork",
        "optimizer_imported": False,
        "ema_imported": False,
        "history_imported": False,
        "rng_imported": False,
        "data_cursor_imported": False,
    }
    (staging / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (staging / "COMPLETE").write_text(receipt["policy_sha256"] + "\n", encoding="utf-8")
    os.replace(staging, destination)
    return receipt


class V2DirectoryCheckpointCodec:
    plugin_id = "v2_directory"

    def inspect(self, path: Path) -> Mapping[str, Any]:
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or not (resolved / "COMPLETE").is_file():
            raise CheckpointError("v2 checkpoint is incomplete")
        manifest_path = resolved / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
            raise CheckpointError("unsupported v2 checkpoint schema")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or not files:
            raise CheckpointError("v2 checkpoint manifest has no files")
        for name, expected in files.items():
            candidate = resolved / name
            if candidate.parent != resolved or not candidate.is_file():
                raise CheckpointError(f"v2 checkpoint file is missing: {name}")
            if sha256_file(candidate) != expected:
                raise CheckpointError(f"v2 checkpoint hash mismatch: {name}")
        return manifest

    def save(self, path: Path, state: Mapping[str, Any]) -> Mapping[str, Any]:
        if path.exists() or path.is_symlink():
            raise CheckpointError(f"checkpoint output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
        try:
            state_path = staging / "state.json"
            state_path.write_bytes(canonical_json_bytes(dict(state)) + b"\n")
            manifest = {
                "schema_version": CHECKPOINT_SCHEMA,
                "files": {"state.json": sha256_file(state_path)},
            }
            (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
            (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(staging, path)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


class CompactParameterState:
    """Accelerator-compatible state holder that serializes only trainable or full BF16 weights."""

    def __init__(self, accelerator: Any, model: Any, *, trainable_only: bool) -> None:
        self.accelerator = accelerator
        self.model = model
        self.trainable_only = trainable_only

    def state_dict(self) -> dict[str, Any]:
        import torch

        raw_state = self.accelerator.get_state_dict(self.model)
        if not self.trainable_only:
            return {
                name: tensor.detach().cpu().to(dtype=torch.bfloat16)
                for name, tensor in raw_state.items()
            }
        unwrapped = self.accelerator.unwrap_model(self.model)
        trainable_names = {
            name for name, param in unwrapped.named_parameters() if param.requires_grad
        }
        return {
            name: tensor.detach().cpu().to(dtype=torch.bfloat16)
            for name, tensor in raw_state.items()
            if name in trainable_names
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise CheckpointError(
                f"Unexpected keys when restoring compact parameter state: {unexpected[:5]}"
            )
        if not self.trainable_only and missing:
            raise CheckpointError(
                f"Missing keys when restoring full parameter state: {missing[:5]}"
            )


def register_compact_parameter_state(
    accelerator: Any,
    model: Any,
    *,
    trainable_only: bool,
) -> CompactParameterState:
    state = CompactParameterState(accelerator, model, trainable_only=trainable_only)
    accelerator.register_for_checkpointing(state)
    return state
