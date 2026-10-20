"""Strict, hashable run contract compiled by WebTrainer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUN_SPEC_SCHEMA = "mmtrainer.run-spec.v2"
# pydantic 的 pattern= 走 re.search,不锚定就等于"包含即可";ID/SHA 必须全串匹配,
# 否则 "../../etc"、"zzz"+64 位 SHA+"zzz" 都能通过校验。
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"

JsonScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = JsonScalar | tuple[JsonScalar, ...]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetRef(StrictModel):
    """Content identity resolved to a local path by the trusted runtime agent."""

    asset_id: str = Field(pattern=ID_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=127)


class ComponentSpec(StrictModel):
    """One registry-selected component and its bounded scalar configuration."""

    plugin_id: str = Field(pattern=ID_PATTERN)
    plugin_version: str = Field(default="2.0.0", min_length=1, max_length=64)
    config: dict[str, ConfigValue] = Field(default_factory=dict)
    assets: tuple[AssetRef, ...] = ()

    @model_validator(mode="after")
    def validate_config_keys(self) -> ComponentSpec:
        for key, value in self.config.items():
            if re.fullmatch(ID_PATTERN, key) is None:
                raise ValueError(f"invalid component config key: {key}")
            lowered = key.lower()
            if any(token in lowered for token in ("path", "directory", "root", "url", "host")):
                raise ValueError(f"component config must reference registered assets, not {key}")
            # 只查 key 名不够:data_jsonl / note 这类名字照样能塞绝对路径或 URI,
            # 运行时代码会把 config 原样展开给数据集构造函数。
            for item in value if isinstance(value, tuple) else (value,):
                if isinstance(item, str) and _looks_like_location(item):
                    raise ValueError(
                        "component config value must be a scalar parameter, not a location: "
                        f"{key}={item!r}"
                    )
        return self


def _looks_like_location(value: str) -> bool:
    """绝对路径、父目录穿越或带 scheme 的 URI——这些必须走 AssetRef,不能藏在 config。"""
    return (
        value.startswith(("/", "~", "\\"))
        or ".." in Path(value).parts
        or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value) is not None
    )


class RuntimeSpec(StrictModel):
    target_id: str = Field(pattern=ID_PATTERN)
    hosts: tuple[str, ...] = Field(min_length=1, max_length=4)
    world_size: Annotated[int, Field(ge=1, le=32)]
    gpus_per_node: Annotated[int, Field(ge=1, le=8)]
    environment_id: str = Field(pattern=ID_PATTERN)
    output_id: str = Field(pattern=ID_PATTERN)
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    exclusive: bool = True

    @model_validator(mode="after")
    def validate_topology(self) -> RuntimeSpec:
        if len(set(self.hosts)) != len(self.hosts):
            raise ValueError("runtime hosts must be unique")
        if self.world_size > len(self.hosts) * self.gpus_per_node:
            raise ValueError("world_size exceeds the registered host GPU capacity")
        for host in self.hosts:
            if re.fullmatch(ID_PATTERN, host) is None:
                raise ValueError(f"invalid registered host id: {host}")
        return self


class LifecycleSpec(StrictModel):
    mode: Literal["fresh", "policy_only"] = "fresh"
    source_policy: AssetRef | None = None
    max_updates: PositiveInt
    checkpoint_every: PositiveInt = 1
    evaluation_every: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_lineage(self) -> LifecycleSpec:
        if self.mode == "fresh" and self.source_policy is not None:
            raise ValueError("fresh lineage cannot declare a source policy")
        if self.mode == "policy_only" and self.source_policy is None:
            raise ValueError("policy_only lineage requires a source policy")
        if self.checkpoint_every > self.max_updates:
            raise ValueError("checkpoint_every cannot exceed max_updates")
        if self.evaluation_every is not None and self.evaluation_every > self.max_updates:
            raise ValueError("evaluation_every cannot exceed max_updates")
        return self


class QualificationSpec(StrictModel):
    launch_enabled: bool = False
    promotion_qualified: bool = False
    runtime_receipt_sha256: str | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> QualificationSpec:
        if self.launch_enabled and self.runtime_receipt_sha256 is None:
            raise ValueError("launch_enabled requires a runtime qualification receipt")
        if self.promotion_qualified and not self.launch_enabled:
            raise ValueError("promotion_qualified requires launch_enabled")
        return self


class RunSpec(StrictModel):
    """The only mutable-input boundary accepted by Training Code v2."""

    schema_version: Literal["mmtrainer.run-spec.v2"] = RUN_SPEC_SCHEMA
    run_id: str = Field(pattern=ID_PATTERN)
    recipe_id: str = Field(pattern=ID_PATTERN)
    recipe_version: str = Field(min_length=1, max_length=64)
    source_release: str = Field(pattern=ID_PATTERN)
    model: ComponentSpec
    objective: ComponentSpec
    dataset: ComponentSpec
    reward: ComponentSpec | None = None
    runtime: RuntimeSpec
    lifecycle: LifecycleSpec
    qualification: QualificationSpec = Field(default_factory=QualificationSpec)

    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_run_spec(path: str | Path) -> RunSpec:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"run spec must not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"run spec is not a regular file: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return RunSpec.model_validate(payload)
