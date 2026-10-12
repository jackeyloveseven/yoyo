from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mmtrainer.config import RunSpec, load_run_spec
from tests.v2.conftest import run_spec_payload


def test_run_spec_is_strict_and_hash_stable(tmp_path) -> None:
    payload = run_spec_payload()
    spec = RunSpec.model_validate(payload)
    path = tmp_path / "run-spec.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = load_run_spec(path)
    assert loaded == spec
    assert loaded.canonical_sha256() == spec.canonical_sha256()

    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunSpec.model_validate(payload)


def test_component_config_rejects_path_and_url_escape() -> None:
    payload = run_spec_payload()
    payload["model"]["config"] = {"model_path": "not-allowed"}
    with pytest.raises(ValidationError, match="registered assets"):
        RunSpec.model_validate(payload)

    payload = run_spec_payload()
    payload["reward"]["config"] = {"endpoint_url": "not-allowed"}
    with pytest.raises(ValidationError, match="registered assets"):
        RunSpec.model_validate(payload)


def test_policy_only_requires_source_and_fresh_forbids_it() -> None:
    payload = run_spec_payload()
    payload["lifecycle"] = {"mode": "policy_only", "max_updates": 1}
    with pytest.raises(ValidationError, match="requires a source policy"):
        RunSpec.model_validate(payload)

    payload = run_spec_payload()
    payload["lifecycle"]["source_policy"] = {
        "asset_id": "old-policy",
        "sha256": "b" * 64,
    }
    with pytest.raises(ValidationError, match="fresh lineage"):
        RunSpec.model_validate(payload)


def test_topology_and_qualification_fail_closed() -> None:
    payload = run_spec_payload()
    payload["runtime"]["world_size"] = 9
    with pytest.raises(ValidationError, match="GPU capacity"):
        RunSpec.model_validate(payload)

    payload = run_spec_payload()
    payload["qualification"]["launch_enabled"] = True
    with pytest.raises(ValidationError, match="qualification receipt"):
        RunSpec.model_validate(payload)
