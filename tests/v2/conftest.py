from __future__ import annotations

from mmtrainer.config import RunSpec


def run_spec_payload(*, launch_enabled: bool = False) -> dict[str, object]:
    digest = "a" * 64
    payload: dict[str, object] = {
        "schema_version": "mmtrainer.run-spec.v2",
        "run_id": "unit-run",
        "recipe_id": "unit-recipe",
        "recipe_version": "2.0.0",
        "source_release": "training-code-v2.0.0",
        "model": {
            "plugin_id": "firered",
            "assets": [{"asset_id": "model", "sha256": digest}],
        },
        "objective": {"plugin_id": "diffusionnft"},
        "dataset": {
            "plugin_id": "manifest",
            "assets": [{"asset_id": "dataset", "sha256": digest}],
        },
        "reward": {
            "plugin_id": "rational_rewards",
            "assets": [{"asset_id": "reward", "sha256": digest}],
        },
        "runtime": {
            "target_id": "local_h200",
            "hosts": ["local_h200"],
            "world_size": 1,
            "gpus_per_node": 8,
            "environment_id": "unit-cpu",
            "output_id": "unit-output",
        },
        "lifecycle": {"mode": "fresh", "max_updates": 1},
        "qualification": {
            "launch_enabled": launch_enabled,
            "runtime_receipt_sha256": digest if launch_enabled else None,
        },
    }
    return payload


def make_run_spec(*, launch_enabled: bool = False) -> RunSpec:
    return RunSpec.model_validate(run_spec_payload(launch_enabled=launch_enabled))
