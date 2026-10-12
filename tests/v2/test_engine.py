from __future__ import annotations

import pytest

from mmtrainer.config import RunSpec
from mmtrainer.engine import Phase, TrainingEngine
from mmtrainer.errors import CapabilityError, QualificationError
from mmtrainer.registry import build_builtin_registries
from mmtrainer.runtime import MemoryEventSink
from tests.v2.conftest import make_run_spec, run_spec_payload


def test_preflight_resolves_all_plugins_without_loading_weights() -> None:
    sink = MemoryEventSink()
    engine = TrainingEngine(make_run_spec(), build_builtin_registries(), event_sink=sink)
    result = engine.run(Phase.PREFLIGHT)
    assert result["model"]["runtime_status"] == "receipt_required"
    assert result["objective"]["plugin_id"] == "diffusionnft"
    assert [event for event, _ in sink.events] == [
        "phase_started",
        "phase_completed",
    ]


def test_train_is_disabled_before_runtime_receipt() -> None:
    engine = TrainingEngine(make_run_spec(), build_builtin_registries())
    with pytest.raises(QualificationError, match="disabled"):
        engine.run(Phase.TRAIN)


def test_capability_and_reward_mismatch_fail_before_import() -> None:
    payload = run_spec_payload()
    payload["model"]["plugin_id"] = "qwen_image"
    with pytest.raises(CapabilityError, match="lacks objective capabilities"):
        TrainingEngine(RunSpec.model_validate(payload), build_builtin_registries()).validate()

    payload = run_spec_payload()
    payload["reward"] = None
    with pytest.raises(CapabilityError, match="requires a reward"):
        TrainingEngine(RunSpec.model_validate(payload), build_builtin_registries()).validate()


def test_catalog_plugin_still_refuses_train_after_metadata_qualification() -> None:
    engine = TrainingEngine(make_run_spec(launch_enabled=True), build_builtin_registries())
    with pytest.raises(QualificationError, match="cannot load model weights"):
        engine.run(Phase.TRAIN)
