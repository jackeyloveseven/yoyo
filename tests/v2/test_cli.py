from __future__ import annotations

import json
import sys
from pathlib import Path

from mmtrainer.cli import main
from mmtrainer.config import RunSpec
from mmtrainer.registry import build_builtin_registries
from mmtrainer.version import __version__
from tests.v2.conftest import run_spec_payload


def test_core_import_does_not_load_torch() -> None:
    cached_torch = sys.modules.pop("torch", None)
    try:
        __import__("mmtrainer")
        assert "torch" not in sys.modules
    finally:
        if cached_torch is not None:
            sys.modules["torch"] = cached_torch


def test_package_version_matches_release_file() -> None:
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    assert __version__ == version_file.read_text(encoding="utf-8").strip()


def test_plan_validate_and_preflight_cli(tmp_path, capsys) -> None:
    payload = run_spec_payload()
    spec = RunSpec.model_validate(payload)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "validate", "--spec", str(path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["spec_sha256"] == spec.canonical_sha256()

    assert (
        main(
            [
                "phase",
                "run",
                "--spec",
                str(path),
                "--expected-sha256",
                spec.canonical_sha256(),
                "--phase",
                "preflight",
            ]
        )
        == 0
    )
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["model"]["plugin_id"] == "firered"


def test_registry_cli_lists_bounded_plugins(capsys) -> None:
    assert main(["registry", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # 模型清单随注册表动态演进,不在此处硬编码(避免每加一个插件改一次测试)。
    assert payload["models"] == sorted(
        entry.plugin_id for entry in build_builtin_registries().models.entries()
    )


def test_user_input_errors_exit_2(tmp_path, capsys) -> None:
    """用户输入类错误必须以 exit 2 收口,不能把 traceback 甩给用户。"""
    missing = tmp_path / "missing.json"
    assert main(["plan", "validate", "--spec", str(missing)]) == 2

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["plan", "validate", "--spec", str(broken)]) == 2

    assert main(["compose", "--model", "nope", "--objective", "image_sft", "--reward", "none"]) == 2

    streams = capsys.readouterr()
    assert streams.err.count("mmtrain:") == 3
    assert "Traceback" not in streams.err
