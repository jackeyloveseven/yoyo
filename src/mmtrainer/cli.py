"""The single command-line boundary for Training Code v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mmtrainer.config import load_run_spec
from mmtrainer.engine import Phase, TrainingEngine
from mmtrainer.errors import MMTrainerError
from mmtrainer.registry import build_builtin_registries
from mmtrainer.runtime import import_v1_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmtrain")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan_commands = plan.add_subparsers(dest="plan_command", required=True)
    validate = plan_commands.add_parser("validate")
    validate.add_argument("--spec", required=True, type=Path)

    phase = commands.add_parser("phase")
    phase_commands = phase.add_subparsers(dest="phase_command", required=True)
    run = phase_commands.add_parser("run")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--expected-sha256", required=True)
    run.add_argument("--phase", required=True, choices=[item.value for item in Phase])

    checkpoint = commands.add_parser("checkpoint")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    import_policy = checkpoint_commands.add_parser("import-policy")
    import_policy.add_argument("--source", required=True, type=Path)
    import_policy.add_argument("--source-sha256", required=True)
    import_policy.add_argument("--source-release", required=True)
    import_policy.add_argument("--model", required=True)
    import_policy.add_argument("--output", required=True, type=Path)

    registry = commands.add_parser("registry")
    registry_commands = registry.add_subparsers(dest="registry_command", required=True)
    registry_commands.add_parser("list")

    compose = commands.add_parser("compose")
    compose.add_argument("--model", help="model plugin id (skip interactive prompt)")
    compose.add_argument("--objective", help="objective plugin id (skip interactive prompt)")
    compose.add_argument("--reward", help="reward plugin id (skip interactive prompt)")
    compose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the blueprint to a .json or .md file instead of only printing",
    )
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _validate_plan(spec_path: Path) -> int:
    spec = load_run_spec(spec_path)
    result = TrainingEngine(spec, build_builtin_registries()).validate()
    _print(result)
    return 0


def _run_phase(spec_path: Path, expected_sha256: str, phase: str) -> int:
    spec = load_run_spec(spec_path)
    if spec.canonical_sha256() != expected_sha256:
        raise ValueError("run spec SHA-256 differs from the immutable execution plan")
    result = TrainingEngine(spec, build_builtin_registries()).run(Phase(phase))
    _print(result)
    return 0


def _import_policy(args: argparse.Namespace) -> int:
    registries = build_builtin_registries()
    model_type = registries.models.resolve(args.model)
    receipt = import_v1_policy(
        source=args.source,
        expected_source_sha256=args.source_sha256,
        source_release=args.source_release,
        model_plugin_id=args.model,
        allowed_keys=model_type.v1_policy_keys,
        output=args.output,
    )
    _print(receipt)
    return 0


def _list_registry() -> int:
    registries = build_builtin_registries()
    payload = {
        "models": [entry.plugin_id for entry in registries.models.entries()],
        "objectives": [entry.plugin_id for entry in registries.objectives.entries()],
        "rewards": [entry.plugin_id for entry in registries.rewards.entries()],
        "datasets": [entry.plugin_id for entry in registries.datasets.entries()],
        "checkpoints": [entry.plugin_id for entry in registries.checkpoints.entries()],
    }
    _print(payload)
    return 0


def _pick(prompt: str, options: list[str], preset: str | None) -> str:
    """Return ``preset`` if valid, else interactively ask the user to pick one."""

    if preset is not None:
        if preset not in options:
            raise ValueError(f"{preset!r} is not one of: {', '.join(options)}")
        return preset
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        raw = input(f"选择 [1-{len(options)}] (回车默认 1): ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        print(f"无效输入,请输入 1-{len(options)} 或插件 id。")


def _run_compose(args: argparse.Namespace) -> int:
    from mmtrainer.compose import (
        ComposeChoice,
        build_blueprint,
        render_blueprint_markdown,
    )
    from mmtrainer.compose.compose import compatible_objectives, compatible_rewards

    registries = build_builtin_registries()
    model_options = sorted(entry.plugin_id for entry in registries.models.entries())
    model_id = _pick("① 选择模型:", model_options, args.model)

    objective_options = compatible_objectives(registries, model_id)
    objective_id = _pick(f"② 选择目标(模型 {model_id} 可跑):", objective_options, args.objective)

    reward_options = compatible_rewards(registries, objective_id)
    reward_id = _pick("③ 选择 reward:", reward_options, args.reward)

    blueprint = build_blueprint(
        registries,
        ComposeChoice(model_id=model_id, objective_id=objective_id, reward_id=reward_id),
    )
    output: Path | None = args.output
    if output is None:
        _print(blueprint)
        return 0
    suffix = output.suffix.lower()
    if suffix == ".json":
        output.write_text(
            json.dumps(blueprint, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif suffix in (".md", ".markdown"):
        output.write_text(render_blueprint_markdown(blueprint), encoding="utf-8")
    else:
        raise ValueError(f"--output 只支持 .json 或 .md,收到: {output.name}")
    print(f"蓝图已写入 {output}")
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "plan" and args.plan_command == "validate":
        return _validate_plan(args.spec)
    if args.command == "phase" and args.phase_command == "run":
        return _run_phase(args.spec, args.expected_sha256, args.phase)
    if args.command == "checkpoint" and args.checkpoint_command == "import-policy":
        return _import_policy(args)
    if args.command == "registry" and args.registry_command == "list":
        return _list_registry()
    if args.command == "compose":
        return _run_compose(args)
    raise AssertionError("argparse returned an unregistered command")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return dispatch(args)
    except MMTrainerError as error:
        print(f"mmtrain: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError, EOFError) as error:
        # 用户输入类错误(spec 不存在/JSON 语法错/schema 校验失败/参数不匹配)
        # 必须以 exit 2 收口,不能把 traceback 甩给用户。ValidationError 与
        # JSONDecodeError 都是 ValueError 的子类,FileNotFoundError 是 OSError。
        print(f"mmtrain: {error}", file=sys.stderr)
        return 2
