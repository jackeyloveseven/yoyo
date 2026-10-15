"""文档/站点里写的 CLI 命令行,必须能被真实 argparse 接受(不执行)。

文档命令腐烂(写出的命令不存在、参数拼错、子命令已被重命名)是最常见的文档病,
且 `check_docs.py` 只查 markdown 链接、不查反引号里的 shell。本测试把文档里出现的
每一条 `mmtrain ...` 命令行抽出来,用真实 CLI 解析器验证其参数存在——文档与 CLI
同源,是"结果可核对"红线的一部分。

策略:把整条命令喂给真实 CLI 解析器并捕获 ``SystemExit``——非法参数/子命令以非零
码退出,合法命令以 0 退出(argparse 在分发前先校验参数,不会执行真实命令)。**不能
追加 ``--help`` 短路**:argparse 遇到 ``--help`` 立刻退出,看不到后面的未知参数。
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import sys
import tempfile
from pathlib import Path

from mmtrainer.cli import main

ROOT = Path(__file__).resolve().parents[2]

# 所有可能写着命令的对外文档面:根 *.md、docs/、站点、examples/README。
_DOC_SURFACES: tuple[Path, ...] = (
    *sorted(ROOT.glob("*.md")),
    *sorted((ROOT / "docs").rglob("*.md")),
    ROOT / "site" / "index.html",
    ROOT / "examples" / "README.md",
)


def _documented_commands() -> list[tuple[str, str]]:
    """抽出文档里全部 `mmtrain ...` 命令行(已合并续行、剥掉行尾注释)。

    只有站点 ``index.html`` 需要剥 HTML 标签——markdown 代码块里的 ``<sha256>``
    这类占位符若被当 HTML 剥掉,会把占位符从命令里抹掉。
    """
    commands: list[tuple[str, str]] = []
    for path in _DOC_SURFACES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\\\n\s*", " ", text)  # 反斜杠续行
        strip_tags = path.suffix == ".html"
        for line in text.splitlines():
            if strip_tags:
                line = re.sub(r"<[^>]+>", "", line)
            stripped = line.strip().lstrip("$").strip()
            if not stripped.startswith("mmtrain "):
                continue
            stripped = re.sub(r"\s+#.*$", "", stripped).strip()
            commands.append((path.relative_to(ROOT).as_posix(), stripped))
    return commands


def _redirect_outputs(argv: list[str]) -> list[str]:
    """把 ``--output``/``-o`` 的目标改到临时文件,保住参数结构但消除落盘副作用。"""
    redirected: list[str] = []
    skip_next = False
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token in ("--output", "-o") and i + 1 < len(argv):
            suffix = Path(argv[i + 1]).suffix or ".out"
            redirected += [token, str(Path(tempfile.mkdtemp(prefix="cli-doc-")) / f"out{suffix}")]
            skip_next = True
        else:
            redirected.append(token)
    return redirected


def _parses(command: str) -> tuple[bool, str]:
    """命令(不含 'mmtrain' 前缀)能否被 argparse 接受。返回 (是否合法, 错误信息)。

    只关心**参数层面**的合法性(argparse 报错 = exit 2 或 stderr 含 "error"),
    不管业务逻辑——文档里大量命令指向占位路径(``/path/to/...``、
    ``training_code/recipes/...``),文件不存在是运行时的事,不该让这条测试红。
    给 ``--help`` 短路不行(argparse 遇 ``--help`` 立刻退出,看不到后面的未知参数);
    裸交互命令(``mmtrain compose``)需要 stdin,喂 EOF 让其以 SystemExit 退出。
    """
    try:
        argv = shlex.split(command)[1:]
    except ValueError as exc:
        return False, f"无法 shlex: {exc}"
    # 副作用隔离:把 --output <path> 重定向到临时目录,避免 compose --output blueprint.md
    # 这类合法 quickstart 命令在仓库根落盘产物(曾因此让 check_docs 把产物算进预算)。
    argv = _redirect_outputs(argv)
    stderr = io.StringIO()
    exit_code: int | None = None
    old_stdin = sys.stdin
    sys.stdin = io.StringIO("")  # 裸交互命令读到 EOF 即可退出
    try:
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            try:
                main(argv)
            except SystemExit as exc:
                exit_code = exc.code if isinstance(exc.code, int) else None
            except (OSError, FileNotFoundError):
                # 参数解析已通过,业务代码在打开文件——占位路径,视为参数层 OK
                return True, ""
            except EOFError:
                return True, ""
    finally:
        sys.stdin = old_stdin
    errors = [ln for ln in stderr.getvalue().splitlines() if "error" in ln.lower()]
    if exit_code not in (0, None) or errors:
        return False, (errors[-1].strip() if errors else f"exit={exit_code}")
    return True, ""


def test_documented_cli_commands_are_valid() -> None:
    """文档里写的每条 mmtrain 命令,参数都必须真实存在。"""
    commands = _documented_commands()
    assert commands, "文档里一条 mmtrain 命令都没抽到——抽取逻辑坏了?"
    bad: list[str] = []
    for source, command in commands:
        ok, why = _parses(command)
        if not ok:
            bad.append(f"{source}\n    $ {command}\n    → {why}")
    assert not bad, "以下文档命令行与 CLI 不一致:\n" + "\n".join(bad)
