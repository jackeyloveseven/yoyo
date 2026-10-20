"""文档/站点里写的 CLI 命令行,必须能被真实 argparse 接受(不执行)。

文档命令腐烂(写出的命令不存在、参数拼错、子命令已被重命名)是最常见的文档病,
且 `check_docs.py` 只查 markdown 链接、不查反引号里的 shell。本测试把文档与宣传片
脚本里出现的 `mmtrain ...` 命令行抽出来,用真实 CLI 解析器验证其参数存在——文档与
CLI 同源,是"结果可核对"红线的一部分。覆盖范围是行首命令(含站点 `<code>` 行)与
脚本字符串字面量;`SHA=$(mmtrain ...)` 这类 shell 内联写法暂不在内。

策略:把整条命令喂给 ``_parser().parse_args``——**只解析、不分发**。早先版本调用
``main(argv)``,argparse 校验完会继续 dispatch 真实业务代码(``registry list`` 真跑、
``phase run`` 在路径存在时真启动),文档一旦写成真实路径就会在测试里执行训练;
``--output`` 还需重定向以免落盘。改为直接调用解析器后,写盘、stdin、dispatch 三个
副作用同时消失。**不能追加 ``--help`` 短路**:argparse 遇到 ``--help`` 立刻退出,
看不到后面的未知参数。
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

from mmtrainer.cli import _parser

ROOT = Path(__file__).resolve().parents[2]

# 所有可能写着命令的对外文档面:根 *.md、docs/、站点、examples/README。
_DOC_SURFACES: tuple[Path, ...] = (
    *sorted(ROOT.glob("*.md")),
    *sorted((ROOT / "docs").rglob("*.md")),
    ROOT / "site" / "index.html",
    ROOT / "examples" / "README.md",
)

# 宣传片脚本里的演示命令同样是上镜材料:字符串字面量里的 mmtrain 命令也要能解析。
_SCRIPT_COMMAND_SOURCES: tuple[Path, ...] = (ROOT / "site" / "promo" / "render_promo.py",)


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
    for path in _SCRIPT_COMMAND_SOURCES:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'"(mmtrain [^"]+)"', text):
            command = re.sub(r"\s+#.*$", "", match.group(1)).strip()
            if not command.isascii():  # 标题文案(含中文)不是命令
                continue
            commands.append((path.relative_to(ROOT).as_posix(), command))
    return commands


def _parses(command: str) -> tuple[bool, str]:
    """命令(不含 'mmtrain' 前缀)能否被 argparse 接受。返回 (是否合法, 错误信息)。

    只做解析、不分发:``parse_args`` 合法即返回 Namespace,非法才抛 ``SystemExit``
    (退出码 2,usage 写 stderr)。因此不存在"文档写对了路径就真跑训练"的风险,
    也不必再重定向 ``--output`` 或伪造 stdin。
    """
    try:
        argv = shlex.split(command)[1:]
    except ValueError as exc:
        return False, f"无法 shlex: {exc}"
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            _parser().parse_args(argv)
        except SystemExit as exc:
            if exc.code in (0, None):
                return True, ""
            message = [ln for ln in stderr.getvalue().splitlines() if "error" in ln.lower()]
            return False, (message[-1].strip() if message else f"exit={exc.code}")
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
