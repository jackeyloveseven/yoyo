"""Ratcheting architecture checks for the v2 source tree."""

from __future__ import annotations

import ast
import hashlib
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

HARD_FILE_LINES = 2_000
CORE_FILE_LINES = 800
PLUGIN_FILE_LINES = 1_200
FUNCTION_LINES = 120
FUNCTION_ARGS = 8
CYCLOMATIC_COMPLEXITY = 15
NON_TEST_LOC_BUDGET = 103_748
DUPLICATE_TOKEN_WINDOW = 50
CORE_DIRECTORIES = frozenset({"config", "engine", "interfaces", "quality", "registry", "runtime"})
PLUGIN_DIRECTORIES = frozenset({"algorithms", "datasets", "models", "nn", "plugins", "rewards"})
FORBIDDEN_IMPORTS = frozenset({"paramiko", "subprocess"})
HOME_PATH = re.compile(r"/home/(?:ubuntu|jackeylove)/")


@dataclass(frozen=True)
class QualityReport:
    files: int
    non_test_python_loc: int
    argparse_files: int
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def _python_files(source_root: Path) -> tuple[Path, ...]:
    return tuple(
        path for path in sorted(source_root.rglob("*.py")) if "__pycache__" not in path.parts
    )


def _line_limit(relative: Path) -> int:
    head = relative.parts[1] if len(relative.parts) > 1 else ""
    if head in CORE_DIRECTORIES or len(relative.parts) == 2:
        return CORE_FILE_LINES
    if head in PLUGIN_DIRECTORIES:
        return PLUGIN_FILE_LINES
    return HARD_FILE_LINES


def _complexity(node: ast.AST) -> int:
    branch_types = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Match,
        ast.BoolOp,
        ast.comprehension,
    )
    return 1 + sum(isinstance(child, branch_types) for child in ast.walk(node))


def _argument_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    return (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
        + int(args.vararg is not None)
        + int(args.kwarg is not None)
    )


def _module_name(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((source_root.name, *parts))


def _internal_imports(tree: ast.Module) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("mmtrainer"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("mmtrainer"):
                imports.add(node.module)
    return imports


def _dependency_graph(
    source_root: Path,
    parsed: dict[Path, ast.Module],
) -> dict[str, set[str]]:
    modules = {_module_name(source_root, path): path for path in parsed}
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        for imported in _internal_imports(parsed[path]):
            candidates = [
                candidate
                for candidate in modules
                if imported == candidate or imported.startswith(candidate + ".")
            ]
            if candidates:
                target = max(candidates, key=len)
                if target != module:
                    graph[module].add(target)
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for node in graph:
        if node not in indices:
            visit(node)
    return components


def _normalized_tokens(path: Path) -> list[str]:
    result: list[str] = []
    with path.open("rb") as handle:
        tokens = tokenize.tokenize(handle.readline)
        for token in tokens:
            if token.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.COMMENT,
            }:
                continue
            value = token.string
            if token.type == tokenize.STRING:
                value = "STRING"
            elif token.type == tokenize.NUMBER:
                value = "NUMBER"
            result.append(value)
    return result


def _duplicate_windows(files: tuple[Path, ...]) -> list[tuple[Path, Path]]:
    owners: dict[bytes, Path] = {}
    duplicates: set[tuple[Path, Path]] = set()
    for path in files:
        tokens = _normalized_tokens(path)
        local: set[bytes] = set()
        for offset in range(max(0, len(tokens) - DUPLICATE_TOKEN_WINDOW + 1)):
            window = "\x1f".join(tokens[offset : offset + DUPLICATE_TOKEN_WINDOW]).encode("utf-8")
            digest = hashlib.blake2b(window, digest_size=16).digest()
            if digest in local:
                continue
            local.add(digest)
            owner = owners.setdefault(digest, path)
            if owner != path:
                duplicates.add(tuple(sorted((owner, path))))
    return sorted(duplicates)


def _import_violations(
    relative: Path,
    tree: ast.Module,
) -> tuple[list[str], bool]:
    violations: list[str] = []
    argparse_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            argparse_imported |= "argparse" in names
            forbidden = sorted(names & FORBIDDEN_IMPORTS)
            if forbidden:
                violations.append(f"forbidden-runtime-import:{relative}:{','.join(forbidden)}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            argparse_imported |= node.module == "argparse"
            if node.module in FORBIDDEN_IMPORTS:
                violations.append(f"forbidden-runtime-import:{relative}:{node.module}")
    return violations, argparse_imported


def _function_violations(relative: Path, tree: ast.Module) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if function_lines > FUNCTION_LINES:
                violations.append(
                    f"function-lines:{relative}:{node.lineno}:{node.name}:"
                    f"{function_lines}>{FUNCTION_LINES}"
                )
            arguments = _argument_count(node)
            if arguments > FUNCTION_ARGS:
                violations.append(
                    f"function-args:{relative}:{node.lineno}:{node.name}:"
                    f"{arguments}>{FUNCTION_ARGS}"
                )
            complexity = _complexity(node)
            if complexity > CYCLOMATIC_COMPLEXITY:
                violations.append(
                    f"function-complexity:{relative}:{node.lineno}:{node.name}:"
                    f"{complexity}>{CYCLOMATIC_COMPLEXITY}"
                )
    return violations


def _is_broad_handler(node: ast.ExceptHandler) -> bool:
    return node.type is None or (
        isinstance(node.type, ast.Name) and node.type.id in {"BaseException", "Exception"}
    )


def _has_sys_path_mutation(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"append", "insert"}:
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Attribute) or owner.attr != "path":
            continue
        if isinstance(owner.value, ast.Name) and owner.value.id == "sys":
            return True
    return False


def _audit_file(
    source_root: Path,
    path: Path,
    tree: ast.Module,
    text: str,
) -> tuple[list[str], bool]:
    relative = path.relative_to(source_root.parent)
    lines = text.splitlines()
    limit = min(HARD_FILE_LINES, _line_limit(relative))
    violations: list[str] = []
    if len(lines) > limit:
        violations.append(f"file-lines:{relative}:{len(lines)}>{limit}")
    if HOME_PATH.search(text):
        violations.append(f"absolute-home-path:{relative}")
    if _has_sys_path_mutation(tree):
        violations.append(f"sys-path-mutation:{relative}")
    import_violations, argparse_imported = _import_violations(relative, tree)
    violations.extend(import_violations)
    violations.extend(_function_violations(relative, tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _is_broad_handler(node):
            violations.append(f"broad-except:{relative}:{node.lineno}")
    return violations, argparse_imported


# 同家族版本演进的显式豁免:键 = "子版本模块",值 = 它被允许依赖的家族基线模块。
# 仅用于"同一模型的新版本适配器复用旧版函数"(如 Qwen-Image-Edit 2.x 复用原版
# VAE/pack/前向协议)。每条豁免必须在 docs/adr/ 有对应 ADR 记录理由,防止
# 白名单沦为绕过架构门的后门。
MODEL_FAMILY_EXEMPTIONS: dict[str, frozenset[str]] = {
    "qwen_image_edit2": frozenset({"qwen_image"}),
}


def audit_source_tree(source_root: str | Path) -> QualityReport:
    root = Path(source_root).resolve(strict=True)
    if root.name != "mmtrainer":
        raise ValueError("quality root must be the mmtrainer package directory")
    files = _python_files(root)
    violations: list[str] = []
    parsed: dict[Path, ast.Module] = {}
    argparse_files = 0
    loc = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        loc += len(text.splitlines())
        tree = ast.parse(text, filename=str(path))
        parsed[path] = tree
        file_violations, argparse_imported = _audit_file(root, path, tree, text)
        violations.extend(file_violations)
        argparse_files += int(argparse_imported)
    if argparse_files > 1:
        violations.append(f"argparse-boundaries:{argparse_files}>1")
    if loc > NON_TEST_LOC_BUDGET:
        violations.append(f"non-test-python-loc:{loc}>{NON_TEST_LOC_BUDGET}")

    graph = _dependency_graph(root, parsed)
    for cycle in _cycles(graph):
        violations.append("dependency-cycle:" + ",".join(cycle))
    for module, imports in graph.items():
        if not module.startswith("mmtrainer.models."):
            continue
        owner = module.split(".", 2)[2].split(".", 1)[0]
        exempt = MODEL_FAMILY_EXEMPTIONS.get(owner, frozenset())
        for imported in imports:
            if imported.startswith("mmtrainer.models."):
                target = imported.split(".", 2)[2].split(".", 1)[0]
                if target != owner and target not in exempt:
                    violations.append(f"cross-model-import:{module}->{imported}")

    duplicate_candidates = tuple(
        path for path in files if path.name not in {"__init__.py", "__main__.py"}
    )
    for left, right in _duplicate_windows(duplicate_candidates):
        violations.append(
            "duplicate-token-window:"
            f"{left.relative_to(root.parent)}:{right.relative_to(root.parent)}"
        )
    return QualityReport(
        files=len(files),
        non_test_python_loc=loc,
        argparse_files=argparse_files,
        violations=tuple(sorted(set(violations))),
    )
